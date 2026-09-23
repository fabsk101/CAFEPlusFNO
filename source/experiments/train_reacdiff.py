"""Audited, offline PDEBench Reaction-Diffusion-1024 paper experiment.

The numerical path reproduces the pinned SirenFNO ``train_ReacDiff.py``
runtime. Acquisition is intentionally separated: this entry point accepts
only the locally verified public HDF5 artifact and has no network fallback.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.common.dataset_provenance import verify_dataset
from experiments.common.sirenfno_backend import (
    REPOSITORY_ROOT,
    audit_runtime_sources,
    bootstrap_sirenfno_backend,
    source_repository_state,
    upstream_manifest,
)

BACKEND_PROVENANCE = bootstrap_sirenfno_backend()

import h5py
import neuralop
import neuralop.utils as neuralop_utils
import numpy as np
import torch
from experiments.common.checkpoints import (
    normalize_environment_metadata,
    prepare_weights_only_checkpoint,
)
from experiments.common.evaluation import (
    corrected_relative_l2_metadata,
    evaluate_corrected_relative_l2,
)
from baseline.AMFNO import FNO1dMLP
from baseline.UFNO import UFNO1d
from neuralop.losses import LpLoss
from neuralop.models import FNO1d
from neuralop.utils import (
    count_model_params,
    rollout_loss_lp,
    rollout_step_model_1d_channel_first,
)
from SirenFNO1D import SirenFNO1d
from torch.utils.data import DataLoader
from utils import RolloutRAMDataset, _find_data_dataset, load_subset_to_ram
import train_ReacDiff as train_reacdiff_reference

from experiments.common.seed import set_seed
from experiments.configs.reacdiff1d import (
    AUTHOR_COMMENTED_TIME_STEPS,
    BATCH_SIZE,
    CAFEPLUSFNO_FACTORIZATIONS,
    CANONICAL_SEED,
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_NAME,
    DATASET_SOURCE,
    EPOCHS,
    EVAL_DROP_LAST,
    EVAL_INTERVAL,
    EXPECTED_CAFE_PARAMETER_COUNTS,
    EXPECTED_PINNED_PARAMETER_COUNTS,
    EXPECTED_TIME_STEPS,
    EXPERIMENT_ID,
    FIELD_DIM,
    INPUT_DIM,
    INPUT_STEPS,
    LEARNING_RATE,
    MAIN_REPORTING_METRIC,
    MODEL_CHOICES,
    NORMALIZATION_POLICY,
    N_TEST,
    N_TRAIN,
    N_VAL,
    NUM_WORKERS,
    OUTPUT_DIM,
    PHYSICAL_CHANNELS,
    PIN_MEMORY,
    PUSHFORWARD_DETACH,
    REACDIFF_SOURCE_BLOBS,
    REDUCE_T,
    REDUCE_X,
    RESOLUTION,
    ROLLOUT,
    SCHEDULER_T_MAX,
    SECONDARY_REPORTING_METRICS,
    SEED_POLICY,
    SELECTION_POLICY,
    SPLIT_POLICY,
    TRAIN_DROP_LAST,
    TEMPORAL_PROVENANCE_NOTE,
    USE_AMP,
    WEIGHT_DECAY,
    model_configuration_provenance,
    model_constructor_kwargs,
)
from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D


@dataclass(frozen=True)
class ModelBuild:
    """One model and its fully resolved reconstructable configuration."""

    model: torch.nn.Module
    configuration: dict[str, Any]
    factorization: str | None
    rank: int | float | None


@dataclass(frozen=True)
class ReacDiffData:
    """The three released-style loaders and sanitized source metadata."""

    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_eval_loader: DataLoader
    total_trajectories: int
    time_steps: int
    resolution: int
    dataset_key: str


_RFF_MODEL_NAMES = frozenset(
    {
        "sirenfno",
        "cpsirenfno",
        "ttsirenfno",
        "tuckersirenfno",
        "cafe_plus_fno",
        "cp_cafe_plus_fno",
        "tt_cafe_plus_fno",
        "tucker_cafe_plus_fno",
    }
)
_NON_RFF_MODEL_NAMES = frozenset({"fno", "tfno_cp", "amfno", "ufno"})


def author_local_import_audit() -> dict[str, str]:
    """Validate actual runtime sources plus the pinned reference pipeline."""

    return audit_runtime_sources(
        official_components={
            "neuralop": neuralop,
            "FNO1d": FNO1d,
            "LpLoss": LpLoss,
            "count_model_params": count_model_params,
            "neuralop.utils(runtime)": neuralop_utils,
            "rollout_step_model_1d_channel_first": (
                rollout_step_model_1d_channel_first
            ),
            "rollout_loss_lp": rollout_loss_lp,
            "RolloutRAMDataset": RolloutRAMDataset,
            "_find_data_dataset": _find_data_dataset,
            "load_subset_to_ram": load_subset_to_ram,
            "train_ReacDiff(reference_only)": train_reacdiff_reference,
            "SirenFNO1d": SirenFNO1d,
            "FNO1dMLP": FNO1dMLP,
            "UFNO1d": UFNO1d,
        },
        proposed_components={"CAFEPlusFNO1D": CAFEPlusFNO1D},
    )


AUTHOR_LOCAL_IMPORT_PATHS = author_local_import_audit()


def rff_metadata(model_name: str) -> dict[str, str]:
    if model_name in _RFF_MODEL_NAMES:
        return {
            "rff_rng_policy": "global_torch_rng",
            "rff_seed_source": "global_experiment_seed",
        }
    if model_name in _NON_RFF_MODEL_NAMES:
        return {
            "rff_rng_policy": "not_applicable",
            "rff_seed_source": "not_applicable",
        }
    raise KeyError(f"Unknown ReacDiff model for RFF metadata: {model_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one released ReacDiff model for one seed."
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "reacdiff1024_sirenfno_81918ec",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _validate_cafe_configuration(
    model_name: str,
    requested: dict[str, object],
    resolved: dict[str, object],
) -> tuple[str, int]:
    for key, requested_value in requested.items():
        expected_value = requested_value
        if key == "padding" and isinstance(requested_value, int):
            expected_value = (requested_value,)
        if resolved.get(key) != expected_value:
            raise RuntimeError(
                "CAFE+FNO constructor/config parity failed for "
                f"{key!r}: requested {requested_value!r}, "
                f"resolved {resolved.get(key)!r}."
            )
    expected_variant = CAFEPLUSFNO_FACTORIZATIONS[model_name]
    factorization = str(resolved["factorization"])
    rank = int(resolved["rank"])
    if factorization != expected_variant["factorization"]:
        raise RuntimeError("CAFE+FNO resolved to the wrong factorization.")
    if rank != expected_variant["rank"]:
        raise RuntimeError("CAFE+FNO resolved to the wrong ReacDiff rank.")
    return factorization, rank


def build_model(model_name: str, device: torch.device) -> ModelBuild:
    """Construct one exact released baseline or precommitted CAFE model."""

    name = model_name.lower()
    kwargs = model_constructor_kwargs(name)
    if name in {"fno", "tfno_cp"}:
        model = FNO1d(**kwargs)
        factorization = kwargs.get("factorization")
        rank = kwargs.get("rank")
    elif name == "amfno":
        model = FNO1dMLP(**kwargs)
        factorization = None
        rank = None
    elif name == "ufno":
        model = UFNO1d(**kwargs)
        factorization = None
        rank = None
    elif name in {
        "sirenfno",
        "cpsirenfno",
        "ttsirenfno",
        "tuckersirenfno",
    }:
        model = SirenFNO1d(**kwargs)
        factorization = str(kwargs["factorization"])
        rank = kwargs.get("rank")
    elif name in CAFEPLUSFNO_FACTORIZATIONS:
        model = CAFEPlusFNO1D(**kwargs)
        configuration = model.get_config()
        factorization, rank = _validate_cafe_configuration(
            name, kwargs, configuration
        )
        return ModelBuild(
            model=model.to(device),
            configuration=configuration,
            factorization=factorization,
            rank=rank,
        )
    else:
        raise ValueError(f"Unsupported ReacDiff model: {model_name}")
    return ModelBuild(
        model=model.to(device),
        configuration=kwargs,
        factorization=factorization,
        rank=rank,
    )


def expected_reacdiff_file(data_root: Path) -> Path:
    return data_root / DATASET_FILENAME


def verify_reacdiff_dataset(data_root: Path) -> dict[str, Any]:
    """Fail closed unless the exact public HDF5 artifact is verified."""

    try:
        return verify_dataset(DATASET_ID, data_root)
    except (FileNotFoundError, RuntimeError):
        raise RuntimeError(
            "Reaction-Diffusion data is missing or failed checksum "
            "verification.\nPrepare it with:\n"
            "python -B data/download_data.py --dataset reacdiff1024 "
            "--data-root data"
        ) from None


def fixed_split_indices(total_trajectories: int) -> tuple[np.ndarray, np.ndarray]:
    required = N_TRAIN + N_VAL + N_TEST
    if total_trajectories < required:
        raise ValueError(
            f"ReacDiff split requires {required} trajectories, got "
            f"{total_trajectories}."
        )
    all_indices = np.arange(total_trajectories)
    return (
        all_indices[:N_TRAIN],
        all_indices[N_TRAIN + N_VAL : required],
    )


def inspect_source_data(path: Path) -> tuple[int, int, int, str]:
    """Resolve the released helper's unique primary dataset and shape."""

    try:
        with h5py.File(path, "r") as handle:
            dataset = _find_data_dataset(handle)
            if dataset.ndim != 3:
                raise RuntimeError("The ReacDiff dataset must have rank three.")
            total, time_steps, resolution = (
                int(value) for value in dataset.shape
            )
            key = dataset.name
    except (OSError, RuntimeError, ValueError):
        raise RuntimeError(
            "The verified local Reaction-Diffusion HDF5 file is invalid."
        ) from None
    return total, time_steps, resolution, key


def build_data_loaders(
    train_cpu: torch.Tensor,
    test_cpu: torch.Tensor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct DataLoaders with the released author's exact options."""

    train_dataset = RolloutRAMDataset(
        train_cpu, INPUT_STEPS, ROLLOUT, None, None
    )
    train_eval_dataset = RolloutRAMDataset(
        train_cpu, INPUT_STEPS, ROLLOUT, None, None
    )
    test_eval_dataset = RolloutRAMDataset(
        test_cpu, INPUT_STEPS, ROLLOUT, None, None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=TRAIN_DROP_LAST,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=EVAL_DROP_LAST,
    )
    test_eval_loader = DataLoader(
        test_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=EVAL_DROP_LAST,
    )
    return train_loader, train_eval_loader, test_eval_loader


def load_reacdiff(data_root: Path) -> ReacDiffData:
    """Load the verified source through the pinned offline-only RAM path."""

    data_path = expected_reacdiff_file(data_root)
    if not data_path.is_file():
        raise FileNotFoundError(
            "Reaction-Diffusion data is missing or failed checksum "
            "verification.\nPrepare it with:\n"
            "python -B data/download_data.py --dataset reacdiff1024 "
            "--data-root data"
        )
    if (REDUCE_X, REDUCE_T) != (1, 1):
        raise RuntimeError("ReacDiff direct loading requires reduce_x=reduce_t=1.")
    total, time_steps, resolution, dataset_key = inspect_source_data(data_path)
    if time_steps != EXPECTED_TIME_STEPS or resolution != RESOLUTION:
        raise RuntimeError(
            "The local ReacDiff dataset has unexpected temporal or spatial shape."
        )
    if time_steps < INPUT_STEPS + ROLLOUT:
        raise ValueError("The ReacDiff time dimension is shorter than 10-to-10.")
    train_indices, test_indices = fixed_split_indices(total)
    train_cpu = load_subset_to_ram(str(data_path), train_indices)
    test_cpu = load_subset_to_ram(str(data_path), test_indices)
    expected_tail = (time_steps, resolution, PHYSICAL_CHANNELS)
    if tuple(train_cpu.shape[1:]) != expected_tail:
        raise RuntimeError("The pinned loader produced invalid ReacDiff training data.")
    if tuple(test_cpu.shape[1:]) != expected_tail:
        raise RuntimeError("The pinned loader produced invalid ReacDiff test data.")
    loaders = build_data_loaders(train_cpu, test_cpu)
    return ReacDiffData(
        train_loader=loaders[0],
        train_eval_loader=loaders[1],
        test_eval_loader=loaders[2],
        total_trajectories=total,
        time_steps=time_steps,
        resolution=resolution,
        dataset_key=dataset_key,
    )


def validate_model_parameter_count(model_name: str, count: int) -> None:
    expected = {
        **EXPECTED_PINNED_PARAMETER_COUNTS,
        **EXPECTED_CAFE_PARAMETER_COUNTS,
    }.get(model_name)
    if expected is None or count != expected:
        raise RuntimeError(
            f"ReacDiff {model_name} parameter count changed: expected "
            f"{expected}, got {count}."
        )


def rollout_reacdiff(
    model: torch.nn.Module,
    x_hist: torch.Tensor,
    steps: int,
    pushforward_detach: bool = True,
) -> torch.Tensor:
    return rollout_step_model_1d_channel_first(
        model,
        x_hist,
        steps,
        pushforward_detach=pushforward_detach,
        field_dim=FIELD_DIM,
    )


LOG_FIELDNAMES = (
    "epoch",
    "train_corrected_step_relative_l2",
    "train_corrected_trajectory_relative_l2",
    "test_corrected_step_relative_l2",
    "test_corrected_trajectory_relative_l2",
    "train_evaluation_sample_count",
    "test_evaluation_sample_count",
    "evaluation_horizon",
    "learning_rate",
    "train_time_seconds",
    "epoch_time_seconds",
)


def run_training_loop(
    *,
    model: torch.nn.Module,
    data: ReacDiffData,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    csv_path: Path,
) -> list[dict[str, float | int]]:
    """Run the released epoch/batch/scheduler/evaluation order exactly."""

    # Preserve the released training objective; corrected reporting metrics do
    # not call this mean-reduced LpLoss.
    lp_rel_train = LpLoss(d=1, p=2, reduction="mean")
    history: list[dict[str, float | int]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_start = time.perf_counter()
            for x_batch, y_batch in data.train_loader:
                x_batch = x_batch.to(device, non_blocking=True).float()
                y_batch = y_batch.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                prediction = rollout_reacdiff(
                    model,
                    x_batch,
                    ROLLOUT,
                    pushforward_detach=PUSHFORWARD_DETACH,
                )
                loss = rollout_loss_lp(prediction, y_batch, lp_rel_train)
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start
            scheduler.step()
            train_metrics = evaluate_corrected_relative_l2(
                model,
                data.train_eval_loader,
                horizon=ROLLOUT,
                rollout_fn=rollout_reacdiff,
                device=device,
                time_dim=1,
                channel_dim=2,
            )
            test_metrics = evaluate_corrected_relative_l2(
                model,
                data.test_eval_loader,
                horizon=ROLLOUT,
                rollout_fn=rollout_reacdiff,
                device=device,
                time_dim=1,
                channel_dim=2,
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_corrected_step_relative_l2": (
                    train_metrics.corrected_step_relative_l2
                ),
                "train_corrected_trajectory_relative_l2": (
                    train_metrics.corrected_trajectory_relative_l2
                ),
                "test_corrected_step_relative_l2": (
                    test_metrics.corrected_step_relative_l2
                ),
                "test_corrected_trajectory_relative_l2": (
                    test_metrics.corrected_trajectory_relative_l2
                ),
                "train_evaluation_sample_count": train_metrics.sample_count,
                "test_evaluation_sample_count": test_metrics.sample_count,
                "evaluation_horizon": test_metrics.horizon,
                "learning_rate": epoch_learning_rate,
                "train_time_seconds": train_time,
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch:03d}/{EPOCHS:03d} | "
                f"T'={data.time_steps} X'={data.resolution} | "
                f"train_time {train_time:.2f}s | "
                "step_rel(train/test) "
                f"{train_metrics.corrected_step_relative_l2:.10f}/"
                f"{test_metrics.corrected_step_relative_l2:.10f} | "
                "trajectory_rel(train/test) "
                f"{train_metrics.corrected_trajectory_relative_l2:.10f}/"
                f"{test_metrics.corrected_trajectory_relative_l2:.10f}"
            )
    return history


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(source_state: dict[str, Any]) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return normalize_environment_metadata({
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "tensorly_version": _package_version("tensorly"),
        "tensorly_torch_version": _package_version("tensorly-torch"),
        "neuraloperator_version": str(neuralop.__version__),
        "neuraloperator_source": BACKEND_PROVENANCE["neuraloperator_source"],
        "neuraloperator_source_id": (
            f"SirenFNO@{BACKEND_PROVENANCE['commit']}:neuralop"
        ),
        "neuraloperator_git_tree_sha1": BACKEND_PROVENANCE[
            "neuraloperator_git_tree_sha1"
        ],
        "sirenfno_upstream_commit": BACKEND_PROVENANCE["commit"],
        "implementation_import_paths": AUTHOR_LOCAL_IMPORT_PATHS,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name() if cuda_available else None,
        **source_state,
    })


def public_run_label(run_dir: Path, model_name: str, seed: int) -> str:
    try:
        return run_dir.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external-results-root>/{model_name}/seed_{seed}"


def prepare_run_paths(
    results_root: Path, model_name: str, seed: int, overwrite: bool
) -> tuple[Path, Path, Path, Path]:
    run_dir = results_root / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "final_checkpoint.pt"
    existing = [
        path for path in (csv_path, summary_path, checkpoint_path) if path.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Run output already exists for {model_name}/seed_{seed}: {names}. "
            "Use --overwrite only when replacement is intended."
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return run_dir, csv_path, summary_path, checkpoint_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    source_state = source_repository_state(
        allow_dirty=bool(args.allow_dirty_source)
    )
    data_root = args.data_root.expanduser()
    dataset_identity = verify_reacdiff_dataset(data_root)
    set_seed(args.seed)
    data = load_reacdiff(data_root)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    built = build_model(args.model, device)
    model = built.model
    parameter_count = count_model_params(model)
    validate_model_parameter_count(args.model, parameter_count)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX
    )
    run_dir, csv_path, summary_path, checkpoint_path = prepare_run_paths(
        args.results_root.expanduser(), args.model, args.seed, args.overwrite
    )
    run_label = public_run_label(run_dir, args.model, args.seed)
    print(
        f"\n========== Reaction-Diffusion model: {args.model} | "
        f"seed: {args.seed} =========="
    )
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count}")
    print(f"Run directory: {run_label}")
    started = time.perf_counter()
    history = run_training_loop(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        csv_path=csv_path,
    )
    total_training_time = time.perf_counter() - started
    if len(history) != EPOCHS or history[-1]["epoch"] != EPOCHS:
        raise RuntimeError(
            "Training did not finish at the fixed final epoch; final artifacts "
            "will not be written."
        )
    final = history[-1]
    environment = environment_metadata(source_state)
    from scripts import run_paper_sweep as paper_sweep

    contract = paper_sweep.run_contract_metadata("reacdiff1024", source_state)
    environment.update(paper_sweep.run_environment_provenance(contract))
    model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    run_metadata = {
        "experiment": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "dataset": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_loading_mode": "verified_local_offline",
        "dataset_loader_provenance": (
            "Reuses the pinned SirenFNO _find_data_dataset, "
            "load_subset_to_ram, RolloutRAMDataset, split, and DataLoader "
            "semantics after local identity verification; the released "
            "download call is intentionally excluded from training."
        ),
        "training_network_access_required": False,
        "dataset_identity": dataset_identity,
        "dataset_hdf5_key": data.dataset_key,
        "source_resolution": RESOLUTION,
        "loaded_resolution": data.resolution,
        "source_time_steps": EXPECTED_TIME_STEPS,
        "author_commented_time_steps": AUTHOR_COMMENTED_TIME_STEPS,
        "loaded_time_steps": data.time_steps,
        "temporal_provenance_note": TEMPORAL_PROVENANCE_NOTE,
        "reduce_x": REDUCE_X,
        "reduce_t": REDUCE_T,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_policy": SPLIT_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "normalization_mean": None,
        "normalization_std": None,
        "model": args.model,
        "model_class": model_class,
        "model_hyperparameters": built.configuration,
        "model_configuration_provenance": model_configuration_provenance(
            args.model
        ),
        "factorization": built.factorization,
        "rank": built.rank,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "experiment_seed": args.seed,
        "seed_policy": SEED_POLICY,
        **rff_metadata(args.model),
        "input_steps": INPUT_STEPS,
        "rollout_steps": ROLLOUT,
        "physical_channels": PHYSICAL_CHANNELS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "field_dim": FIELD_DIM,
        "pushforward_detach": PUSHFORWARD_DETACH,
        "epochs": EPOCHS,
        "final_epoch": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_loader_shuffle": True,
        "train_loader_drop_last": TRAIN_DROP_LAST,
        "evaluation_loader_shuffle": False,
        "evaluation_loader_drop_last": EVAL_DROP_LAST,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "scheduler_step_policy": "exactly_once_after_each_epoch_train_loop",
        "eval_interval": EVAL_INTERVAL,
        "amp_enabled": USE_AMP,
        "training_loss": "LpLoss(d=1,p=2,reduction=mean) rollout_loss_lp",
        **contract,
        **corrected_relative_l2_metadata(
            sample_count={
                "train": int(final["train_evaluation_sample_count"]),
                "test": int(final["test_evaluation_sample_count"]),
            },
            horizon=int(final["evaluation_horizon"]),
            evaluation_space="physical_source_values_no_normalization",
        ),
        "main_reporting_metric": MAIN_REPORTING_METRIC,
        "secondary_reporting_metrics": SECONDARY_REPORTING_METRICS,
        "final_train_corrected_step_relative_l2": final[
            "train_corrected_step_relative_l2"
        ],
        "final_train_corrected_trajectory_relative_l2": final[
            "train_corrected_trajectory_relative_l2"
        ],
        "final_test_corrected_step_relative_l2": final[
            "test_corrected_step_relative_l2"
        ],
        "final_test_corrected_trajectory_relative_l2": final[
            "test_corrected_trajectory_relative_l2"
        ],
        "checkpoint_selection_policy": SELECTION_POLICY,
        "comparison_roster_policy": (
            "released_reacdiff_pipeline_repository_fixed_twelve_models"
        ),
        "neuraloperator_source": AUTHOR_LOCAL_IMPORT_PATHS["neuralop"],
        "implementation_provenance": upstream_manifest(),
        "reacdiff_runtime_source_blobs": REACDIFF_SOURCE_BLOBS,
        **source_state,
    }
    checkpoint = prepare_weights_only_checkpoint({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_configuration": built.configuration,
        "epoch": EPOCHS,
        **run_metadata,
        "environment": environment,
    })
    torch.save(checkpoint, checkpoint_path)
    summary = {
        **run_metadata,
        "total_training_time_seconds": total_training_time,
        "final_checkpoint": "final_checkpoint.pt",
        "training_log": "training_log.csv",
        "run_directory": run_label,
        **environment,
    }
    summary["checkpoint_sha256"] = paper_sweep.sha256_file(checkpoint_path)
    summary_path.write_text(
        json.dumps(
            _jsonable(summary), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed fixed final epoch {EPOCHS}; final corrected test "
        "trajectory relative L2="
        f"{float(final['test_corrected_trajectory_relative_l2']):.10f}."
    )


if __name__ == "__main__":
    main()
