"""Audited PDEBench Burgers-1D paper experiment.

The data preparation, normalization, autoregressive rollout, losses, and
optimization order reproduce the pinned SirenFNO ``train_Burgers.py``.  Data
acquisition is deliberately excluded: training accepts only the locally
verified public HDF5 artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
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

# Verify and prioritize the exact author checkout before importing its modules.
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
from baseline.AMFNO import FNO1dMLP
from baseline.UFNO import UFNO1d
from neuralop.models import FNO1d
from neuralop.utils import (
    count_model_params,
    evaluate_rel_l2_metrics,
    full_rel_l2_rel,
    rel_l2_sum_over_batch,
    rollout_loss_rel_l2,
    rollout_step_model_1d_scalar,
)
from SirenFNO1D import SirenFNO1d
from torch.utils.data import DataLoader
from utils import (
    RolloutRAMDataset1D,
    _find_data_dataset,
    compute_mean_std_from_ram,
    load_subset_to_ram_1d_scalar,
    preprocess_decimated_hdf5_1d_scalar,
)
import train_Burgers as train_burgers_reference

from experiments.common.seed import set_seed
from experiments.configs.burgers1d import (
    BATCH_SIZE,
    BURGERS_SOURCE_BLOBS,
    CAFEPLUSFNO_FACTORIZATIONS,
    CANONICAL_SEED,
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_NAME,
    DATASET_SOURCE,
    EPOCHS,
    EVAL_DROP_LAST,
    EVAL_INTERVAL,
    EXPECTED_PINNED_PARAMETER_COUNTS,
    EXPECTED_TIME_STEPS,
    INPUT_STEPS,
    LEARNING_RATE,
    MAX_NORMALIZATION_TRAJECTORIES,
    MODEL_CHOICES,
    N_TEST,
    N_TRAIN,
    N_VAL,
    NORMALIZATION_POLICY,
    NUM_WORKERS,
    PIN_MEMORY,
    PUSHFORWARD_DETACH,
    REDUCE_T,
    REDUCE_X,
    RESOLUTION,
    ROLLOUT,
    SCHEDULER_T_MAX,
    SEED_POLICY,
    SELECTION_POLICY,
    SPLIT_POLICY,
    TRAIN_DROP_LAST,
    USE_AMP,
    WEIGHT_DECAY,
    model_configuration_provenance,
    model_constructor_kwargs,
)
from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D


@dataclass(frozen=True)
class ModelBuild:
    """A model and its fully resolved, checkpoint-reconstructable config."""

    model: torch.nn.Module
    configuration: dict[str, Any]
    factorization: str | None
    rank: int | float | None


@dataclass(frozen=True)
class BurgersData:
    """The exact three author-style loaders and public-safe preprocessing data."""

    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_eval_loader: DataLoader
    mean: float
    std: float
    total_trajectories: int
    time_steps: int
    resolution: int
    reduced_filename: str


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
            "count_model_params": count_model_params,
            "neuralop.utils(runtime)": neuralop_utils,
            "rollout_step_model_1d_scalar": rollout_step_model_1d_scalar,
            "rollout_loss_rel_l2": rollout_loss_rel_l2,
            "RolloutRAMDataset1D": RolloutRAMDataset1D,
            "preprocess_decimated_hdf5_1d_scalar": (
                preprocess_decimated_hdf5_1d_scalar
            ),
            "load_subset_to_ram_1d_scalar": load_subset_to_ram_1d_scalar,
            "compute_mean_std_from_ram": compute_mean_std_from_ram,
            "train_Burgers(reference_only)": train_burgers_reference,
            "SirenFNO1d": SirenFNO1d,
            "FNO1dMLP": FNO1dMLP,
            "UFNO1d": UFNO1d,
        },
        proposed_components={"CAFEPlusFNO1D": CAFEPlusFNO1D},
    )


AUTHOR_LOCAL_IMPORT_PATHS = author_local_import_audit()


def rff_metadata(model_name: str) -> dict[str, str]:
    """Describe the actual random-feature initialization policy."""

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
    raise KeyError(f"Unknown Burgers model for RFF metadata: {model_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Burgers-1D paper model for one seed."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Canonical model identifier; no model is selected implicitly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_SEED,
        help=f"Explicit multi-seed harness value (default: {CANONICAL_SEED}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=f"Directory containing the verified {DATASET_FILENAME} file.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            Path("results") / "burgers1024_sirenfno_81918ec_protocol_v2"
        ),
        help="Root under which <model>/seed_<seed>/ is created.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device without a hard-coded CUDA index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this run's three known output files.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Permit an explicitly non-release run from a dirty source tree.",
    )
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
    """Require exact constructor/config parity except padding normalization."""

    for key, requested_value in requested.items():
        expected_value = requested_value
        if key == "padding" and isinstance(requested_value, int):
            expected_value = (requested_value,)
        resolved_value = resolved.get(key)
        if resolved_value != expected_value:
            raise RuntimeError(
                "CAFE+FNO constructor/config parity failed for "
                f"{key!r}: requested {requested_value!r}, "
                f"resolved {resolved_value!r}."
            )

    expected_variant = CAFEPLUSFNO_FACTORIZATIONS[model_name]
    factorization = str(resolved["factorization"])
    rank = int(resolved["rank"])
    if factorization != expected_variant["factorization"]:
        raise RuntimeError("CAFE+FNO resolved to the wrong factorization.")
    if rank != expected_variant["rank"]:
        raise RuntimeError(
            "CAFE+FNO resolved to the wrong Burgers factorization rank: "
            f"expected {expected_variant['rank']}, resolved {rank}."
        )
    return factorization, rank


def build_model(model_name: str, device: torch.device) -> ModelBuild:
    """Construct exactly one audited Burgers model on the selected device."""

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
        raise ValueError(f"Unsupported Burgers model: {model_name}")

    return ModelBuild(
        model=model.to(device),
        configuration=kwargs,
        factorization=factorization,
        rank=rank,
    )


def expected_burgers_file(data_root: Path) -> Path:
    return data_root / DATASET_FILENAME


def verify_burgers_dataset(data_root: Path) -> dict[str, Any]:
    """Fail closed unless the exact local public HDF5 file is verified."""

    try:
        return verify_dataset(DATASET_ID, data_root)
    except (FileNotFoundError, RuntimeError):
        raise RuntimeError(
            "Burgers-1D data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset burgers1d --data-root data"
        ) from None


def fixed_split_indices(total_trajectories: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the pinned first-1000/next-200 deterministic split."""

    required = N_TRAIN + N_VAL + N_TEST
    if total_trajectories < required:
        raise ValueError(
            f"Burgers split requires {required} trajectories, got "
            f"{total_trajectories}."
        )
    all_indices = np.arange(total_trajectories)
    train_indices = all_indices[:N_TRAIN]
    test_indices = all_indices[N_TRAIN + N_VAL : required]
    return train_indices, test_indices


def _preprocess_verified_dataset(data_path: Path) -> Path:
    """Run the pinned preprocessor without exposing an absolute local path."""

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            reduced = preprocess_decimated_hdf5_1d_scalar(
                str(data_path), REDUCE_X, REDUCE_T
            )
    except (OSError, RuntimeError, ValueError):
        raise RuntimeError(
            "Burgers preprocessing failed for the verified local dataset."
        ) from None
    print("Burgers preprocessing cache: ready")
    return Path(reduced)


def _inspect_preprocessed_data(path: Path) -> tuple[int, int, int, dict[str, Any]]:
    try:
        with h5py.File(path, "r") as handle:
            dataset = _find_data_dataset(handle)
            if len(dataset.shape) < 3:
                raise RuntimeError("The preprocessed Burgers dataset is malformed.")
            shape = tuple(int(value) for value in dataset.shape[:3])
            attributes = {
                "reduce_x": dataset.attrs.get("reduce_x"),
                "reduce_t": dataset.attrs.get("reduce_t"),
                "source": dataset.attrs.get("source"),
            }
    except (OSError, RuntimeError, ValueError):
        raise RuntimeError(
            "The local Burgers preprocessing cache is invalid."
        ) from None
    return shape[0], shape[1], shape[2], attributes


def _verify_cache_provenance(
    attributes: dict[str, Any], source_path: Path
) -> None:
    """Reject a stale cache while keeping absolute paths out of output."""

    try:
        cached_source = Path(str(attributes["source"])).resolve()
        reduce_x = int(attributes["reduce_x"])
        reduce_t = int(attributes["reduce_t"])
    except (TypeError, ValueError):
        raise RuntimeError(
            "The local Burgers preprocessing cache has invalid provenance."
        ) from None
    if (
        cached_source != source_path.resolve()
        or reduce_x != REDUCE_X
        or reduce_t != REDUCE_T
    ):
        raise RuntimeError(
            "The local Burgers preprocessing cache does not match the verified "
            "source or decimation settings. Remove only the generated "
            "<data-root>/1D_Burgers_Sols_Nu0.001_rx1_rt1.hdf5 cache and retry."
        )


def build_data_loaders(
    train_cpu: torch.Tensor,
    test_cpu: torch.Tensor,
    mean: float,
    std: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct DataLoaders with the pinned author's exact options."""

    train_dataset = RolloutRAMDataset1D(
        train_cpu, INPUT_STEPS, ROLLOUT, mean, std
    )
    train_eval_dataset = RolloutRAMDataset1D(
        train_cpu, INPUT_STEPS, ROLLOUT, mean, std
    )
    test_eval_dataset = RolloutRAMDataset1D(
        test_cpu, INPUT_STEPS, ROLLOUT, mean, std
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


def load_burgers(data_root: Path) -> BurgersData:
    """Load the verified local dataset via the pinned offline-only pipeline."""

    data_path = expected_burgers_file(data_root)
    if not data_path.is_file():
        raise FileNotFoundError(
            "Burgers-1D data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset burgers1d --data-root data"
        )

    reduced_path = _preprocess_verified_dataset(data_path)
    total, time_steps, resolution, attributes = _inspect_preprocessed_data(
        reduced_path
    )
    _verify_cache_provenance(attributes, data_path)
    if time_steps != EXPECTED_TIME_STEPS or resolution != RESOLUTION:
        raise RuntimeError(
            "The local Burgers dataset has unexpected temporal or spatial shape."
        )
    if time_steps < INPUT_STEPS + ROLLOUT:
        raise ValueError(
            "The preprocessed Burgers time dimension is shorter than the "
            "configured input and rollout window."
        )

    train_indices, test_indices = fixed_split_indices(total)
    train_cpu = load_subset_to_ram_1d_scalar(
        str(reduced_path), train_indices
    )
    test_cpu = load_subset_to_ram_1d_scalar(str(reduced_path), test_indices)
    mean, std = compute_mean_std_from_ram(
        train_cpu,
        max_traj_for_stats=min(
            MAX_NORMALIZATION_TRAJECTORIES, train_cpu.shape[0]
        ),
    )
    loaders = build_data_loaders(train_cpu, test_cpu, mean, std)
    return BurgersData(
        train_loader=loaders[0],
        train_eval_loader=loaders[1],
        test_eval_loader=loaders[2],
        mean=float(mean),
        std=float(std),
        total_trajectories=total,
        time_steps=time_steps,
        resolution=resolution,
        reduced_filename=reduced_path.name,
    )


def validate_baseline_parameter_count(model_name: str, count: int) -> None:
    """Detect accidental drift in every pinned/paper-resolved baseline."""

    expected = EXPECTED_PINNED_PARAMETER_COUNTS.get(model_name)
    if expected is not None and count != expected:
        raise RuntimeError(
            f"Burgers {model_name} parameter count changed: expected "
            f"{expected}, got {count}."
        )


def _step_count_from_target(
    _pred_seq: torch.Tensor, y_seq: torch.Tensor, _rollout: int
) -> int:
    return int(y_seq.size(1))


LOG_FIELDNAMES = (
    "epoch",
    "train_step_relative_l2",
    "train_full_relative_l2",
    "test_step_relative_l2",
    "test_full_relative_l2",
    "learning_rate",
    "train_time_seconds",
    "epoch_time_seconds",
)


def run_training_loop(
    *,
    model: torch.nn.Module,
    data: BurgersData,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    csv_path: Path,
) -> list[dict[str, float | int]]:
    """Run the pinned 500-epoch training/evaluation ordering exactly."""

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
                prediction = rollout_step_model_1d_scalar(
                    model,
                    x_batch,
                    ROLLOUT,
                    pushforward_detach=PUSHFORWARD_DETACH,
                )
                loss = rollout_loss_rel_l2(prediction, y_batch)
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start
            scheduler.step()

            train_step_rel, train_full_rel = evaluate_rel_l2_metrics(
                model,
                data.train_eval_loader,
                ROLLOUT,
                rollout_step_model_1d_scalar,
                rel_l2_sum_over_batch,
                full_rel_l2_rel,
                device,
                step_count_fn=_step_count_from_target,
            )
            test_step_rel, test_full_rel = evaluate_rel_l2_metrics(
                model,
                data.test_eval_loader,
                ROLLOUT,
                rollout_step_model_1d_scalar,
                rel_l2_sum_over_batch,
                full_rel_l2_rel,
                device,
                step_count_fn=_step_count_from_target,
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_step_relative_l2": float(train_step_rel),
                "train_full_relative_l2": float(train_full_rel),
                "test_step_relative_l2": float(test_step_rel),
                "test_full_relative_l2": float(test_full_rel),
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
                f"{train_step_rel:.10f}/{test_step_rel:.10f} | "
                "full_rel(train/test) "
                f"{train_full_rel:.10f}/{test_full_rel:.10f}"
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


def environment_metadata(source_state: dict[str, Any]) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return normalize_environment_metadata({
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
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
        path
        for path in (csv_path, summary_path, checkpoint_path)
        if path.exists()
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

    # Exact Darcy/NS policy: local provenance verification precedes the one
    # global seed; every data/model RNG consumer follows that single call.
    data_root = args.data_root.expanduser()
    dataset_identity = verify_burgers_dataset(data_root)
    set_seed(args.seed)
    data = load_burgers(data_root)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    built = build_model(args.model, device)
    model = built.model
    parameter_count = count_model_params(model)
    validate_baseline_parameter_count(args.model, parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_T_MAX,
    )
    run_dir, csv_path, summary_path, checkpoint_path = prepare_run_paths(
        args.results_root.expanduser(), args.model, args.seed, args.overwrite
    )
    run_label = public_run_label(run_dir, args.model, args.seed)

    print(
        f"\n========== Burgers-1D model: {args.model} | "
        f"seed: {args.seed} =========="
    )
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count}")
    print(f"NeuralOperator: {neuralop.__version__}")
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

    contract = paper_sweep.run_contract_metadata("burgers1024", source_state)
    environment.update(paper_sweep.run_environment_provenance(contract))
    model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    run_metadata = {
        "dataset": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_loading_mode": "verified_local_offline",
        "dataset_loader_provenance": (
            "Reuses the pinned SirenFNO preprocessing, RAM loading, "
            "normalization, RolloutRAMDataset1D, and DataLoader semantics after "
            "local dataset verification, with no training-time acquisition."
        ),
        "training_network_access_required": False,
        "dataset_identity": dataset_identity,
        "source_resolution": RESOLUTION,
        "reduced_resolution": data.resolution,
        "source_time_steps": EXPECTED_TIME_STEPS,
        "reduced_time_steps": data.time_steps,
        "reduce_x": REDUCE_X,
        "reduce_t": REDUCE_T,
        "preprocessed_cache_filename": data.reduced_filename,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_policy": SPLIT_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "normalization_max_train_trajectories": (
            MAX_NORMALIZATION_TRAJECTORIES
        ),
        "normalization_mean": data.mean,
        "normalization_std": data.std,
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
        "eval_interval": EVAL_INTERVAL,
        "amp_enabled": USE_AMP,
        "training_loss": "rollout_loss_rel_l2",
        "evaluation_metrics": ["step relative L2", "full relative L2"],
        "final_train_step_relative_l2": final["train_step_relative_l2"],
        "final_train_full_relative_l2": final["train_full_relative_l2"],
        "final_test_step_relative_l2": final["test_step_relative_l2"],
        "final_test_full_relative_l2": final["test_full_relative_l2"],
        "checkpoint_selection_policy": SELECTION_POLICY,
        "neuraloperator_source": AUTHOR_LOCAL_IMPORT_PATHS["neuralop"],
        "implementation_provenance": upstream_manifest(),
        "burgers_runtime_source_blobs": BURGERS_SOURCE_BLOBS,
        **contract,
        "evaluation_sample_count": {
            "train": len(data.train_eval_loader.dataset),
            "test": len(data.test_eval_loader.dataset),
        },
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

    print("\nFixed-final-epoch artifacts written:")
    print(f"  {run_label}/{csv_path.name}")
    print(f"  {run_label}/{summary_path.name}")
    print(f"  {run_label}/{checkpoint_path.name}")


if __name__ == "__main__":
    main()
