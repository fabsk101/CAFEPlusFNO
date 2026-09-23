"""Verified-local Airfoil experiment for the unified eleven-model comparison.

Training performs no acquisition or network fallback. The scheduler ordering
reproduces the audited AM-FNO Airfoil source: a scheduler step follows every
optimizer step, followed by one additional scheduler step at each epoch end.
"""

from __future__ import annotations

import argparse
import csv
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

import neuralop
import numpy as np
import torch
from experiments.common.checkpoints import (
    normalize_environment_metadata,
    prepare_weights_only_checkpoint,
)
from baseline.AMFNO import FNO2dMLP
from neuralop.losses import LpLoss
from neuralop.models import FNO
from neuralop.utils import count_model_params
from SirenFNO2D import SirenFNO2d
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.seed import set_seed
from experiments.configs.airfoil import (
    AIRFOIL_EXCLUDED_BASELINES,
    AIRFOIL_RUNTIME_SOURCE_BLOBS,
    BATCH_SIZE,
    CAFEPLUSFNO_FACTORIZATIONS,
    CANONICAL_SEED,
    DATASET_FILENAMES,
    DATASET_ID,
    DATASET_LOADING_MODE,
    DATASET_NAME,
    DATASET_SOURCE,
    EPOCHS,
    EVAL_DROP_LAST,
    EVAL_INTERVAL,
    EXPECTED_SCHEDULER_T_MAX,
    EXPECTED_TRAIN_BATCHES_PER_EPOCH,
    EXPECTED_CAFE_PARAMETER_COUNTS,
    EXPECTED_PINNED_PARAMETER_COUNTS,
    EXPERIMENT_ID,
    INPUT_DIM,
    LEARNING_RATE,
    MAIN_REPORTING_METRIC,
    MODEL_CHOICES,
    NORMALIZATION_POLICY,
    N_TEST,
    N_TRAIN,
    N_VAL,
    NUM_WORKERS,
    OUTPUT_DIM,
    PIN_MEMORY,
    RAW_SAMPLE_COUNT,
    RAW_SOLUTION_CHANNELS,
    RESOLUTION,
    SCHEDULER_STEP_POLICY,
    SCHEDULER_T_MAX_POLICY,
    SEED_POLICY,
    SELECTION_POLICY,
    SPLIT_POLICY,
    TARGET_FIELD_INDEX,
    TRAIN_DROP_LAST,
    USE_AMP,
    WEIGHT_DECAY,
    model_configuration_provenance,
    model_constructor_kwargs,
)
from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D


@dataclass(frozen=True)
class ModelBuild:
    """A model and its fully resolved, checkpoint-reconstructable config."""

    model: torch.nn.Module
    configuration: dict[str, Any]
    factorization: str | None
    rank: int | float | None


@dataclass(frozen=True)
class AirfoilData:
    """Exact fixed-split Airfoil loaders and public-safe source dimensions."""

    train_loader: DataLoader
    test_loader: DataLoader
    total_samples: int
    input_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]


_RFF_MODEL_NAMES = frozenset(
    {
        "sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno",
        "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
        "tucker_cafe_plus_fno",
    }
)
_NON_RFF_MODEL_NAMES = frozenset({"fno", "tfno_cp", "amfno"})


def author_local_import_audit() -> dict[str, str]:
    """Validate every upstream runtime component used by this experiment."""

    return audit_runtime_sources(
        official_components={
            "neuralop": neuralop,
            "FNO": FNO,
            "LpLoss": LpLoss,
            "count_model_params": count_model_params,
            "SirenFNO2d": SirenFNO2d,
            "FNO2dMLP": FNO2dMLP,
        },
        proposed_components={"CAFEPlusFNO2D": CAFEPlusFNO2D},
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
    raise KeyError(f"Unknown Airfoil model for RFF metadata: {model_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one unified Airfoil paper model for one seed."
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "airfoil221x51_sirenfno_81918ec",
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
    """Require exact constructor/config parity except 2-D padding normalization."""

    for key, requested_value in requested.items():
        expected_value = requested_value
        if key == "padding" and isinstance(requested_value, int):
            expected_value = (requested_value, requested_value)
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
        raise RuntimeError("CAFE+FNO resolved to the wrong Airfoil rank.")
    return factorization, rank


def build_model(model_name: str, device: torch.device) -> ModelBuild:
    """Construct exactly one explicit Airfoil model on the selected device."""

    name = model_name.lower()
    kwargs = model_constructor_kwargs(name)
    if name in {"fno", "tfno_cp"}:
        model = FNO(**kwargs)
        factorization = kwargs.get("factorization")
        rank = kwargs.get("rank")
    elif name == "amfno":
        model = FNO2dMLP(**kwargs)
        factorization = None
        rank = None
    elif name in {"sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno"}:
        model = SirenFNO2d(**kwargs)
        factorization = str(kwargs["factorization"])
        rank = kwargs.get("rank")
    elif name in CAFEPLUSFNO_FACTORIZATIONS:
        model = CAFEPlusFNO2D(**kwargs)
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
        raise ValueError(f"Unsupported Airfoil model: {model_name}")

    return ModelBuild(
        model=model.to(device),
        configuration=kwargs,
        factorization=factorization,
        rank=rank,
    )


def expected_airfoil_files(data_root: Path) -> tuple[Path, Path, Path]:
    return tuple(data_root / name for name in DATASET_FILENAMES)  # type: ignore[return-value]


def verify_airfoil_dataset(data_root: Path) -> dict[str, Any]:
    """Fail closed unless all three exact local NPY files are verified."""

    try:
        return verify_dataset(DATASET_ID, data_root)
    except (FileNotFoundError, RuntimeError):
        raise RuntimeError(
            "Airfoil data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset airfoil221x51 "
            "--data-root data"
        ) from None


def fixed_split_indices(total_samples: int) -> tuple[slice, slice]:
    required = N_TRAIN + N_VAL + N_TEST
    if total_samples < required:
        raise ValueError(
            f"Airfoil split requires {required} samples, got {total_samples}."
        )
    return slice(0, N_TRAIN), slice(N_TRAIN + N_VAL, required)


def build_data_loaders(
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    test_input: torch.Tensor,
    test_target: torch.Tensor,
) -> tuple[DataLoader, DataLoader]:
    """Construct the author-style shuffled train and fixed test loaders."""

    train_loader = DataLoader(
        TensorDataset(train_input, train_target),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=TRAIN_DROP_LAST,
    )
    test_loader = DataLoader(
        TensorDataset(test_input, test_target),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=EVAL_DROP_LAST,
    )
    return train_loader, test_loader


def load_airfoil(data_root: Path) -> AirfoilData:
    """Load X/Y coordinates and Q[:, 4] locally with no network fallback."""

    paths = expected_airfoil_files(data_root)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            "Airfoil data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset airfoil221x51 "
            "--data-root data"
        )

    try:
        x_coordinates = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        y_coordinates = np.load(paths[1], mmap_mode="r", allow_pickle=False)
        solution = np.load(paths[2], mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        raise RuntimeError("The verified local Airfoil NPY files are invalid.") from None

    coordinate_shape = (RAW_SAMPLE_COUNT, *RESOLUTION)
    solution_shape = (RAW_SAMPLE_COUNT, RAW_SOLUTION_CHANNELS, *RESOLUTION)
    if tuple(x_coordinates.shape) != coordinate_shape:
        raise RuntimeError("NACA_Cylinder_X.npy has an unexpected shape.")
    if tuple(y_coordinates.shape) != coordinate_shape:
        raise RuntimeError("NACA_Cylinder_Y.npy has an unexpected shape.")
    if tuple(solution.shape) != solution_shape:
        raise RuntimeError("NACA_Cylinder_Q.npy has an unexpected shape.")

    train_slice, test_slice = fixed_split_indices(RAW_SAMPLE_COUNT)

    def materialize(index: slice) -> tuple[torch.Tensor, torch.Tensor]:
        input_array = np.stack(
            (x_coordinates[index], y_coordinates[index]), axis=-1
        ).astype(np.float32, copy=False)
        target_array = np.asarray(
            solution[index, TARGET_FIELD_INDEX, :, :]
        )[..., None].astype(np.float32, copy=False)
        if not np.isfinite(input_array).all() or not np.isfinite(target_array).all():
            raise RuntimeError("The selected Airfoil tensors contain non-finite values.")
        return (
            torch.from_numpy(np.ascontiguousarray(input_array)),
            torch.from_numpy(np.ascontiguousarray(target_array)),
        )

    train_input, train_target = materialize(train_slice)
    test_input, test_target = materialize(test_slice)
    train_loader, test_loader = build_data_loaders(
        train_input, train_target, test_input, test_target
    )
    return AirfoilData(
        train_loader=train_loader,
        test_loader=test_loader,
        total_samples=RAW_SAMPLE_COUNT,
        input_shape=(*RESOLUTION, INPUT_DIM),
        target_shape=(*RESOLUTION, OUTPUT_DIM),
    )


def validate_model_parameter_count(model_name: str, count: int) -> None:
    expected = {
        **EXPECTED_PINNED_PARAMETER_COUNTS,
        **EXPECTED_CAFE_PARAMETER_COUNTS,
    }.get(model_name)
    if expected is not None and count != expected:
        raise RuntimeError(
            f"Airfoil {model_name} parameter count changed: expected "
            f"{expected}, got {count}."
        )


def _channel_first(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.permute(0, 3, 1, 2).contiguous().to(
        device, non_blocking=True
    )


def evaluate_relative_l2(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: LpLoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for input_batch, target_batch in loader:
            input_batch = _channel_first(input_batch, device)
            target_batch = _channel_first(target_batch, device)
            prediction = model(input_batch)
            total += float(loss_function(prediction, target_batch).item())
            samples += int(input_batch.shape[0])
    if samples == 0:
        raise RuntimeError("Airfoil evaluation loader is empty.")
    return total / samples


LOG_FIELDNAMES = (
    "epoch",
    "train_relative_l2",
    "test_relative_l2",
    "learning_rate",
    "learning_rate_last_batch",
    "learning_rate_after_epoch_step",
    "train_time_seconds",
    "epoch_time_seconds",
)


def run_training_loop(
    *,
    model: torch.nn.Module,
    data: AirfoilData,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    csv_path: Path,
) -> list[dict[str, float | int]]:
    """Run the AM-FNO Airfoil optimizer/scheduler ordering exactly."""

    relative_l2 = LpLoss(d=2, p=2, reduction="sum")
    history: list[dict[str, float | int]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            last_batch_learning_rate = epoch_learning_rate
            train_relative_l2_sum = 0.0
            train_samples = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_start = time.perf_counter()

            for input_batch, target_batch in data.train_loader:
                input_batch = _channel_first(input_batch, device)
                target_batch = _channel_first(target_batch, device)
                last_batch_learning_rate = float(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(input_batch)
                loss = relative_l2(prediction, target_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_relative_l2_sum += float(loss.detach().item())
                train_samples += int(input_batch.shape[0])

            if train_samples == 0:
                raise RuntimeError("Airfoil training loader is empty.")
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start

            # Preserved intentionally from the audited AM-FNO Airfoil source.
            scheduler.step()
            post_epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            test_relative_l2 = evaluate_relative_l2(
                model, data.test_loader, relative_l2, device
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_relative_l2": train_relative_l2_sum / train_samples,
                "test_relative_l2": test_relative_l2,
                "learning_rate": epoch_learning_rate,
                "learning_rate_last_batch": last_batch_learning_rate,
                "learning_rate_after_epoch_step": post_epoch_learning_rate,
                "train_time_seconds": train_time,
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch:03d}/{EPOCHS:03d} | "
                f"train_rel {row['train_relative_l2']:.10f} | "
                f"test_rel {test_relative_l2:.10f}"
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
    source_state = source_repository_state(allow_dirty=bool(args.allow_dirty_source))

    # Exact repository policy: verification, one global seed, loader, model.
    data_root = args.data_root.expanduser()
    dataset_identity = verify_airfoil_dataset(data_root)
    set_seed(args.seed)
    data = load_airfoil(data_root)
    if len(data.train_loader) != EXPECTED_TRAIN_BATCHES_PER_EPOCH:
        raise RuntimeError(
            "Airfoil training loader length changed: expected "
            f"{EXPECTED_TRAIN_BATCHES_PER_EPOCH}, got {len(data.train_loader)}."
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    built = build_model(args.model, device)
    model = built.model
    parameter_count = count_model_params(model)
    validate_model_parameter_count(args.model, parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler_t_max = EPOCHS * len(data.train_loader)
    if scheduler_t_max != EXPECTED_SCHEDULER_T_MAX:
        raise RuntimeError(
            "Airfoil scheduler T_max changed: expected "
            f"{EXPECTED_SCHEDULER_T_MAX}, got {scheduler_t_max}."
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max
    )
    run_dir, csv_path, summary_path, checkpoint_path = prepare_run_paths(
        args.results_root.expanduser(), args.model, args.seed, args.overwrite
    )
    run_label = public_run_label(run_dir, args.model, args.seed)

    print(f"\n========== Airfoil model: {args.model} | seed: {args.seed} ==========")
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

    contract = paper_sweep.run_contract_metadata("airfoil221x51", source_state)
    environment.update(paper_sweep.run_environment_provenance(contract))
    model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    run_metadata = {
        "experiment": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "dataset": DATASET_NAME,
        "dataset_filenames": DATASET_FILENAMES,
        "dataset_source": DATASET_SOURCE,
        "dataset_loading_mode": DATASET_LOADING_MODE,
        "dataset_loader_provenance": (
            "Loads verified local Geo-FNO Airfoil NPY files only; stacks X/Y "
            "as two input channels and selects Q[:, 4] as the scalar target."
        ),
        "training_network_access_required": False,
        "dataset_identity": dataset_identity,
        "excluded_baselines": AIRFOIL_EXCLUDED_BASELINES,
        "source_sample_count": data.total_samples,
        "source_resolution": RESOLUTION,
        "input_shape": data.input_shape,
        "target_shape": data.target_shape,
        "target_field_index": TARGET_FIELD_INDEX,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_policy": SPLIT_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "model": args.model,
        "model_class": model_class,
        "model_hyperparameters": built.configuration,
        "model_configuration_provenance": model_configuration_provenance(args.model),
        "factorization": built.factorization,
        "rank": built.rank,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "experiment_seed": args.seed,
        "seed_policy": SEED_POLICY,
        **rff_metadata(args.model),
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
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
        "scheduler_t_max": scheduler_t_max,
        "scheduler_t_max_policy": SCHEDULER_T_MAX_POLICY,
        "scheduler_step_policy": SCHEDULER_STEP_POLICY,
        "scheduler_provenance": "AM-FNO Airfoil author-source-exact policy",
        "eval_interval": EVAL_INTERVAL,
        "amp_enabled": USE_AMP,
        "training_loss": "relative_L2_sum_divided_by_sample_count",
        "evaluation_metrics": ["relative L2"],
        "main_reporting_metric": MAIN_REPORTING_METRIC,
        "final_train_relative_l2": final["train_relative_l2"],
        "final_test_relative_l2": final["test_relative_l2"],
        "checkpoint_selection_policy": SELECTION_POLICY,
        "neuraloperator_source": AUTHOR_LOCAL_IMPORT_PATHS["neuralop"],
        "implementation_provenance": upstream_manifest(),
        "airfoil_runtime_source_blobs": AIRFOIL_RUNTIME_SOURCE_BLOBS,
        **contract,
        "evaluation_sample_count": {
            "train": len(data.train_loader.dataset),
            "test": len(data.test_loader.dataset),
        },
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
        f"Completed fixed final epoch {EPOCHS}; "
        f"test relative L2={float(final['test_relative_l2']):.10f}."
    )


if __name__ == "__main__":
    main()
