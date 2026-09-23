"""Audited Navier-Stokes 128x128 main-comparison paper experiment.

The dataset and training protocol, plus every baseline architecture setting,
are transcribed from the SirenFNO-author ``train_NS.py``.  One invocation
trains exactly one model and one explicit experiment seed.  CSV logging only
records values already computed by ``neuralop.Trainer``.
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

# Verify and prioritize the exact official checkout before neuralop imports.
BACKEND_PROVENANCE = bootstrap_sirenfno_backend()

import torch
import neuralop
from experiments.common.checkpoints import (
    normalize_environment_metadata,
    prepare_weights_only_checkpoint,
)
from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets.navier_stokes import (
    NavierStokesDataset,
    load_navier_stokes_pt as load_navier_stokes_pt_reference,
)
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from torch.utils.data import DataLoader

from experiments.common.seed import set_seed
from experiments.configs.ns2d import (
    BATCH_SIZE,
    CAFEPLUSFNO_FACTORIZATIONS,
    CANONICAL_SEED,
    DATASET_NAME,
    DATASET_SOURCE,
    EPOCHS,
    EVAL_INTERVAL,
    LEARNING_RATE,
    MODEL_CHOICES,
    N_TEST,
    N_TRAIN,
    RESOLUTION,
    SCHEDULER_T_MAX,
    SEED_POLICY,
    SELECTION_POLICY,
    TEST_BATCH_SIZE,
    WEIGHT_DECAY,
    model_constructor_kwargs,
)
from baseline.AMFNO import FNO2dMLP
from baseline.UFNO import UFNO
from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D
from SirenFNO2D import SirenFNO2d


@dataclass(frozen=True)
class ModelBuild:
    """A model and the resolved constructor metadata recorded with its run."""

    model: torch.nn.Module
    configuration: dict[str, Any]
    factorization: str | None
    rank: int | float | None


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
    """Validate and report public-safe implementation paths used by NS."""

    return audit_runtime_sources(
        official_components={
            "neuralop": neuralop,
            "FNO": FNO,
            "Trainer": Trainer,
            "LpLoss": LpLoss,
            "H1Loss": H1Loss,
            "NavierStokesDataset(runtime)": NavierStokesDataset,
            "load_navier_stokes_pt(reference_only)": load_navier_stokes_pt_reference,
            "AdamW": AdamW,
            "count_model_params": count_model_params,
            "SirenFNO2d": SirenFNO2d,
            "FNO2dMLP": FNO2dMLP,
            "UFNO": UFNO,
        },
        proposed_components={"CAFEPlusFNO2D": CAFEPlusFNO2D},
    )


AUTHOR_LOCAL_IMPORT_PATHS = author_local_import_audit()


def rff_metadata(model_name: str) -> dict[str, str]:
    """Describe the actual RFF initialization policy for a known model."""

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
    raise KeyError(f"Unknown Navier-Stokes model for RFF metadata: {model_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Navier-Stokes 128x128 paper model for one seed."
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
        help="Directory containing official nsforcing train/test PT files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "ns128_sirenfno_81918ec",
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
    """Require constructor/config parity, allowing only padding normalization."""

    for key, requested_value in requested.items():
        expected_value = requested_value
        if key == "padding" and isinstance(requested_value, int):
            expected_value = (requested_value, requested_value)
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
            "CAFE+FNO resolved to the wrong Navier-Stokes factorization rank: "
            f"expected {expected_variant['rank']}, resolved {rank}."
        )
    return factorization, rank


def build_model(model_name: str, device: torch.device) -> ModelBuild:
    """Construct one audited Navier-Stokes model on the selected device."""

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
    elif name == "ufno":
        model = UFNO(**kwargs)
        factorization = None
        rank = None
    elif name in {
        "sirenfno",
        "cpsirenfno",
        "ttsirenfno",
        "tuckersirenfno",
    }:
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
        raise ValueError(f"Unsupported Navier-Stokes model: {model_name}")

    return ModelBuild(
        model=model.to(device),
        configuration=kwargs,
        factorization=factorization,
        rank=rank,
    )


def expected_ns_files(data_root: Path) -> tuple[Path, Path]:
    """Return filenames used by the pinned author-bundled NS loader."""

    return (
        data_root / f"nsforcing_train_{RESOLUTION}.pt",
        data_root / f"nsforcing_test_{RESOLUTION}.pt",
    )


def verify_ns_dataset(data_root: Path) -> dict[str, Any]:
    """Fail closed unless the exact local NS128 files pass provenance checks."""

    try:
        return verify_dataset("ns128", data_root)
    except (FileNotFoundError, RuntimeError):
        raise RuntimeError(
            "NS128 data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset ns128 --data-root data"
        ) from None


def load_ns(data_root: Path):
    """Reproduce the pinned official NS loader with downloads disabled."""

    missing = [path for path in expected_ns_files(data_root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "NS128 data is missing or failed checksum verification.\n"
            "Prepare it with:\n"
            "python -B data/download_data.py --dataset ns128 --data-root data"
        )

    dataset = NavierStokesDataset(
        root_dir=str(data_root),
        n_train=N_TRAIN,
        n_tests=[N_TEST],
        batch_size=BATCH_SIZE,
        test_batch_sizes=[TEST_BATCH_SIZE],
        train_resolution=RESOLUTION,
        test_resolutions=[RESOLUTION],
        encode_input=True,
        encode_output=True,
        encoding="channel-wise",
        channel_dim=1,
        subsampling_rate=None,
        download=False,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {
        resolution: DataLoader(
            dataset.test_dbs[resolution],
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        for resolution, test_batch_size in zip(
            [RESOLUTION], [TEST_BATCH_SIZE]
        )
    }
    return train_loader, test_loaders, dataset.data_processor


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _single_eval_metric(metrics: dict[str, Any], metric_name: str) -> float:
    suffix = f"_{metric_name}"
    matches = [value for key, value in metrics.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one test metric ending in {suffix!r}, got {sorted(metrics)}."
        )
    return _as_float(matches[0])


class CsvLoggingTrainer(Trainer):
    """Record existing Trainer metrics without extra numerical passes."""

    FIELDNAMES = (
        "epoch",
        "train_relative_l2",
        "test_relative_l2",
        "test_h1",
        "learning_rate",
        "train_time_seconds",
        "epoch_time_seconds",
    )

    def __init__(self, *, csv_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self.history: list[dict[str, float | int]] = []
        self._epoch_wall_start = 0.0
        self._latest_training: dict[str, float | int] = {}
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writeheader()

    def on_epoch_start(self, epoch):
        self._epoch_wall_start = time.perf_counter()
        return super().on_epoch_start(epoch)

    def train_one_epoch(self, epoch, train_loader, training_loss):
        train_err, avg_loss, avg_lasso_loss, train_time = super().train_one_epoch(
            epoch, train_loader, training_loss
        )
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self._latest_training = {
            "epoch": int(epoch) + 1,
            "train_relative_l2": _as_float(avg_loss),
            "learning_rate": learning_rate,
            "train_time_seconds": _as_float(train_time),
        }
        return train_err, avg_loss, avg_lasso_loss, train_time

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
    ):
        metrics = super().evaluate_all(
            epoch=epoch,
            eval_losses=eval_losses,
            test_loaders=test_loaders,
            eval_modes=eval_modes,
            max_autoregressive_steps=max_autoregressive_steps,
        )
        row = {
            **self._latest_training,
            "test_relative_l2": _single_eval_metric(metrics, "l2"),
            "test_h1": _single_eval_metric(metrics, "h1"),
            "epoch_time_seconds": time.perf_counter() - self._epoch_wall_start,
        }
        self.history.append(row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writerow(row)
        return metrics


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
    existing = [path for path in (csv_path, summary_path, checkpoint_path) if path.exists()]
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
    neuraloperator_version = str(neuralop.__version__)
    device = resolve_device(args.device)
    source_state = source_repository_state(
        allow_dirty=bool(args.allow_dirty_source)
    )

    # Match train_darcy.py exactly: verify local data first, then apply one
    # CLI-controlled global seed before DataLoader and model construction.
    data_root = args.data_root.expanduser()
    dataset_identity = verify_ns_dataset(data_root)
    set_seed(args.seed)
    train_loader, test_loaders, data_processor = load_ns(data_root)
    data_processor = data_processor.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    built = build_model(args.model, device)
    model = built.model
    parameter_count = count_model_params(model)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = l2loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

    optimizer = AdamW(
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

    print(f"\n========== NS128 model: {args.model} | seed: {args.seed} ==========")
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count}")
    print(f"NeuralOperator: {neuraloperator_version}")
    print(f"NeuralOperator source: {AUTHOR_LOCAL_IMPORT_PATHS['neuralop']}")
    print(f"Run directory: {run_label}")

    trainer = CsvLoggingTrainer(
        csv_path=csv_path,
        model=model,
        n_epochs=EPOCHS,
        device=str(device),
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=EVAL_INTERVAL,
        use_distributed=False,
        verbose=True,
    )

    started = time.perf_counter()
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )
    total_training_time = time.perf_counter() - started

    if len(trainer.history) != EPOCHS:
        raise RuntimeError(
            f"Expected {EPOCHS} logged epochs, found {len(trainer.history)}; "
            "final artifacts will not be written."
        )
    final = trainer.history[-1]
    if final["epoch"] != EPOCHS:
        raise RuntimeError("Training did not finish at the fixed final epoch.")

    env = environment_metadata(source_state)
    from scripts import run_paper_sweep as paper_sweep

    contract = paper_sweep.run_contract_metadata("ns128", source_state)
    env.update(paper_sweep.run_environment_provenance(contract))
    provenance = upstream_manifest()
    model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    rff_meta = rff_metadata(args.model)
    run_metadata = {
        "dataset": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_loading_mode": "verified_local_offline",
        "dataset_loader_provenance": (
            "Reproduces the pinned load_navier_stokes_pt pipeline while disabling "
            "its download path after local dataset verification."
        ),
        "training_network_access_required": False,
        "resolution": f"{RESOLUTION}x{RESOLUTION}",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "model": args.model,
        "model_class": model_class,
        "model_hyperparameters": built.configuration,
        "factorization": built.factorization,
        "rank": built.rank,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "experiment_seed": args.seed,
        "seed_policy": SEED_POLICY,
        **rff_meta,
        "epochs": EPOCHS,
        "final_epoch": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "optimizer": "NeuralOperator AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "eval_interval": EVAL_INTERVAL,
        "training_loss": "relative L2",
        "evaluation_metrics": ["relative L2", "H1"],
        "final_train_loss": final["train_relative_l2"],
        "final_test_relative_l2": final["test_relative_l2"],
        "final_test_h1": final["test_h1"],
        "checkpoint_selection_policy": SELECTION_POLICY,
        "dataset_identity": dataset_identity,
        "neuraloperator_source": AUTHOR_LOCAL_IMPORT_PATHS["neuralop"],
        "implementation_provenance": provenance,
        **contract,
        "evaluation_sample_count": {
            "train": len(train_loader.dataset),
            "test": len(next(iter(test_loaders.values())).dataset),
        },
    }

    checkpoint = prepare_weights_only_checkpoint({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_configuration": built.configuration,
        "epoch": EPOCHS,
        **run_metadata,
        "environment": env,
    })
    torch.save(checkpoint, checkpoint_path)

    summary = {
        **run_metadata,
        "total_training_time_seconds": total_training_time,
        "final_checkpoint": "final_checkpoint.pt",
        "training_log": "training_log.csv",
        **env,
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
