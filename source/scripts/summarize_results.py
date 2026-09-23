"""Re-evaluate fixed-final checkpoints and build strict paper tables.

The ``reevaluate`` command never mutates a training run. It reconstructs the
model from the configuration stored in the checkpoint, strictly loads the
weights, restores the verified test split, and writes a versioned evaluation
record. The ``aggregate`` command accepts those records and refuses duplicate
or missing seeds and incompatible protocols unless partial output is requested
explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import hashlib
import importlib
import io
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.configs.seeds import PAPER_SEEDS  # noqa: E402


EVALUATION_RECORD_SCHEMA_VERSION = 2
AGGREGATE_SCHEMA_VERSION = 2
CORRECTED_EVALUATION_PROTOCOL = "corrected_relative_l2_v1"
COMPLETION_EVALUATION_SCHEMA = "paper_evaluation_record_v2"
PAPER_FIXED_FINAL_EPOCH = 500


@dataclass(frozen=True)
class DatasetAdapter:
    module: str
    dataset_id: str
    evaluation_kind: str
    evaluation_protocol: str
    evaluation_space: str


DATASETS: dict[str, DatasetAdapter] = {
    "airfoil221x51": DatasetAdapter(
        "experiments.train_airfoil",
        "airfoil221x51",
        "airfoil",
        "airfoil_sample_relative_l2_v1",
        "physical_source_values_no_normalization",
    ),
    "burgers1024": DatasetAdapter(
        "experiments.train_burgers",
        "burgers1d",
        "burgers",
        "burgers_rollout_relative_l2_v1",
        "train_normalized_values",
    ),
    "cfd1d1024": DatasetAdapter(
        "experiments.train_cfd1d",
        "cfd1d1024",
        "corrected_rollout",
        CORRECTED_EVALUATION_PROTOCOL,
        "physical_source_values_no_normalization",
    ),
    "cfd2d128": DatasetAdapter(
        "experiments.train_cfd2d",
        "cfd2d128",
        "corrected_rollout",
        CORRECTED_EVALUATION_PROTOCOL,
        "physical_source_values_no_normalization",
    ),
    "darcy128": DatasetAdapter(
        "experiments.train_darcy",
        "darcy128",
        "neuraloperator_single_step",
        "darcy_decoded_relative_l2_h1_v1",
        "decoded_physical_values",
    ),
    "ns128": DatasetAdapter(
        "experiments.train_ns2d",
        "ns128",
        "neuraloperator_single_step",
        "ns_decoded_relative_l2_h1_v1",
        "decoded_physical_values",
    ),
    "reacdiff1024": DatasetAdapter(
        "experiments.train_reacdiff",
        "reacdiff1024",
        "corrected_rollout",
        CORRECTED_EVALUATION_PROTOCOL,
        "physical_source_values_no_normalization",
    ),
}


PAPER_METRIC_FIELDS: dict[str, dict[str, str]] = {
    "airfoil221x51": {"final_test_relative_l2": "relative_l2"},
    "burgers1024": {
        "final_test_step_relative_l2": "step_relative_l2",
        "final_test_full_relative_l2": "trajectory_relative_l2",
    },
    "cfd1d1024": {
        "final_test_corrected_step_relative_l2": "corrected_step_relative_l2",
        "final_test_corrected_trajectory_relative_l2": (
            "corrected_trajectory_relative_l2"
        ),
    },
    "cfd2d128": {
        "final_test_corrected_step_relative_l2": "corrected_step_relative_l2",
        "final_test_corrected_trajectory_relative_l2": (
            "corrected_trajectory_relative_l2"
        ),
    },
    "darcy128": {
        "final_test_relative_l2": "relative_l2",
        "final_test_h1": "h1",
    },
    "ns128": {
        "final_test_relative_l2": "relative_l2",
        "final_test_h1": "h1",
    },
    "reacdiff1024": {
        "final_test_corrected_step_relative_l2": "corrected_step_relative_l2",
        "final_test_corrected_trajectory_relative_l2": (
            "corrected_trajectory_relative_l2"
        ),
    },
}


class ResultValidationError(RuntimeError):
    """Raised when a run or evaluation record is unsafe to use."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "result_validation_failed",
        safe_fields: Mapping[str, str | int] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.safe_fields = dict(safe_fields or {})


class ResultOutputExistsError(FileExistsError):
    """Typed non-destructive output collision with public-safe CLI fields."""

    reason_code = "output_exists"

    def __init__(self, *, output: str) -> None:
        super().__init__("Output already exists; use --overwrite intentionally.")
        self.safe_fields: dict[str, str | int] = {"output": output}


def _reject_nonfinite_json(token: str) -> None:
    raise ResultValidationError(f"Non-finite JSON token is forbidden: {token}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(
            "Invalid JSON artifact.",
            reason_code="invalid_json_input",
            safe_fields={"input": _public_path(path, artifact="evaluation-record")},
        ) from exc
    if not isinstance(value, dict):
        raise ResultValidationError(
            "Expected a JSON object.",
            reason_code="invalid_json_input",
            safe_fields={"input": _public_path(path, artifact="evaluation-record")},
        )
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResultValidationError("Metadata is not finite canonical JSON.") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{label} must be numeric.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ResultValidationError(f"{label} must be finite.")
    return converted


def _evaluation_test_count(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("test")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResultValidationError("Evaluation test sample count must be positive.")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{label} must be a non-empty string.")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(rendered + "\n", encoding="utf-8")


_PUBLIC_IDENTITY_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}\Z")
_PUBLIC_ARTIFACTS = {
    "aggregate-output",
    "checkpoint",
    "evaluation-record",
    "result",
}


def _public_identity(value: Any) -> str | None:
    if isinstance(value, str) and _PUBLIC_IDENTITY_RE.fullmatch(value):
        return value
    return None


def _public_path(
    path: Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    artifact: str = "result",
) -> str:
    """Render repo paths relatively and external paths without their basename."""

    rendered_path = str(path)
    host_path = Path(path)
    windows_form = PureWindowsPath(rendered_path)
    posix_form = PurePosixPath(rendered_path)

    # A foreign absolute path is relative according to the host Path class.
    # Resolving it first would incorrectly place it below the repository and
    # disclose every synthetic drive/profile component as a "relative" path.
    foreign_absolute = not host_path.is_absolute() and (
        windows_form.is_absolute()
        or bool(windows_form.drive)
        or posix_form.is_absolute()
    )
    try:
        if foreign_absolute:
            raise ValueError("foreign absolute path")
        return host_path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        safe_artifact = artifact if artifact in _PUBLIC_ARTIFACTS else "result"
        safe_dataset = _public_identity(dataset)
        safe_model = _public_identity(model)
        safe_seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        if safe_dataset is not None and safe_model is not None and safe_seed is not None:
            return (
                f"<external-results>/{safe_dataset}/{safe_model}/"
                f"seed-{safe_seed}/{safe_artifact}"
            )
        if safe_dataset is not None:
            return f"<external-results>/{safe_dataset}/{safe_artifact}"
        return f"<external-results>/{safe_artifact}"


def _same_json(left: Any, right: Any) -> bool:
    return _json_bytes(left) == _json_bytes(right)


def _compare_dataset_identity(
    recorded: Any,
    verified: Any,
    *,
    expected_dataset_id: str,
) -> None:
    if not isinstance(recorded, dict) or not isinstance(verified, dict):
        raise ResultValidationError("Dataset identity metadata is missing.")
    if recorded.get("dataset_id") != expected_dataset_id:
        raise ResultValidationError("Checkpoint dataset id does not match the CLI dataset.")
    if not _same_json(recorded, verified):
        raise ResultValidationError(
            "Checkpoint dataset identity differs from the currently verified data."
        )


def _constructor_for(module: Any, model_name: str) -> Any:
    if model_name in {"fno", "tfno_cp"}:
        candidates = ("FNO1d", "FNO")
    elif model_name == "amfno":
        candidates = ("FNO1dMLP", "FNO2dMLP")
    elif model_name == "ufno":
        candidates = ("UFNO1d", "UFNO")
    elif model_name in {"sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno"}:
        candidates = ("SirenFNO1d", "SirenFNO2d")
    elif model_name in {
        "cafe_plus_fno",
        "cp_cafe_plus_fno",
        "tt_cafe_plus_fno",
        "tucker_cafe_plus_fno",
    }:
        candidates = ("CAFEPlusFNO1D", "CAFEPlusFNO2D")
    else:
        raise ResultValidationError(f"Unsupported model in checkpoint: {model_name}")
    for name in candidates:
        constructor = getattr(module, name, None)
        if constructor is not None:
            return constructor
    raise ResultValidationError("The training module cannot reconstruct this model class.")


def _verify_finite_state_dict(state: Any, torch: Any) -> None:
    if not isinstance(state, dict) or not state:
        raise ResultValidationError("Checkpoint model_state_dict is empty or missing.")
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ResultValidationError("Checkpoint state_dict has an invalid entry.")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ResultValidationError(
                f"Checkpoint state tensor is non-finite: {key}"
            )


def _reconstruct_model(
    module: Any,
    checkpoint: Mapping[str, Any],
    *,
    device: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any], int]:
    model_name = _require_nonempty_string(checkpoint.get("model"), "checkpoint.model")
    configuration = checkpoint.get("model_configuration")
    if not isinstance(configuration, dict) or not configuration:
        raise ResultValidationError("Checkpoint model_configuration is empty or missing.")
    constructor = _constructor_for(module, model_name)
    try:
        model = constructor(**configuration).to(device)
    except Exception as exc:
        raise ResultValidationError(
            "The stored model configuration could not reconstruct the model."
        ) from exc
    if hasattr(model, "get_config"):
        resolved = model.get_config()
        if not _same_json(resolved, configuration):
            raise ResultValidationError(
                "The reconstructed model does not resolve to the stored configuration."
            )
    state = checkpoint.get("model_state_dict")
    _verify_finite_state_dict(state, torch)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ResultValidationError(
            "Checkpoint weights do not strictly match the stored model configuration."
        ) from exc
    count_function = getattr(module, "count_model_params", None)
    if count_function is None:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    else:
        parameter_count = int(count_function(model))
    recorded_count = checkpoint.get("parameter_count")
    if recorded_count != parameter_count:
        raise ResultValidationError(
            "Checkpoint parameter count does not match the reconstructed model."
        )
    return model, configuration, parameter_count


def _rebatch(loader: Any, batch_size: int, torch: Any) -> Any:
    if batch_size <= 0:
        raise ResultValidationError("Evaluation batch size must be positive.")
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


@contextlib.contextmanager
def _preserve_model_mode_and_rng(model: Any, torch: Any):
    mode_states = tuple((item, item.training) for item in model.modules())
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        for item, was_training in mode_states:
            item.training = was_training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _evaluate_rollout(
    adapter: DatasetAdapter,
    module: Any,
    model: Any,
    data: Any,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[dict[str, float], int, int, float, str]:
    horizon = int(module.ROLLOUT)
    loader = _rebatch(data.test_eval_loader, batch_size, torch)

    if adapter.evaluation_kind == "burgers":
        with _preserve_model_mode_and_rng(model, torch):
            step, trajectory = module.evaluate_rel_l2_metrics(
                model,
                loader,
                horizon,
                module.rollout_step_model_1d_scalar,
                module.rel_l2_sum_over_batch,
                module.full_rel_l2_rel,
                device,
                step_count_fn=module._step_count_from_target,
            )
        return (
            {
                "step_relative_l2": float(step),
                "trajectory_relative_l2": float(trajectory),
            },
            len(loader.dataset),
            horizon,
            1e-12,
            "B,T,C=1,*spatial",
        )

    evaluation = importlib.import_module("experiments.common.evaluation")
    rollout_fn = (
        module.rollout_cfd1d
        if adapter.module.endswith("train_cfd1d")
        else module.rollout_cfd2d
        if adapter.module.endswith("train_cfd2d")
        else module.rollout_reacdiff
    )
    result = evaluation.evaluate_corrected_relative_l2(
        model,
        loader,
        horizon=horizon,
        rollout_fn=rollout_fn,
        device=device,
        time_dim=1,
        channel_dim=2,
        epsilon=evaluation.DEFAULT_RELATIVE_L2_EPSILON,
    )
    metrics = {
        "corrected_step_relative_l2": result.corrected_step_relative_l2,
        "corrected_trajectory_relative_l2": (
            result.corrected_trajectory_relative_l2
        ),
    }
    return (
        metrics,
        result.sample_count,
        result.horizon,
        result.epsilon,
        evaluation.CANONICAL_TRAJECTORY_LAYOUT,
    )


def _evaluate_airfoil(
    module: Any,
    model: Any,
    data: Any,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[dict[str, float], int, int, float, str]:
    loader = _rebatch(data.test_loader, batch_size, torch)
    loss = module.LpLoss(d=2, p=2, reduction="sum")
    with _preserve_model_mode_and_rng(model, torch):
        metric = module.evaluate_relative_l2(
            model,
            loader,
            loss,
            device,
        )
    return (
        {"relative_l2": float(metric)},
        len(loader.dataset),
        1,
        float(loss.eps),
        "B,C=1,*spatial",
    )


def _evaluate_neuraloperator(
    module: Any,
    model: Any,
    loaded: tuple[Any, Any, Any],
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[dict[str, float], int, int, float, str]:
    _train_loader, test_loaders, data_processor = loaded
    if not isinstance(test_loaders, dict) or len(test_loaders) != 1:
        raise ResultValidationError("Expected exactly one verified test loader.")
    loader = _rebatch(next(iter(test_loaders.values())), batch_size, torch)
    data_processor = data_processor.to(device)
    l2_loss = module.LpLoss(d=2, p=2, reduction="sum")
    h1_loss = module.H1Loss(d=2, reduction="sum")
    mode_states = tuple((item, item.training) for item in model.modules())
    processor_mode_states = tuple(
        (item, item.training) for item in data_processor.modules()
    )
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    l2_sum = 0.0
    h1_sum = 0.0
    sample_count = 0
    epsilon = float(l2_loss.eps)
    try:
        model.eval()
        data_processor.eval()
        with torch.no_grad():
            for sample in loader:
                sample = data_processor.preprocess(sample)
                output = model(**sample)
                output, sample = data_processor.postprocess(output, sample)
                target = sample["y"]
                l2_value = float(l2_loss(output, target).detach().cpu().item())
                h1_value = float(h1_loss(output, target).detach().cpu().item())
                if not math.isfinite(l2_value) or not math.isfinite(h1_value):
                    raise ResultValidationError(
                        "Decoded L2/H1 evaluation produced NaN or infinity."
                    )
                l2_sum += l2_value
                h1_sum += h1_value
                sample_count += int(target.shape[0])
    finally:
        for item, was_training in mode_states:
            item.training = was_training
        for item, was_training in processor_mode_states:
            item.training = was_training
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if sample_count <= 0:
        raise ResultValidationError("Evaluation loader produced no samples.")
    return (
        {"relative_l2": l2_sum / sample_count, "h1": h1_sum / sample_count},
        sample_count,
        1,
        epsilon,
        "B,C=1,*spatial",
    )


def _verified_dataset_identity(adapter: DatasetAdapter, module: Any, data_root: Path) -> dict[str, Any]:
    if adapter.module.endswith("train_darcy"):
        return module.verify_dataset(adapter.dataset_id, data_root)
    for name in (
        "verify_ns_dataset",
        "verify_burgers_dataset",
        "verify_cfd1d_dataset",
        "verify_cfd2d_dataset",
        "verify_airfoil_dataset",
        "verify_reacdiff_dataset",
    ):
        function = getattr(module, name, None)
        if function is not None:
            return function(data_root)
    raise ResultValidationError("The training module has no dataset verifier.")


def _load_dataset(adapter: DatasetAdapter, module: Any, data_root: Path) -> Any:
    del adapter
    for name in (
        "load_darcy",
        "load_ns",
        "load_burgers",
        "load_cfd1d",
        "load_cfd2d",
        "load_airfoil",
        "load_reacdiff",
    ):
        function = getattr(module, name, None)
        if function is not None:
            return function(data_root)
    raise ResultValidationError("The training module has no data loader.")


def _build_evaluation_definition(
    adapter: DatasetAdapter,
    *,
    sample_count: int,
    horizon: int,
    epsilon: float,
    layout: str,
    metric_names: Sequence[str],
) -> dict[str, Any]:
    """Build the complete code-owned metadata for one evaluator family."""

    actual_metrics = set(metric_names)
    if adapter.evaluation_kind == "corrected_rollout":
        expected_metrics = {
            "corrected_step_relative_l2",
            "corrected_trajectory_relative_l2",
        }
        if actual_metrics != expected_metrics:
            raise ResultValidationError(
                "Corrected evaluator produced an unexpected metric set."
            )
        evaluation = importlib.import_module("experiments.common.evaluation")
        definition = evaluation.corrected_relative_l2_metadata(
            sample_count={"test": sample_count},
            horizon=horizon,
            evaluation_space=adapter.evaluation_space,
            epsilon=epsilon,
        )
        if definition["evaluation_tensor_layout"] != layout:
            raise ResultValidationError(
                "Corrected evaluator layout disagrees with its metadata builder."
            )
        return dict(definition)

    if adapter.evaluation_kind == "burgers":
        expected_metrics = {"step_relative_l2", "trajectory_relative_l2"}
        if actual_metrics != expected_metrics:
            raise ResultValidationError(
                "Burgers evaluator produced an unexpected metric set."
            )
        return {
            "evaluation_protocol_version": adapter.evaluation_protocol,
            "evaluation_implementation": (
                "third_party/SirenFNO/neuralop/utils.py:"
                "evaluate_rel_l2_metrics+rel_l2_sum_over_batch+full_rel_l2_rel"
            ),
            "evaluation_space": adapter.evaluation_space,
            "evaluation_tensor_layout": layout,
            "evaluation_epsilon": epsilon,
            "evaluation_zero_reference_policy": "target_l2 + epsilon",
            "evaluation_norm_axes": {
                "step_relative_l2": "C,*spatial",
                "trajectory_relative_l2": "T,C,*spatial",
            },
            "evaluation_aggregation": {
                "step_relative_l2": "arithmetic_mean_over_samples_and_time",
                "trajectory_relative_l2": "arithmetic_mean_over_samples",
            },
            "evaluation_horizon": horizon,
            "evaluation_sample_count": {"test": sample_count},
            "evaluation_metrics": [
                "step_relative_l2",
                "trajectory_relative_l2",
            ],
        }

    if adapter.evaluation_kind == "airfoil":
        expected_metrics = {"relative_l2"}
        implementation = (
            "experiments/train_airfoil.py:evaluate_relative_l2 with pinned "
            "NeuralOperator LpLoss(d=2,p=2,reduction=sum)"
        )
        norm_axes = {"relative_l2": "*spatial (single output channel)"}
        aggregation = {
            "relative_l2": "sum_over_samples_and_channel_then_divide_samples"
        }
    else:
        expected_metrics = {"relative_l2", "h1"}
        implementation = (
            "pinned NeuralOperator decoded LpLoss/H1Loss with reduction=sum"
        )
        norm_axes = {
            "relative_l2": "*spatial (single output channel)",
            "h1": "pinned NeuralOperator H1Loss(d=2,reduction=sum)",
        }
        aggregation = {
            "relative_l2": "sum_over_samples_and_channel_then_divide_samples",
            "h1": "sum_over_samples_and_channel_then_divide_samples",
        }
    if actual_metrics != expected_metrics:
        raise ResultValidationError(
            "Single-step evaluator produced an unexpected metric set."
        )
    return {
        "evaluation_protocol_version": adapter.evaluation_protocol,
        "evaluation_implementation": implementation,
        "evaluation_space": adapter.evaluation_space,
        "evaluation_tensor_layout": layout,
        "evaluation_epsilon": epsilon,
        "evaluation_zero_reference_policy": "target_l2 + epsilon",
        "evaluation_norm_axes": norm_axes,
        "evaluation_aggregation": aggregation,
        "evaluation_horizon": horizon,
        "evaluation_sample_count": {"test": sample_count},
        "evaluation_metrics": sorted(expected_metrics),
    }


def reevaluate_checkpoint(
    *,
    dataset: str,
    checkpoint_path: Path,
    data_root: Path,
    batch_size: int,
    device_name: str,
    output_root: Path | None,
    allow_dirty_source: bool,
    diagnostic_allow_dirty_training_artifact: bool,
    allow_legacy_torch_version: bool,
    allow_legacy_neuraloperator_metadata: bool = False,
    allowed_historical_training_source_commit: str | None = None,
    allowed_historical_checkpoint_sha256: str | None = None,
    overwrite: bool,
) -> Path:
    """Strictly re-evaluate one checkpoint and return its new result path."""

    adapter = DATASETS[dataset]
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ResultValidationError(
            "Checkpoint does not exist.",
            reason_code="input_missing",
            safe_fields={
                "checkpoint": _public_path(
                    checkpoint_path,
                    dataset=dataset,
                    artifact="checkpoint",
                ),
                "dataset": dataset,
            },
        )
    canonical_checkpoint = (
        checkpoint_path.parent / "final_checkpoint.pt"
    ).resolve()
    if checkpoint_path != canonical_checkpoint:
        raise ResultValidationError(
            "Checkpoint input is not the canonical training artifact.",
            reason_code="checkpoint_path_mismatch",
            safe_fields={
                "checkpoint": _public_path(
                    checkpoint_path,
                    dataset=dataset,
                    artifact="checkpoint",
                ),
                "dataset": dataset,
            },
        )
    module = importlib.import_module(adapter.module)
    torch = importlib.import_module("torch")
    sweep = importlib.import_module("scripts.run_paper_sweep")
    completion_spec = sweep.DATASETS[dataset]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ResultValidationError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    source_state_function = getattr(module, "source_repository_state", None)
    source_state = (
        source_state_function(allow_dirty=allow_dirty_source)
        if callable(source_state_function)
        else None
    )
    evaluation_source = (
        _require_nonempty_string(
            source_state.get("source_repository_commit"),
            "evaluation source commit",
        )
        if isinstance(source_state, Mapping)
        else None
    )
    evaluation_dirty = (
        source_state.get("source_repository_dirty")
        if isinstance(source_state, Mapping)
        else None
    )
    if source_state is not None and type(evaluation_dirty) is not bool:
        raise ResultValidationError("Evaluation source dirty state is unavailable.")
    try:
        training_artifact = sweep.validate_training_artifacts(
            checkpoint_path.parent,
            spec=completion_spec,
            current_source_commit=evaluation_source,
            allowed_historical_training_source_commit=(
                allowed_historical_training_source_commit
            ),
            allowed_historical_checkpoint_sha256=(
                allowed_historical_checkpoint_sha256
            ),
            diagnostic_allow_dirty_training_source=(
                diagnostic_allow_dirty_training_artifact
            ),
            allow_legacy_torch_version=allow_legacy_torch_version,
            allow_legacy_neuraloperator_metadata=(
                allow_legacy_neuraloperator_metadata
            ),
        )
    except Exception as exc:
        if not isinstance(exc, sweep.TrainingArtifactValidationError):
            raise
        reasons = tuple(getattr(getattr(exc, "audit", None), "reasons", ()))
        reason_code = (
            "checkpoint_load_failed"
            if any(
                reason in {"checkpoint.unreadable", "checkpoint.not_mapping"}
                for reason in reasons
            )
            else "training_artifact_validation_failed"
        )
        raise ResultValidationError(
            f"Training artifacts failed the shared completion audit: {exc}",
            reason_code=reason_code,
            safe_fields={
                "checkpoint": _public_path(
                    checkpoint_path,
                    dataset=dataset,
                    artifact="checkpoint",
                ),
                "dataset": dataset,
            },
        ) from exc
    summary = dict(training_artifact.summary)
    checkpoint = dict(training_artifact.checkpoint)
    if evaluation_source is None or type(evaluation_dirty) is not bool:
        raise ResultValidationError("Evaluation source provenance is unavailable.")
    checkpoint_digest = training_artifact.checkpoint_sha256
    verified_identity = _verified_dataset_identity(adapter, module, data_root)
    _compare_dataset_identity(
        checkpoint["dataset_identity"],
        verified_identity,
        expected_dataset_id=adapter.dataset_id,
    )
    model_name = training_artifact.model
    seed = training_artifact.seed
    configuration = dict(checkpoint["model_configuration"])
    module.set_seed(seed)
    loaded_data = _load_dataset(adapter, module, data_root)
    if adapter.evaluation_kind == "burgers":
        for key, actual in (
            ("normalization_mean", loaded_data.mean),
            ("normalization_std", loaded_data.std),
        ):
            expected = summary.get(key)
            if expected is None or float(expected) != float(actual):
                raise ResultValidationError(
                    "Burgers normalization state does not match the training metadata."
                )
    model, configuration, parameter_count = _reconstruct_model(
        module,
        checkpoint,
        device=device,
        torch=torch,
    )
    started = time.perf_counter()
    if adapter.evaluation_kind in {"corrected_rollout", "burgers"}:
        metrics, sample_count, horizon, epsilon, layout = _evaluate_rollout(
            adapter,
            module,
            model,
            loaded_data,
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
    elif adapter.evaluation_kind == "airfoil":
        metrics, sample_count, horizon, epsilon, layout = _evaluate_airfoil(
            module,
            model,
            loaded_data,
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
    else:
        metrics, sample_count, horizon, epsilon, layout = _evaluate_neuraloperator(
            module,
            model,
            loaded_data,
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
    elapsed = time.perf_counter() - started
    for name, value in metrics.items():
        _finite_number(value, f"metric {name}")
    if sample_count != int(module.N_TEST):
        raise ResultValidationError(
            f"Evaluation processed {sample_count} samples; expected {module.N_TEST}."
        )

    evaluation_definition = _build_evaluation_definition(
        adapter,
        sample_count=sample_count,
        horizon=horizon,
        epsilon=epsilon,
        layout=layout,
        metric_names=tuple(metrics),
    )

    paper_metrics: dict[str, float] = {}
    for field in completion_spec["paper_metric_columns"]:
        suffix = str(field).removeprefix("final_test_")
        if suffix == "full_relative_l2":
            source_name = "trajectory_relative_l2"
        else:
            source_name = suffix
        if source_name not in metrics:
            raise ResultValidationError(
                f"The evaluator did not produce required paper metric {field}."
            )
        paper_metrics[str(field)] = float(metrics[source_name])

    diagnostic_only = bool(
        training_artifact.training_source_dirty or evaluation_dirty
    )
    diagnostic_reasons = []
    if training_artifact.training_source_dirty:
        diagnostic_reasons.append("dirty_training_source")
    if evaluation_dirty:
        diagnostic_reasons.append("dirty_evaluation_source")
    training_protocol = dict(training_artifact.training_configuration)

    record: dict[str, Any] = {
        "record_type": "checkpoint_evaluation",
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
        "evaluation_schema_version": sweep.EVALUATION_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_id": adapter.dataset_id,
        "experiment": training_artifact.experiment,
        "model": model_name,
        "factorization": checkpoint.get("factorization"),
        "rank": checkpoint.get("rank"),
        "seed": seed,
        "fixed_final_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_policy": summary["checkpoint_selection_policy"],
        "source_checkpoint": _public_path(
            checkpoint_path,
            dataset=dataset,
            model=model_name,
            seed=seed,
            artifact="checkpoint",
        ),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_load_metadata": dict(
            training_artifact.checkpoint_load_metadata
        ),
        "training_source_commit": training_artifact.training_source_commit,
        "training_source_dirty": training_artifact.training_source_dirty,
        "training_source_reuse": dict(training_artifact.training_source_reuse),
        "evaluation_source_commit": evaluation_source,
        "evaluation_source_dirty": evaluation_dirty,
        "training_artifact_validation_status": (
            training_artifact.validation_status
        ),
        "training_artifact_evaluation_classification": (
            training_artifact.evaluation_classification
        ),
        "original_training_evaluation_protocol_version": (
            training_artifact.original_evaluation_protocol_version
        ),
        "training_log_identity": dict(training_artifact.training_log_identity),
        "diagnostic_only": diagnostic_only,
        "paper_eligible": not diagnostic_only,
        "diagnostic_reasons": diagnostic_reasons,
        "training_protocol": training_protocol,
        "training_protocol_sha256": canonical_sha256(training_protocol),
        "training_protocol_version": training_artifact.training_protocol_version,
        "training_configuration": dict(
            training_artifact.training_configuration
        ),
        "data_configuration": dict(training_artifact.data_configuration),
        "training_protocol_identifier": (
            training_artifact.training_protocol_identifier
        ),
        "model_configuration": configuration,
        "model_configuration_sha256": canonical_sha256(configuration),
        "dataset_identity": verified_identity,
        "dataset_identity_sha256": canonical_sha256(verified_identity),
        "evaluation_protocol_version": adapter.evaluation_protocol,
        "evaluation_space": adapter.evaluation_space,
        "evaluation_tensor_layout": layout,
        "evaluation_epsilon": epsilon,
        "evaluation_zero_reference_policy": evaluation_definition[
            "evaluation_zero_reference_policy"
        ],
        "evaluation_horizon": horizon,
        "evaluation_sample_count": {"test": sample_count},
        "evaluation_batch_size": batch_size,
        "evaluation_definition": evaluation_definition,
        "evaluation_definition_sha256": canonical_sha256(evaluation_definition),
        "metrics": metrics,
        "parameter_count": parameter_count,
        "training_time_seconds": summary.get("total_training_time_seconds"),
        "reevaluation_time_seconds": elapsed,
        **paper_metrics,
    }
    _add_recomputed_contract_hashes(record, record)
    _json_bytes(record)

    if output_root is None:
        result_path = (
            checkpoint_path.parent
            / "evaluations"
            / adapter.evaluation_protocol
            / "result.json"
        )
    else:
        result_path = (
            output_root.expanduser()
            / dataset
            / model_name
            / f"seed_{seed}"
            / adapter.evaluation_protocol
            / "result.json"
        )
    if result_path.exists() and not overwrite:
        raise ResultOutputExistsError(
            output=_public_path(
                result_path,
                dataset=dataset,
                model=model_name,
                seed=seed,
                artifact="evaluation-record",
            )
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(result_path, record)
    return result_path


def _require_lower_hex(value: Any, label: str, length: int) -> str:
    rendered = _require_nonempty_string(value, label)
    if len(rendered) != length or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ResultValidationError(
            f"{label} is not a lowercase hexadecimal digest."
        )
    return rendered


def _paper_contract(dataset: str) -> tuple[Any, Mapping[str, Any]]:
    """Return the authoritative sweep contract for a public dataset key."""

    sweep = importlib.import_module("scripts.run_paper_sweep")
    spec = sweep.DATASETS.get(dataset)
    if not isinstance(spec, Mapping):
        raise ResultValidationError("Dataset has no approved paper contract.")
    adapter = DATASETS.get(dataset)
    if (
        adapter is None
        or adapter.dataset_id != spec.get("dataset_id")
        or adapter.evaluation_protocol
        != spec.get("evaluation_protocol_version")
        or adapter.evaluation_space != spec.get("reevaluation_space")
    ):
        raise ResultValidationError(
            "Evaluation adapter disagrees with the approved paper contract."
        )
    if spec.get("epochs") != PAPER_FIXED_FINAL_EPOCH:
        raise ResultValidationError(
            "Approved paper contract is not the fixed 500-epoch protocol."
        )
    return sweep, spec


def _expected_paper_evaluation_definition(dataset: str) -> dict[str, Any]:
    """Derive the full evaluator metadata without trusting an input record."""

    adapter = DATASETS[dataset]
    _sweep, spec = _paper_contract(dataset)
    sample_count = int(spec["data_configuration"]["n_test"])
    horizon = int(spec["evaluation_metadata"]["evaluation_horizon"])
    if adapter.evaluation_kind == "corrected_rollout":
        evaluation = importlib.import_module("experiments.common.evaluation")
        epsilon = float(evaluation.DEFAULT_RELATIVE_L2_EPSILON)
        layout = str(evaluation.CANONICAL_TRAJECTORY_LAYOUT)
    elif adapter.evaluation_kind == "burgers":
        epsilon = 1e-12
        layout = "B,T,C=1,*spatial"
    else:
        module = importlib.import_module(adapter.module)
        loss = module.LpLoss(d=2, p=2, reduction="sum")
        epsilon = float(loss.eps)
        layout = "B,C=1,*spatial"
    return _build_evaluation_definition(
        adapter,
        sample_count=sample_count,
        horizon=horizon,
        epsilon=epsilon,
        layout=layout,
        metric_names=tuple(spec["record_metric_names"]),
    )


def _canonical_dataset_identity(
    sweep: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Recreate the exact public identity returned by dataset verification."""

    manifests = sweep.dataset_manifest()
    dataset_id = str(spec["dataset_id"])
    manifest = manifests.get(dataset_id)
    if not isinstance(manifest, Mapping):
        raise ResultValidationError(
            "Approved dataset is missing from the canonical manifest."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ResultValidationError("Canonical dataset manifest has no files.")
    canonical_files: dict[str, dict[str, Any]] = {}
    for filename, metadata in files.items():
        if not isinstance(filename, str) or not isinstance(metadata, Mapping):
            raise ResultValidationError("Canonical dataset file identity is invalid.")
        canonical_files[filename] = {
            "path": f"data/{filename}",
            "sha256": metadata.get("sha256"),
            "size_bytes": metadata.get("size_bytes"),
        }
    return {
        "dataset_id": dataset_id,
        "source": manifest.get("source"),
        "resolution": manifest.get("resolution"),
        "n_train": manifest.get("n_train"),
        "n_test": manifest.get("n_test"),
        "files": canonical_files,
    }


@functools.lru_cache(maxsize=None)
def _resolved_model_contract(dataset: str, model: str) -> dict[str, Any]:
    """Resolve one real paper model once and retain only its public contract."""

    sweep, spec = _paper_contract(dataset)
    if model not in tuple(spec.get("models", ())):
        raise ResultValidationError(
            "Evaluation model is not in the approved dataset model roster."
        )
    try:
        built = sweep._build_validation_model(spec, model)
        parameter_count = sweep._validation_parameter_count(spec, built.model)
    except Exception as exc:
        raise ResultValidationError(
            "Approved model configuration could not be resolved."
        ) from exc
    configuration = json.loads(_json_bytes(built.configuration).decode("utf-8"))
    return {
        "configuration": configuration,
        "factorization": built.factorization,
        "rank": built.rank,
        "parameter_count": int(parameter_count),
    }


def _training_log_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the per-seed file digest but preserve the CSV protocol shape."""

    return {key: value for key, value in identity.items() if key != "sha256"}


def _validate_training_log_identity(
    value: Any,
    *,
    sweep: Any,
    spec: Mapping[str, Any],
    classification: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ResultValidationError("Training-log identity is empty or missing.")
    identity = dict(value)
    _require_lower_hex(identity.get("sha256"), "training_log_identity.sha256", 64)
    columns = identity.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
        or "epoch" not in columns
    ):
        raise ResultValidationError("Training-log columns are invalid.")
    required_columns = {
        "epoch",
        "learning_rate",
        "train_time_seconds",
        "epoch_time_seconds",
    }
    missing_columns = required_columns - set(columns)
    if missing_columns:
        raise ResultValidationError(
            "Training-log identity is missing required production columns."
        )
    for key, expected in (
        ("row_count", PAPER_FIXED_FINAL_EPOCH),
        ("first_epoch", 1),
        ("final_epoch", PAPER_FIXED_FINAL_EPOCH),
    ):
        actual = identity.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ResultValidationError(
                f"Training-log {key} does not match the complete paper protocol."
            )
    expected_epoch_hash = canonical_sha256(
        list(range(1, PAPER_FIXED_FINAL_EPOCH + 1))
    )
    if identity.get("epoch_sequence_sha256") != expected_epoch_hash:
        raise ResultValidationError(
            "Training-log epoch sequence does not identify epochs 1 through 500."
        )
    metric_columns = identity.get("metric_columns")
    if (
        not isinstance(metric_columns, Mapping)
        or not metric_columns
        or any(
            not isinstance(metric, str)
            or not metric
            or not isinstance(column, str)
            or not column
            or column not in columns
            for metric, column in metric_columns.items()
        )
    ):
        raise ResultValidationError("Training-log metric columns are invalid.")
    expected_current = dict(spec["metric_columns"])
    if classification == "current":
        allowed_metric_contracts = (expected_current,)
    else:
        allowed_metric_contracts = (
            dict(sweep._legacy_metric_columns(spec)),
            expected_current,
        )
    if not any(
        _same_json(metric_columns, expected) for expected in allowed_metric_contracts
    ):
        raise ResultValidationError(
            "Training-log metric columns do not match the approved protocol."
        )
    return identity


def _training_provenance_payload(
    record: Mapping[str, Any],
    *,
    include_per_seed_identity: bool,
) -> dict[str, Any]:
    log_identity = dict(record["training_log_identity"])
    if not include_per_seed_identity:
        log_identity = _training_log_contract(log_identity)
    reuse = record.get("training_source_reuse")
    aggregation_source_commit = record["training_source_commit"]
    if isinstance(reuse, Mapping):
        aggregation_source_commit = reuse.get(
            "aggregation_source_commit", aggregation_source_commit
        )
    payload = {
        "training_source_commit": (
            record["training_source_commit"]
            if include_per_seed_identity
            else aggregation_source_commit
        ),
        "training_source_dirty": record["training_source_dirty"],
        "training_artifact_validation_status": record[
            "training_artifact_validation_status"
        ],
        "training_artifact_evaluation_classification": record[
            "training_artifact_evaluation_classification"
        ],
        "original_training_evaluation_protocol_version": record[
            "original_training_evaluation_protocol_version"
        ],
        "training_log_identity": log_identity,
    }
    if include_per_seed_identity:
        payload["checkpoint_load_metadata"] = record[
            "checkpoint_load_metadata"
        ]
        if isinstance(reuse, Mapping):
            payload["training_source_reuse"] = dict(reuse)
    return payload


def _evaluation_provenance_payload(
    record: Mapping[str, Any],
    *,
    include_batch_size: bool,
) -> dict[str, Any]:
    payload = {
        "evaluation_schema_version": record["evaluation_schema_version"],
        "evaluation_protocol_version": record["evaluation_protocol_version"],
        "evaluation_source_commit": record["evaluation_source_commit"],
        "evaluation_source_dirty": record["evaluation_source_dirty"],
        "evaluation_space": record["evaluation_space"],
        "evaluation_tensor_layout": record["evaluation_tensor_layout"],
        "evaluation_epsilon": record["evaluation_epsilon"],
        "evaluation_zero_reference_policy": record[
            "evaluation_zero_reference_policy"
        ],
        "evaluation_horizon": record["evaluation_horizon"],
        "evaluation_sample_count": record["evaluation_sample_count"],
        "evaluation_definition": record["evaluation_definition"],
    }
    if include_batch_size:
        payload["evaluation_batch_size"] = record["evaluation_batch_size"]
    return payload


def _paper_group_contract_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return every stable condition that may define one paper mean.

    Per-seed file hashes/timings and evaluation batch size are deliberately
    excluded.  The approved evaluators are batch-size invariant, while all
    mathematical evaluation metadata remains in ``evaluation_definition``.
    """

    return {
        "dataset": record["dataset"],
        "dataset_id": record["dataset_id"],
        "experiment": record["experiment"],
        "fixed_final_epoch": record["fixed_final_epoch"],
        "checkpoint_selection_policy": record["checkpoint_selection_policy"],
        "model": record["model"],
        "factorization": record.get("factorization"),
        "rank": record.get("rank"),
        "parameter_count": record["parameter_count"],
        "model_configuration": record["model_configuration"],
        "training_protocol_version": record["training_protocol_version"],
        "training_protocol_identifier": record["training_protocol_identifier"],
        "training_protocol": record["training_protocol"],
        "training_configuration": record["training_configuration"],
        "data_configuration": record["data_configuration"],
        "dataset_identity": record["dataset_identity"],
        "training_provenance": _training_provenance_payload(
            record, include_per_seed_identity=False
        ),
        "evaluation_provenance": _evaluation_provenance_payload(
            record, include_batch_size=False
        ),
        "diagnostic_only": record["diagnostic_only"],
        "paper_eligible": record["paper_eligible"],
    }


def _add_recomputed_contract_hashes(
    validated: dict[str, Any],
    original: Mapping[str, Any],
) -> None:
    payloads = {
        "training_configuration_sha256": validated["training_configuration"],
        "data_configuration_sha256": validated["data_configuration"],
        "training_provenance_sha256": _training_provenance_payload(
            validated, include_per_seed_identity=True
        ),
        "evaluation_provenance_sha256": _evaluation_provenance_payload(
            validated, include_batch_size=True
        ),
        "paper_group_contract_sha256": _paper_group_contract_payload(validated),
    }
    for field, payload in payloads.items():
        actual = canonical_sha256(payload)
        supplied = original.get(field)
        if supplied is not None and supplied != actual:
            raise ResultValidationError(f"{field} does not match its payload.")
        validated[field] = actual


def _validate_training_source_reuse(record: Mapping[str, Any]) -> None:
    """Validate optional exact historical-source evidence for schema-v2 records."""

    reuse = record.get("training_source_reuse")
    if reuse is None:
        # Backward compatibility for records written before this evidence field.
        return
    if not isinstance(reuse, Mapping):
        raise ResultValidationError("Training source reuse evidence is invalid.")
    policy = reuse.get("policy")
    aggregation_commit = reuse.get("aggregation_source_commit")
    _require_lower_hex(
        aggregation_commit, "training source aggregation commit", 40
    )
    if policy == "current_source_commit":
        if set(reuse) != {"policy", "aggregation_source_commit"}:
            raise ResultValidationError(
                "Current-source reuse evidence has unexpected fields."
            )
        if aggregation_commit != record["training_source_commit"]:
            raise ResultValidationError(
                "Current-source reuse evidence disagrees with training provenance."
            )
        return
    if policy != "exact_historical_commit_and_checkpoint_sha256" or set(
        reuse
    ) != {
        "policy",
        "aggregation_source_commit",
        "historical_training_source_commit",
        "historical_checkpoint_sha256",
        "verified_change_scope",
        "source_diff_sha256",
        "changed_paths",
    }:
        raise ResultValidationError("Historical source reuse evidence is invalid.")
    _require_lower_hex(
        reuse.get("historical_training_source_commit"),
        "historical training source commit",
        40,
    )
    _require_lower_hex(
        reuse.get("historical_checkpoint_sha256"),
        "historical checkpoint sha256",
        64,
    )
    _require_lower_hex(reuse.get("source_diff_sha256"), "source diff sha256", 64)
    changed_paths = reuse.get("changed_paths")
    allowed_paths = {
        "README.md",
        "experiments/common/checkpoints.py",
        "scripts/run_paper_sweep.py",
        "scripts/summarize_results.py",
        "tests/test_results.py",
        "tests/test_sweep.py",
    }
    if (
        not isinstance(changed_paths, list)
        or not changed_paths
        or len(set(changed_paths)) != len(changed_paths)
        or any(type(path) is not str or path not in allowed_paths for path in changed_paths)
    ):
        raise ResultValidationError("Historical source diff path evidence is invalid.")
    if (
        reuse["historical_training_source_commit"]
        != record["training_source_commit"]
        or reuse["historical_checkpoint_sha256"] != record["checkpoint_sha256"]
        or aggregation_commit != record["evaluation_source_commit"]
        or reuse["verified_change_scope"]
        != "checkpoint_storage_loading_validation_only"
    ):
        raise ResultValidationError(
            "Historical source reuse evidence disagrees with exact provenance."
        )


def validate_evaluation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record against code-owned paper contracts, not its claims."""

    if record.get("record_type") != "checkpoint_evaluation":
        raise ResultValidationError("Input is not a checkpoint evaluation record.")
    if record.get("schema_version") != EVALUATION_RECORD_SCHEMA_VERSION:
        raise ResultValidationError("Unsupported evaluation record schema version.")
    dataset = _require_nonempty_string(record.get("dataset"), "dataset")
    adapter = DATASETS.get(dataset)
    if adapter is None:
        raise ResultValidationError(f"Unsupported evaluation dataset: {dataset}")
    sweep, spec = _paper_contract(dataset)
    if record.get("dataset_id") != spec["dataset_id"]:
        raise ResultValidationError(
            "Public dataset key and internal dataset id do not match."
        )
    if record.get("evaluation_schema_version") != sweep.EVALUATION_SCHEMA_VERSION:
        raise ResultValidationError("Unsupported completion evaluation schema.")
    if record.get("evaluation_protocol_version") != spec[
        "evaluation_protocol_version"
    ]:
        raise ResultValidationError(
            "Legacy or incompatible evaluation protocol is not paper-aggregatable."
        )
    for key in (
        "dataset_id",
        "experiment",
        "model",
        "source_checkpoint",
        "checkpoint_sha256",
        "training_source_commit",
        "evaluation_source_commit",
        "training_protocol_version",
        "training_protocol_identifier",
        "training_protocol_sha256",
        "model_configuration_sha256",
        "dataset_identity_sha256",
        "evaluation_schema_version",
        "evaluation_protocol_version",
        "evaluation_space",
        "evaluation_tensor_layout",
        "evaluation_zero_reference_policy",
        "evaluation_definition_sha256",
        "checkpoint_selection_policy",
        "training_artifact_validation_status",
        "training_artifact_evaluation_classification",
    ):
        _require_nonempty_string(record.get(key), key)
    for key, length in (
        ("checkpoint_sha256", 64),
        ("model_configuration_sha256", 64),
        ("dataset_identity_sha256", 64),
        ("evaluation_definition_sha256", 64),
        ("training_protocol_sha256", 64),
        ("training_source_commit", 40),
        ("evaluation_source_commit", 40),
    ):
        _require_lower_hex(record.get(key), key, length)

    _validate_training_source_reuse(record)

    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ResultValidationError("Evaluation seed must be an integer.")
    for key in ("training_source_dirty", "evaluation_source_dirty"):
        if type(record.get(key)) is not bool:
            raise ResultValidationError(f"{key} must be an explicit boolean.")
        if record[key]:
            raise ResultValidationError(
                "Dirty-source diagnostic evaluations cannot enter a paper aggregate."
            )
    if type(record.get("diagnostic_only")) is not bool:
        raise ResultValidationError("diagnostic_only must be an explicit boolean.")
    if type(record.get("paper_eligible")) is not bool:
        raise ResultValidationError("paper_eligible must be an explicit boolean.")
    if record["diagnostic_only"] or not record["paper_eligible"]:
        raise ResultValidationError(
            "Diagnostic or unverified evaluations cannot enter a paper aggregate."
        )
    if record.get("diagnostic_reasons") not in (None, []):
        raise ResultValidationError("Paper-eligible record has diagnostic reasons.")

    classification = record["training_artifact_evaluation_classification"]
    status = record["training_artifact_validation_status"]
    expected_status = {
        "current": "valid_training_current_evaluation",
        "legacy": "valid_training_legacy_evaluation",
    }.get(classification)
    if status != expected_status:
        raise ResultValidationError(
            "Training validation status and evaluation classification disagree."
        )
    original_protocol = record.get("original_training_evaluation_protocol_version")
    if classification == "current":
        if original_protocol != spec["evaluation_protocol_version"]:
            raise ResultValidationError(
                "Current training evaluation provenance is inconsistent."
            )
    elif original_protocol is not None:
        _require_nonempty_string(
            original_protocol, "original_training_evaluation_protocol_version"
        )
        if original_protocol == spec["evaluation_protocol_version"]:
            raise ResultValidationError(
                "Legacy training evaluation provenance claims the current protocol."
            )

    checkpoint_load_metadata = record.get("checkpoint_load_metadata")
    if not isinstance(checkpoint_load_metadata, Mapping) or set(
        checkpoint_load_metadata
    ) not in (
        {
            "checkpoint_load_mode",
            "legacy_torch_version_compatibility_used",
        },
        {
            "checkpoint_load_mode",
            "legacy_torch_version_compatibility_used",
            "legacy_neuraloperator_metadata_compatibility_used",
        },
    ):
        raise ResultValidationError("Checkpoint load metadata is missing or invalid.")
    if checkpoint_load_metadata["checkpoint_load_mode"] != "weights_only":
        raise ResultValidationError("Checkpoint load mode is not weights-only.")
    if type(
        checkpoint_load_metadata["legacy_torch_version_compatibility_used"]
    ) is not bool:
        raise ResultValidationError(
            "Legacy checkpoint compatibility flag must be an explicit boolean."
        )
    neuralop_compatibility = checkpoint_load_metadata.get(
        "legacy_neuraloperator_metadata_compatibility_used", False
    )
    if type(neuralop_compatibility) is not bool:
        raise ResultValidationError(
            "Legacy neuraloperator metadata compatibility flag must be an "
            "explicit boolean."
        )

    count = _evaluation_test_count(record.get("evaluation_sample_count"))
    horizon = record.get("evaluation_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ResultValidationError("Evaluation horizon must be positive.")
    epsilon = _finite_number(record.get("evaluation_epsilon"), "evaluation_epsilon")
    if epsilon <= 0:
        raise ResultValidationError("evaluation_epsilon must be strictly positive.")
    for key in ("fixed_final_epoch", "parameter_count", "evaluation_batch_size"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResultValidationError(f"{key} must be a positive integer.")
    if record["fixed_final_epoch"] != PAPER_FIXED_FINAL_EPOCH:
        raise ResultValidationError(
            "Evaluation is not from the approved fixed final epoch 500."
        )
    expected_training = spec["training_configuration"]
    expected_data = spec["data_configuration"]
    expected_selection = expected_training["checkpoint_selection_policy"]
    if record["checkpoint_selection_policy"] != expected_selection:
        raise ResultValidationError("Checkpoint selection is not the approved policy.")
    if record["experiment"] != spec.get("experiment"):
        raise ResultValidationError("Evaluation is not from the approved paper experiment.")

    configuration = record.get("model_configuration")
    identity = record.get("dataset_identity")
    protocol = record.get("training_protocol")
    training_configuration = record.get("training_configuration")
    data_configuration = record.get("data_configuration")
    evaluation_definition = record.get("evaluation_definition")
    for value, label in (
        (configuration, "model_configuration"),
        (identity, "dataset_identity"),
        (protocol, "training_protocol"),
        (training_configuration, "training_configuration"),
        (data_configuration, "data_configuration"),
        (evaluation_definition, "evaluation_definition"),
    ):
        if not isinstance(value, Mapping) or not value:
            raise ResultValidationError(f"{label} payload is empty or missing.")
    if canonical_sha256(configuration) != record.get("model_configuration_sha256"):
        raise ResultValidationError("Model configuration hash mismatch.")
    if canonical_sha256(identity) != record.get("dataset_identity_sha256"):
        raise ResultValidationError("Dataset identity hash mismatch.")
    if canonical_sha256(evaluation_definition) != record.get(
        "evaluation_definition_sha256"
    ):
        raise ResultValidationError("Evaluation definition hash mismatch.")
    if canonical_sha256(protocol) != record.get("training_protocol_sha256"):
        raise ResultValidationError("Training protocol hash mismatch.")
    expected_evaluation_definition = _expected_paper_evaluation_definition(
        dataset
    )
    if not _same_json(evaluation_definition, expected_evaluation_definition):
        raise ResultValidationError(
            "Evaluation definition does not match the approved evaluator contract."
        )

    if not _same_json(training_configuration, expected_training):
        raise ResultValidationError(
            "Training configuration does not match the approved paper contract."
        )
    if not _same_json(protocol, expected_training):
        raise ResultValidationError(
            "Training protocol does not match the approved paper contract."
        )
    if training_configuration.get("epochs") != record["fixed_final_epoch"]:
        raise ResultValidationError("Training configuration epoch is inconsistent.")
    if protocol.get("epochs") != record["fixed_final_epoch"]:
        raise ResultValidationError("Training protocol epoch is inconsistent.")
    if record["training_protocol_version"] != spec["training_protocol_version"]:
        raise ResultValidationError("Training protocol version is not approved.")
    expected_identifier = sweep.training_protocol_identifier(spec)
    if record["training_protocol_identifier"] != expected_identifier:
        raise ResultValidationError("Training protocol identifier is not approved.")
    if not _same_json(data_configuration, expected_data):
        raise ResultValidationError(
            "Data configuration does not match the approved paper contract."
        )

    canonical_identity = _canonical_dataset_identity(sweep, spec)
    if not _same_json(identity, canonical_identity):
        raise ResultValidationError(
            "Dataset identity does not match the canonical dataset manifest."
        )
    if identity.get("dataset_id") != record["dataset_id"]:
        raise ResultValidationError("Dataset identity has the wrong internal id.")
    for key in ("resolution", "n_train", "n_test"):
        if not _same_json(identity.get(key), data_configuration.get(key)):
            raise ResultValidationError(
                f"Dataset identity and data configuration disagree on {key}."
            )
    if horizon != data_configuration.get("prediction_horizon"):
        raise ResultValidationError(
            "Evaluation horizon disagrees with the approved data configuration."
        )
    if count != data_configuration.get("n_test"):
        raise ResultValidationError(
            "Evaluation sample count disagrees with the approved test split."
        )

    model_name = str(record["model"])
    resolved_model = _resolved_model_contract(dataset, model_name)
    if not _same_json(configuration, resolved_model["configuration"]):
        raise ResultValidationError(
            "Model configuration does not match the resolved paper model."
        )
    for key in ("factorization", "rank"):
        if not _same_json(record.get(key), resolved_model[key]):
            raise ResultValidationError(
                f"Top-level {key} does not match the resolved paper model."
            )
        if key in configuration and not _same_json(
            configuration[key], record.get(key)
        ):
            raise ResultValidationError(
                f"Model configuration and top-level {key} disagree."
            )
    if record["parameter_count"] != resolved_model["parameter_count"]:
        raise ResultValidationError(
            "Parameter count does not match the resolved paper model."
        )

    training_log_identity = _validate_training_log_identity(
        record.get("training_log_identity"),
        sweep=sweep,
        spec=spec,
        classification=classification,
    )

    if record["evaluation_space"] != spec.get("reevaluation_space"):
        raise ResultValidationError("Evaluation space is not the approved space.")
    for key, expected in spec["evaluation_metadata"].items():
        if key == "evaluation_sample_count":
            if count != _evaluation_test_count(expected):
                raise ResultValidationError(
                    "Evaluation sample count is not the approved sample count."
                )
        elif not _same_json(record.get(key), expected):
            raise ResultValidationError(
                f"Evaluation metadata {key} is not approved."
            )
    for key, expected in (
        ("evaluation_protocol_version", record["evaluation_protocol_version"]),
        ("evaluation_space", record["evaluation_space"]),
        ("evaluation_tensor_layout", record["evaluation_tensor_layout"]),
        ("evaluation_epsilon", record["evaluation_epsilon"]),
        (
            "evaluation_zero_reference_policy",
            record["evaluation_zero_reference_policy"],
        ),
        ("evaluation_horizon", horizon),
        ("evaluation_sample_count", record["evaluation_sample_count"]),
    ):
        if key not in evaluation_definition:
            raise ResultValidationError(
                f"Evaluation definition is missing required field {key}."
            )
        if not _same_json(evaluation_definition[key], expected):
            raise ResultValidationError(
                f"Evaluation definition disagrees with top-level {key}."
            )

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ResultValidationError("Evaluation metrics are empty or missing.")
    for name, value in metrics.items():
        _require_nonempty_string(name, "metric name")
        if _finite_number(value, f"metric {name}") < 0:
            raise ResultValidationError(f"metric {name} cannot be negative.")
    expected_metric_map = PAPER_METRIC_FIELDS[dataset]
    expected_metrics = set(spec["record_metric_names"])
    if set(metrics) != expected_metrics or set(expected_metric_map.values()) != expected_metrics:
        raise ResultValidationError("Evaluation metric set is incomplete or unexpected.")
    declared_metrics = evaluation_definition.get("evaluation_metrics")
    if not isinstance(declared_metrics, list) or set(declared_metrics) != set(metrics):
        raise ResultValidationError(
            "Evaluation definition metric declaration is incomplete or unexpected."
        )
    for paper_field, metric_name in expected_metric_map.items():
        paper_value = _finite_number(record.get(paper_field), paper_field)
        metric_value = float(metrics[metric_name])
        if paper_value < 0:
            raise ResultValidationError(f"{paper_field} cannot be negative.")
        if not math.isclose(paper_value, metric_value, rel_tol=1e-12, abs_tol=1e-15):
            raise ResultValidationError(
                f"Top-level paper metric disagrees with metrics.{metric_name}."
            )
    training_time = record.get("training_time_seconds")
    if training_time is not None:
        if _finite_number(training_time, "training_time_seconds") < 0:
            raise ResultValidationError("training_time_seconds cannot be negative.")
    reevaluation_time = record.get("reevaluation_time_seconds")
    if reevaluation_time is not None and _finite_number(
        reevaluation_time, "reevaluation_time_seconds"
    ) < 0:
        raise ResultValidationError("reevaluation_time_seconds cannot be negative.")

    _json_bytes(record)
    validated = dict(record)
    validated["training_log_identity"] = training_log_identity
    _add_recomputed_contract_hashes(validated, record)
    return validated


def _evaluation_json_paths(inputs: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    for source in inputs:
        source = source.expanduser()
        if source.is_file():
            discovered.append(source)
        elif source.is_dir():
            discovered.extend(source.rglob("result.json"))
        else:
            raise ResultValidationError(
                "Evaluation input does not exist.",
                reason_code="input_missing",
                safe_fields={
                    "input": _public_path(source, artifact="evaluation-record")
                },
            )
    unique = sorted({path.resolve() for path in discovered})
    if not unique:
        raise ResultValidationError("No result.json evaluation records were found.")
    return unique


def load_evaluation_records(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _evaluation_json_paths(inputs):
        record = validate_evaluation_record(_load_json(path))
        record["_input_record"] = _public_path(
            path,
            dataset=str(record["dataset"]),
            model=str(record["model"]),
            seed=int(record["seed"]),
            artifact="evaluation-record",
        )
        records.append(record)
    return records


def _group_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    # Recompute from the complete payload.  Never trust an input-supplied hash,
    # and deliberately omit only per-seed artifact identities/timings and the
    # mathematically irrelevant evaluation batch size.
    return (canonical_sha256(_paper_group_contract_payload(record)),)


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    required_seeds: Sequence[int],
    allow_partial: bool,
) -> dict[str, Any]:
    expected_seeds = tuple(int(seed) for seed in required_seeds)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ResultValidationError("Required seeds must be nonempty and unique.")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    checkpoint_owners: dict[str, tuple[str, str, int]] = {}
    for source in records:
        record = validate_evaluation_record(source)
        checkpoint_digest = str(record["checkpoint_sha256"])
        owner = (str(record["dataset"]), str(record["model"]), int(record["seed"]))
        if (
            checkpoint_digest in checkpoint_owners
            and checkpoint_owners[checkpoint_digest] != owner
        ):
            raise ResultValidationError(
                "Duplicate checkpoint/run detected in aggregation inputs."
            )
        checkpoint_owners[checkpoint_digest] = owner
        groups.setdefault(_group_signature(record), []).append(record)

    aggregates: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    expected_set = set(expected_seeds)
    for _signature, group in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = [int(record["seed"]) for record in group]
        duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        if duplicates:
            raise ResultValidationError(
                "Duplicate seed/run detected for one protocol group: "
                + ", ".join(map(str, duplicates))
            )
        unexpected = sorted(set(seeds) - expected_set)
        if unexpected:
            raise ResultValidationError(
                "Unexpected seed(s) for paper aggregation: "
                + ", ".join(map(str, unexpected))
            )
        missing = sorted(expected_set - set(seeds))
        if missing and not allow_partial:
            raise ResultValidationError(
                "Missing required seed(s): " + ", ".join(map(str, missing))
            )
        metric_names = set(group[0]["metrics"])
        if any(set(record["metrics"]) != metric_names for record in group[1:]):
            raise ResultValidationError("Metric fields differ within one protocol group.")
        if len({record["parameter_count"] for record in group}) != 1:
            raise ResultValidationError(
                "Parameter counts disagree within one model configuration group."
            )
        ordered_group = sorted(group, key=lambda record: int(record["seed"]))
        for record in ordered_group:
            raw.append(
                {
                    "dataset": record["dataset"],
                    "dataset_id": record["dataset_id"],
                    "experiment": record["experiment"],
                    "model": record["model"],
                    "seed": record["seed"],
                    "fixed_final_epoch": record["fixed_final_epoch"],
                    "checkpoint_selection_policy": record[
                        "checkpoint_selection_policy"
                    ],
                    "factorization": record.get("factorization"),
                    "rank": record.get("rank"),
                    "parameter_count": record.get("parameter_count"),
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "training_source_commit": record["training_source_commit"],
                    "training_source_dirty": record["training_source_dirty"],
                    "training_source_reuse_policy": (
                        record.get("training_source_reuse") or {}
                    ).get("policy"),
                    "aggregation_source_commit": (
                        record.get("training_source_reuse") or {}
                    ).get(
                        "aggregation_source_commit",
                        record["training_source_commit"],
                    ),
                    "historical_training_source_commit": (
                        record.get("training_source_reuse") or {}
                    ).get("historical_training_source_commit"),
                    "historical_checkpoint_sha256": (
                        record.get("training_source_reuse") or {}
                    ).get("historical_checkpoint_sha256"),
                    "verified_change_scope": (
                        record.get("training_source_reuse") or {}
                    ).get("verified_change_scope"),
                    "source_diff_sha256": (
                        record.get("training_source_reuse") or {}
                    ).get("source_diff_sha256"),
                    "source_diff_changed_paths": " ".join(
                        (record.get("training_source_reuse") or {}).get(
                            "changed_paths", []
                        )
                    ),
                    "training_artifact_validation_status": record[
                        "training_artifact_validation_status"
                    ],
                    "training_artifact_evaluation_classification": record[
                        "training_artifact_evaluation_classification"
                    ],
                    "original_training_evaluation_protocol_version": record[
                        "original_training_evaluation_protocol_version"
                    ],
                    "training_log_sha256": record["training_log_identity"][
                        "sha256"
                    ],
                    "training_log_epoch_sequence_sha256": record[
                        "training_log_identity"
                    ]["epoch_sequence_sha256"],
                    "checkpoint_load_mode": record["checkpoint_load_metadata"][
                        "checkpoint_load_mode"
                    ],
                    "legacy_torch_version_compatibility_used": record[
                        "checkpoint_load_metadata"
                    ]["legacy_torch_version_compatibility_used"],
                    "legacy_neuraloperator_metadata_compatibility_used": record[
                        "checkpoint_load_metadata"
                    ].get(
                        "legacy_neuraloperator_metadata_compatibility_used",
                        False,
                    ),
                    "evaluation_source_commit": record["evaluation_source_commit"],
                    "evaluation_source_dirty": record["evaluation_source_dirty"],
                    "evaluation_protocol_version": record[
                        "evaluation_protocol_version"
                    ],
                    "evaluation_batch_size": record["evaluation_batch_size"],
                    "evaluation_definition_sha256": record[
                        "evaluation_definition_sha256"
                    ],
                    "model_configuration_sha256": record[
                        "model_configuration_sha256"
                    ],
                    "training_configuration_sha256": record[
                        "training_configuration_sha256"
                    ],
                    "data_configuration_sha256": record[
                        "data_configuration_sha256"
                    ],
                    "training_protocol_identifier": record[
                        "training_protocol_identifier"
                    ],
                    "training_protocol_sha256": record[
                        "training_protocol_sha256"
                    ],
                    "dataset_identity_sha256": record["dataset_identity_sha256"],
                    "training_provenance_sha256": record[
                        "training_provenance_sha256"
                    ],
                    "evaluation_provenance_sha256": record[
                        "evaluation_provenance_sha256"
                    ],
                    "paper_group_contract_sha256": record[
                        "paper_group_contract_sha256"
                    ],
                    "diagnostic_only": record["diagnostic_only"],
                    "paper_eligible": record["paper_eligible"],
                    "training_time_seconds": record.get("training_time_seconds"),
                    "metrics": dict(record["metrics"]),
                }
            )
        base = ordered_group[0]
        metric_aggregates: dict[str, Any] = {}
        for metric_name in sorted(metric_names):
            values = [float(record["metrics"][metric_name]) for record in ordered_group]
            metric_aggregates[metric_name] = {
                "mean": statistics.fmean(values),
                "sample_standard_deviation_ddof_1": (
                    statistics.stdev(values) if len(values) >= 2 else None
                ),
            }
        training_times = [record.get("training_time_seconds") for record in ordered_group]
        if all(value is not None for value in training_times):
            time_values = [float(value) for value in training_times]
            timing = {
                "mean_training_time_seconds": statistics.fmean(time_values),
                "sample_standard_deviation_ddof_1": (
                    statistics.stdev(time_values) if len(time_values) >= 2 else None
                ),
            }
        else:
            timing = None
        training_source_commits = sorted(
            {str(record["training_source_commit"]) for record in ordered_group}
        )
        reuse_evidence = [
            {
                "seed": int(record["seed"]),
                **dict(record["training_source_reuse"]),
            }
            for record in ordered_group
            if isinstance(record.get("training_source_reuse"), Mapping)
        ]
        aggregate_checkpoint_load_metadata = {
            "checkpoint_load_mode": "weights_only",
            "legacy_torch_version_compatibility_used": any(
                bool(
                    record["checkpoint_load_metadata"][
                        "legacy_torch_version_compatibility_used"
                    ]
                )
                for record in ordered_group
            ),
            "legacy_neuraloperator_metadata_compatibility_used": any(
                bool(
                    record["checkpoint_load_metadata"].get(
                        "legacy_neuraloperator_metadata_compatibility_used",
                        False,
                    )
                )
                for record in ordered_group
            ),
        }
        aggregates.append(
            {
                "dataset": base["dataset"],
                "dataset_id": base["dataset_id"],
                "experiment": base["experiment"],
                "model": base["model"],
                "fixed_final_epoch": base["fixed_final_epoch"],
                "checkpoint_selection_policy": base[
                    "checkpoint_selection_policy"
                ],
                "factorization": base.get("factorization"),
                "rank": base.get("rank"),
                "parameter_count": base.get("parameter_count"),
                "seed_count": len(ordered_group),
                "seeds": [record["seed"] for record in ordered_group],
                "missing_required_seeds": missing,
                "partial": bool(missing),
                "training_source_commit": (
                    training_source_commits[0]
                    if len(training_source_commits) == 1
                    else None
                ),
                "training_source_commits": training_source_commits,
                "training_source_reuse_evidence": reuse_evidence,
                "training_source_dirty": base["training_source_dirty"],
                "training_artifact_validation_status": base[
                    "training_artifact_validation_status"
                ],
                "training_artifact_evaluation_classification": base[
                    "training_artifact_evaluation_classification"
                ],
                "original_training_evaluation_protocol_version": base[
                    "original_training_evaluation_protocol_version"
                ],
                "training_log_contract_sha256": canonical_sha256(
                    _training_log_contract(base["training_log_identity"])
                ),
                "checkpoint_load_metadata": aggregate_checkpoint_load_metadata,
                "evaluation_source_commit": base["evaluation_source_commit"],
                "evaluation_source_dirty": base["evaluation_source_dirty"],
                "evaluation_batch_sizes": sorted(
                    {int(record["evaluation_batch_size"]) for record in ordered_group}
                ),
                "evaluation_protocol_version": base[
                    "evaluation_protocol_version"
                ],
                "evaluation_definition_sha256": base[
                    "evaluation_definition_sha256"
                ],
                "model_configuration_sha256": base[
                    "model_configuration_sha256"
                ],
                "training_configuration_sha256": base[
                    "training_configuration_sha256"
                ],
                "data_configuration_sha256": base[
                    "data_configuration_sha256"
                ],
                "training_protocol_identifier": base[
                    "training_protocol_identifier"
                ],
                "training_protocol_sha256": base["training_protocol_sha256"],
                "dataset_identity_sha256": base["dataset_identity_sha256"],
                "paper_group_contract_sha256": base[
                    "paper_group_contract_sha256"
                ],
                "diagnostic_only": base["diagnostic_only"],
                "paper_eligible": base["paper_eligible"],
                "metrics": metric_aggregates,
                "training_timing": timing,
            }
        )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "partial": any(item["partial"] for item in aggregates),
        "required_seeds": list(expected_seeds),
        "standard_deviation": "sample standard deviation (ddof=1); null for n=1",
        "raw_results": raw,
        "aggregates": aggregates,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.8g}"


def _latex_escape(value: Any) -> str:
    rendered = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in rendered)


def _write_aggregate_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    paths = (
        output_dir / "seed_results.csv",
        output_dir / "aggregate_results.csv",
        output_dir / "paper_table.md",
        output_dir / "paper_table.tex",
        output_dir / "results.json",
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ResultOutputExistsError(
            output=_public_path(output_dir, artifact="aggregate-output")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    metric_names = sorted(
        {
            metric
            for row in report["raw_results"]
            for metric in row["metrics"]
        }
    )
    for row in report["raw_results"]:
        raw_rows.append(
            {
                **{key: value for key, value in row.items() if key != "metrics"},
                **{name: row["metrics"].get(name) for name in metric_names},
            }
        )
    raw_fields = [
        "dataset",
        "dataset_id",
        "experiment",
        "model",
        "seed",
        "fixed_final_epoch",
        "checkpoint_selection_policy",
        "factorization",
        "rank",
        "parameter_count",
        "checkpoint_sha256",
        "training_source_commit",
        "training_source_dirty",
        "training_source_reuse_policy",
        "aggregation_source_commit",
        "historical_training_source_commit",
        "historical_checkpoint_sha256",
        "verified_change_scope",
        "source_diff_sha256",
        "source_diff_changed_paths",
        "training_artifact_validation_status",
        "training_artifact_evaluation_classification",
        "original_training_evaluation_protocol_version",
        "training_log_sha256",
        "training_log_epoch_sequence_sha256",
        "checkpoint_load_mode",
        "legacy_torch_version_compatibility_used",
        "legacy_neuraloperator_metadata_compatibility_used",
        "evaluation_source_commit",
        "evaluation_source_dirty",
        "evaluation_protocol_version",
        "evaluation_batch_size",
        "evaluation_definition_sha256",
        "model_configuration_sha256",
        "training_configuration_sha256",
        "data_configuration_sha256",
        "training_protocol_identifier",
        "training_protocol_sha256",
        "dataset_identity_sha256",
        "training_provenance_sha256",
        "evaluation_provenance_sha256",
        "paper_group_contract_sha256",
        "diagnostic_only",
        "paper_eligible",
        "training_time_seconds",
        *metric_names,
    ]
    with paths[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for group in report["aggregates"]:
        for metric_name, values in group["metrics"].items():
            timing = group["training_timing"]
            aggregate_rows.append(
                {
                    "dataset": group["dataset"],
                    "dataset_id": group["dataset_id"],
                    "experiment": group["experiment"],
                    "model": group["model"],
                    "fixed_final_epoch": group["fixed_final_epoch"],
                    "checkpoint_selection_policy": group[
                        "checkpoint_selection_policy"
                    ],
                    "factorization": group["factorization"],
                    "rank": group["rank"],
                    "parameter_count": group["parameter_count"],
                    "metric": metric_name,
                    "mean": values["mean"],
                    "sample_standard_deviation_ddof_1": values[
                        "sample_standard_deviation_ddof_1"
                    ],
                    "seed_count": group["seed_count"],
                    "seeds": " ".join(map(str, group["seeds"])),
                    "missing_required_seeds": " ".join(
                        map(str, group["missing_required_seeds"])
                    ),
                    "partial": group["partial"],
                    "mean_training_time_seconds": (
                        timing["mean_training_time_seconds"]
                        if timing is not None
                        else None
                    ),
                    "training_time_sample_standard_deviation_ddof_1": (
                        timing["sample_standard_deviation_ddof_1"]
                        if timing is not None
                        else None
                    ),
                    "training_source_commit": group["training_source_commit"],
                    "training_source_commits": " ".join(
                        group["training_source_commits"]
                    ),
                    "historical_reuse_seeds": " ".join(
                        str(item["seed"])
                        for item in group["training_source_reuse_evidence"]
                        if item.get("policy")
                        == "exact_historical_commit_and_checkpoint_sha256"
                    ),
                    "historical_reuse_checkpoint_sha256s": " ".join(
                        str(item["historical_checkpoint_sha256"])
                        for item in group["training_source_reuse_evidence"]
                        if item.get("policy")
                        == "exact_historical_commit_and_checkpoint_sha256"
                    ),
                    "training_source_dirty": group["training_source_dirty"],
                    "training_artifact_validation_status": group[
                        "training_artifact_validation_status"
                    ],
                    "training_artifact_evaluation_classification": group[
                        "training_artifact_evaluation_classification"
                    ],
                    "original_training_evaluation_protocol_version": group[
                        "original_training_evaluation_protocol_version"
                    ],
                    "training_log_contract_sha256": group[
                        "training_log_contract_sha256"
                    ],
                    "checkpoint_load_mode": group["checkpoint_load_metadata"][
                        "checkpoint_load_mode"
                    ],
                    "legacy_torch_version_compatibility_used": group[
                        "checkpoint_load_metadata"
                    ]["legacy_torch_version_compatibility_used"],
                    "legacy_neuraloperator_metadata_compatibility_used": group[
                        "checkpoint_load_metadata"
                    ].get(
                        "legacy_neuraloperator_metadata_compatibility_used",
                        False,
                    ),
                    "evaluation_source_commit": group["evaluation_source_commit"],
                    "evaluation_source_dirty": group["evaluation_source_dirty"],
                    "evaluation_batch_sizes": " ".join(
                        map(str, group["evaluation_batch_sizes"])
                    ),
                    "evaluation_protocol_version": group[
                        "evaluation_protocol_version"
                    ],
                    "evaluation_definition_sha256": group[
                        "evaluation_definition_sha256"
                    ],
                    "model_configuration_sha256": group[
                        "model_configuration_sha256"
                    ],
                    "training_configuration_sha256": group[
                        "training_configuration_sha256"
                    ],
                    "data_configuration_sha256": group[
                        "data_configuration_sha256"
                    ],
                    "training_protocol_identifier": group[
                        "training_protocol_identifier"
                    ],
                    "training_protocol_sha256": group[
                        "training_protocol_sha256"
                    ],
                    "dataset_identity_sha256": group["dataset_identity_sha256"],
                    "paper_group_contract_sha256": group[
                        "paper_group_contract_sha256"
                    ],
                    "diagnostic_only": group["diagnostic_only"],
                    "paper_eligible": group["paper_eligible"],
                }
            )
    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else ["dataset"]
    with paths[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    markdown = [
        "| Dataset | Model | Parameters | Metric | Mean | Sample SD (ddof=1) | n | Seeds | Partial |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    latex = [
        r"\begin{tabular}{llrlrrrll}",
        r"Dataset & Model & Parameters & Metric & Mean & Sample SD & n & Seeds & Partial \\",
        r"\hline",
    ]
    for row in aggregate_rows:
        display = {
            **row,
            "mean": _format_number(row["mean"]),
            "sd": _format_number(row["sample_standard_deviation_ddof_1"]),
        }
        markdown.append(
            "| {dataset} | {model} | {parameter_count} | {metric} | {mean} | {sd} | {seed_count} | "
            "{seeds} | {partial} |".format(**display)
        )
        latex.append(
            "{dataset} & {model} & {parameter_count} & {metric} & {mean} & {sd} & "
            "{seed_count} & {seeds} & {partial} \\\\".format(
                **{
                    key: _latex_escape(value)
                    for key, value in display.items()
                }
            )
        )
    latex.append(r"\end{tabular}")
    paths[2].write_text("\n".join(markdown) + "\n", encoding="utf-8")
    paths[3].write_text("\n".join(latex) + "\n", encoding="utf-8")
    _write_json(paths[4], report)
    return paths


class _SafeCLIArgumentError(ValueError):
    """Argument failure whose raw value must never reach public stderr."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _SafeCLIArgumentError("invalid command-line arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reevaluate = subparsers.add_parser(
        "reevaluate", description="Re-evaluate one or more fixed-final checkpoints."
    )
    reevaluate.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    reevaluate.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    reevaluate.add_argument("--data-root", type=Path, default=Path("data"))
    reevaluate.add_argument("--batch-size", type=int, default=32)
    reevaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    reevaluate.add_argument("--output-root", type=Path)
    reevaluate.add_argument("--allow-dirty-source", action="store_true")
    reevaluate.add_argument(
        "--diagnostic-allow-dirty-training-artifact",
        action="store_true",
        help=(
            "permit a provenance-complete dirty training bundle only for a "
            "diagnostic record that is marked paper_eligible=false"
        ),
    )
    reevaluate.add_argument(
        "--allow-legacy-torch-version",
        action="store_true",
        help=(
            "use the narrow weights-only TorchVersion compatibility allowlist "
            "and record that compatibility path in the evaluation artifact"
        ),
    )
    reevaluate.add_argument(
        "--allow-legacy-neuraloperator-metadata",
        action="store_true",
        help=(
            "use the pinned SpectralConv/GELU weights-only compatibility "
            "allowlist for a trusted legacy neuraloperator checkpoint and "
            "record that path in the evaluation artifact"
        ),
    )
    reevaluate.add_argument(
        "--reuse-training-source-commit",
        help=(
            "exact historical training commit to trust for this checkpoint; "
            "requires --reuse-checkpoint-sha256"
        ),
    )
    reevaluate.add_argument(
        "--reuse-checkpoint-sha256",
        help="exact trusted checkpoint digest paired with the historical commit",
    )
    reevaluate.add_argument("--overwrite", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", description="Build strict seed-level and aggregate paper tables."
    )
    aggregate.add_argument("--inputs", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    aggregate.add_argument("--allow-partial", action="store_true")
    aggregate.add_argument("--overwrite", action="store_true")
    return parser


_SAFE_CLI_VALUE_RE = re.compile(r"[A-Za-z0-9_./:=<>+-]{1,240}\Z")


def _safe_cli_value(value: str | int) -> str:
    rendered = str(value)
    return rendered if _SAFE_CLI_VALUE_RE.fullmatch(rendered) else "<redacted>"


def _print_cli_error(
    reason_code: str,
    safe_fields: Mapping[str, str | int] | None = None,
) -> None:
    reason = _safe_cli_value(reason_code)
    fields = " ".join(
        f"{_safe_cli_value(key)}={_safe_cli_value(value)}"
        for key, value in sorted((safe_fields or {}).items())
    )
    suffix = f" {fields}" if fields else ""
    print(f"ERROR reason={reason}{suffix}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "reevaluate":
            if (args.reuse_training_source_commit is None) != (
                args.reuse_checkpoint_sha256 is None
            ):
                raise ResultValidationError(
                    "Historical reuse requires both an exact commit and "
                    "checkpoint digest.",
                    reason_code="invalid_historical_reuse",
                )
            if args.reuse_training_source_commit is not None:
                _require_lower_hex(
                    args.reuse_training_source_commit,
                    "reuse training source commit",
                    40,
                )
                _require_lower_hex(
                    args.reuse_checkpoint_sha256,
                    "reuse checkpoint sha256",
                    64,
                )
            for checkpoint in args.checkpoints:
                # Dataset/checkpoint dependencies may print absolute paths or
                # include them in warnings. The CLI emits only its own public
                # outcome envelope after the operation succeeds or fails.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    result = reevaluate_checkpoint(
                        dataset=args.dataset,
                        checkpoint_path=checkpoint,
                        data_root=args.data_root,
                        batch_size=args.batch_size,
                        device_name=args.device,
                        output_root=args.output_root,
                        allow_dirty_source=args.allow_dirty_source,
                        diagnostic_allow_dirty_training_artifact=(
                            args.diagnostic_allow_dirty_training_artifact
                        ),
                        allow_legacy_torch_version=(
                            args.allow_legacy_torch_version
                        ),
                        allow_legacy_neuraloperator_metadata=(
                            args.allow_legacy_neuraloperator_metadata
                        ),
                        allowed_historical_training_source_commit=(
                            args.reuse_training_source_commit
                        ),
                        allowed_historical_checkpoint_sha256=(
                            args.reuse_checkpoint_sha256
                        ),
                        overwrite=args.overwrite,
                    )
                    written = _load_json(result)
                print(
                    "WROTE "
                    + _public_path(
                        result,
                        dataset=str(written.get("dataset", "")),
                        model=str(written.get("model", "")),
                        seed=written.get("seed"),
                        artifact="evaluation-record",
                    )
                )
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                records = load_evaluation_records(args.inputs)
                report = aggregate_records(
                    records,
                    required_seeds=args.seeds,
                    allow_partial=args.allow_partial,
                )
                paths = _write_aggregate_outputs(
                    args.output_dir, report, overwrite=args.overwrite
                )
            if paths:
                print(
                    "WROTE "
                    + _public_path(args.output_dir, artifact="aggregate-output")
                )
    except _SafeCLIArgumentError:
        _print_cli_error("invalid_arguments")
        return 2
    except ResultOutputExistsError as exc:
        _print_cli_error(exc.reason_code, exc.safe_fields)
        return 2
    except ResultValidationError as exc:
        _print_cli_error(exc.reason_code, exc.safe_fields)
        return 2
    except ImportError:
        _print_cli_error("runtime_dependency_unavailable")
        return 2
    except OSError:
        _print_cli_error("input_output_access_failed")
        return 2
    except (RuntimeError, ValueError):
        _print_cli_error("runtime_validation_failed")
        return 2
    except Exception:
        # CLI output must remain anonymous even for an unexpected dependency or
        # checkpoint exception; callers still receive a non-zero status.
        _print_cli_error("unexpected_failure")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
