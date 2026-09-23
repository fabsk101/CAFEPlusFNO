"""One strict validator/writer shared by training, reuse and aggregation.

Hashes attest byte integrity, not that 500 optimizer epochs actually occurred.
Formal contracts require the complete recorded epoch history and actual states;
synthetic evidence is deliberately labelled diagnostic and is never promoted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS, EXPERIMENT_ID, seed_value
from .protocols import TestContract, pinned_contract
from .runtime import contained, output_path, workspace_root
from .repository import sha256_file

SCHEMA_VERSION = 3


class ArtifactValidationError(ValueError):
    """Only public-safe, fixed reason codes are exposed."""

    def __init__(self, code):
        normalized = code.upper() if isinstance(code, str) else "ARTIFACT_VALIDATION_FAILED"
        self.reason_code = normalized if re.fullmatch(r"[A-Z][A-Z0-9_]{1,99}", normalized) else "ARTIFACT_VALIDATION_FAILED"
        super().__init__(self.reason_code.lower())


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        raise ArtifactValidationError("artifact_noncanonical_metadata") from None


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _same(actual, expected, code):
    if canonical_json(actual) != canonical_json(expected):
        raise ArtifactValidationError(code)


def finite_number(value, *, nonnegative=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ArtifactValidationError("artifact_invalid_numeric_value")
    return float(value)


def _read_json(path):
    def reject_constant(_):
        raise ArtifactValidationError("artifact_nonfinite_json")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactValidationError("artifact_duplicate_json_key")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                           object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ArtifactValidationError("artifact_invalid_summary_json") from None
    if not isinstance(value, dict):
        raise ArtifactValidationError("artifact_summary_not_mapping")
    return value


def _local_file(run_dir, name, expected):
    if name != expected:
        raise ArtifactValidationError("artifact_invalid_local_reference")
    try:
        path = contained(run_dir / expected, workspace_root())
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ArtifactValidationError("artifact_missing_or_shared_file")
    except OSError:
        raise ArtifactValidationError("artifact_file_unreadable") from None
    return path


def build_artifact_contract(dataset, model_variant, condition, *, dataset_identity,
                            runtime, source, test_contract: TestContract | None = None):
    """Construct pinned metadata without consuming the caller's RNG state."""
    from .configs import model_configuration
    from .models import build_model_from_kwargs, describe_model, preserved_rng_state
    protocol = pinned_contract(dataset, test_contract=test_contract)
    _same(dataset_identity, protocol["dataset_identity"], "artifact_dataset_identity_mismatch")
    base = model_configuration(dataset, model_variant) if test_contract is None else test_contract.base_configuration
    with preserved_rng_state():
        built = build_model_from_kwargs(spatial_dim=DATASETS[dataset].spatial_dim,
                                        configuration=base, condition=condition, device="cpu")
        dimensions = describe_model(built.model)
    comparison = {
        "experiment": EXPERIMENT_ID, "dataset": dataset, "dataset_id": protocol["dataset_id"],
        "dataset_contract_key": DATASETS[dataset].contract_key,
        "canonical_model": MODEL_VARIANTS[model_variant], "model_variant": model_variant,
        "condition": condition, "base_configuration": base,
        "resolved_ablation_configuration": built.configuration,
        "training_protocol_version": protocol["training_protocol_version"],
        "training_configuration": protocol["training_configuration"],
        "data_configuration": protocol["data_configuration"],
        "evaluation_definition": protocol["evaluation_definition"],
        "dataset_identity": dataset_identity, "epochs": protocol["training_configuration"]["epochs"],
        "test_contract": protocol["test_contract"],
    }
    return {"schema_version": SCHEMA_VERSION, **comparison,
            "comparison_contract": comparison, "comparison_fingerprint": canonical_sha256(comparison),
            "experiment_id": EXPERIMENT_ID, "ablation_variant": condition,
            "condition_contract": ABLATION_CONDITIONS[condition].__dict__.copy(),
            "factorization": built.factorization, "rank": built.rank,
            "model_dimensions": dimensions, "runtime": runtime, "source": source,
            "model_configuration": built.configuration,
            "final_epoch": comparison["epochs"],
            "checkpoint_selection_policy": protocol["training_configuration"]["checkpoint_selection_policy"]}


def validate_csv(path, *, protocol, final_metrics, identity=None):
    """Validate all logged epochs, finite columns and final metric agreement."""
    from scripts.run_paper_sweep import _read_training_log, _training_log_identity
    epochs = protocol["training_configuration"]["epochs"]
    try:
        columns, rows, reasons = _read_training_log(path, epochs=epochs)
    except (OSError, UnicodeError, csv.Error):
        raise ArtifactValidationError("artifact_csv_unreadable") from None
    if reasons or not rows:
        raise ArtifactValidationError("artifact_csv_epoch_history_invalid")
    if not set(protocol["csv_columns"]).issubset(columns):
        raise ArtifactValidationError("artifact_csv_missing_columns")
    if final_metrics.get("epoch") != epochs or type(final_metrics.get("epoch")) is not int:
        raise ArtifactValidationError("artifact_final_metric_epoch_mismatch")
    for row in rows:
        for column in protocol["csv_columns"]:
            try:
                number = float(row[column])
            except (KeyError, ValueError, TypeError):
                raise ArtifactValidationError("artifact_csv_invalid_numeric") from None
            finite_number(number)
        for column in ("train_evaluation_sample_count", "test_evaluation_sample_count", "evaluation_horizon"):
            if column in protocol["csv_columns"]:
                expected = (protocol["data_configuration"]["n_train"] if column.startswith("train_") else
                            protocol["data_configuration"]["n_test"] if column.startswith("test_") else
                            protocol["evaluation_definition"]["evaluation_horizon"])
                if float(row[column]) != expected:
                    raise ArtifactValidationError("artifact_csv_evaluation_count_mismatch")
        if identity:
            for key, expected in identity.items():
                if key in columns and row[key] != str(expected):
                    raise ArtifactValidationError("artifact_csv_run_identity_mismatch")
    for column in protocol["csv_columns"]:
        number = finite_number(final_metrics.get(column))
        if not math.isclose(number, float(rows[-1][column]), rel_tol=1e-12, abs_tol=1e-15):
            raise ArtifactValidationError("artifact_csv_summary_metric_mismatch")
    return _training_log_identity(path, fieldnames=columns, rows=rows,
                                  metric_columns=protocol["metric_columns"])


def _validate_source(source, *, diagnostic):
    from .repository import load_source_manifest
    if not isinstance(source, dict) or not source:
        raise ArtifactValidationError("artifact_source_missing")
    manifest = load_source_manifest()
    for key in ("base_source_commit", "sirenfno_commit"):
        _same(source.get(key), manifest[key], "artifact_pinned_source_mismatch")
    _same(source.get("base_source_sha256"), manifest["files"], "artifact_pinned_source_hash_mismatch")
    if diagnostic:
        if source.get("source_mode") != "diagnostic":
            raise ArtifactValidationError("artifact_diagnostic_source_mismatch")
        return
    if source.get("source_mode") != "formal" or source.get("source_repository_dirty") is not False or source.get("ablation_package_integrated") is not True:
        raise ArtifactValidationError("artifact_source_not_verified_formal")
    if source.get("source_verification_method") not in ("git_clean_checkout", "release_manifest_hashes"):
        raise ArtifactValidationError("artifact_source_verification_missing")
    for name, length in (("actual_ablation_source_commit", 40), ("ablation_source_sha256", 64)):
        value = source.get(name)
        if not isinstance(value, str) or len(value) != length or any(x not in "0123456789abcdef" for x in value):
            raise ArtifactValidationError("artifact_source_hash_invalid")
    files = source.get("ablation_source_files_sha256")
    if not isinstance(files, dict) or not files:
        raise ArtifactValidationError("artifact_source_file_manifest_missing")
    for name, digest in files.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name or
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
            not isinstance(digest, str) or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest)):
            raise ArtifactValidationError("artifact_source_file_manifest_invalid")
    if canonical_sha256(files) != source.get("ablation_file_manifest_sha256"):
        raise ArtifactValidationError("artifact_source_file_manifest_hash_mismatch")
    method = source["source_verification_method"]
    if ((method == "git_clean_checkout" and source.get("git_ancestry_verified") is not True) or
        (method == "release_manifest_hashes" and source.get("git_ancestry_verified") is not False)):
        raise ArtifactValidationError("artifact_source_method_ancestry_mismatch")
    if method == "release_manifest_hashes":
        digest = source.get("release_provenance_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ArtifactValidationError("artifact_release_provenance_hash_invalid")


@dataclass(frozen=True)
class ValidatedArtifact:
    summary: dict
    checkpoint: Mapping
    metrics: dict[str, float]
    comparison_identity: str
    context_identity: str
    checkpoint_sha256: str
    training_log_identity: dict
    metric_definition: dict


def validate_artifact(summary_path, *, dataset, model_variant, condition, seed,
                      allow_diagnostic=False, test_contract=None, expected_context=None):
    seed = seed_value(seed)
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS or condition not in ABLATION_CONDITIONS:
        raise ArtifactValidationError("artifact_unknown_run_identity")
    from .repository import resolve_repository_root, verify_base_sources
    verify_base_sources(resolve_repository_root())
    path = contained(summary_path, workspace_root())
    summary = _read_json(_local_file(path.parent, path.name, "summary.json"))
    diagnostic = summary.get("result_kind") == "diagnostic"
    if diagnostic and not allow_diagnostic:
        raise ArtifactValidationError("artifact_diagnostic_not_formal")
    if test_contract is not None and not diagnostic:
        raise ArtifactValidationError("artifact_test_contract_cannot_be_formal")
    kind = "diagnostic" if diagnostic else "formal"
    exact = {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID,
             "dataset": dataset, "model_variant": model_variant, "condition": condition,
             "ablation_variant": condition, "seed": seed, "result_kind": kind}
    for key, value in exact.items():
        _same(summary.get(key), value, "artifact_run_identity_mismatch")
    suffix = (EXPERIMENT_ID, kind, dataset, model_variant, condition, f"seed_{seed}")
    if path.parent.parts[-len(suffix):] != suffix:
        raise ArtifactValidationError("artifact_path_identity_mismatch")
    for child in path.parent.iterdir():
        child = contained(child, path.parent)
        if child.is_dir():
            raise ArtifactValidationError("artifact_nested_run_directory")
    _validate_source(summary.get("source"), diagnostic=diagnostic)
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise ArtifactValidationError("artifact_runtime_missing")
    from .runtime import validate_runtime_metadata
    validate_runtime_metadata(runtime, formal=not diagnostic)
    expected = build_artifact_contract(dataset, model_variant, condition,
        dataset_identity=summary.get("dataset_identity"), runtime=runtime,
        source=summary["source"], test_contract=test_contract)
    for key, value in expected.items():
        if key == "model_dimensions":
            for name, dimension in value.items():
                _same(summary.get(key, {}).get(name), dimension, "artifact_model_dimension_mismatch")
        else:
            _same(summary.get(key), value, "artifact_pinned_configuration_mismatch")
    checkpoint_path = _local_file(path.parent, summary.get("final_checkpoint"), "final_checkpoint.pt")
    csv_path = _local_file(path.parent, summary.get("training_log"), "training_log.csv")
    digest = sha256_file(checkpoint_path)
    if digest != summary.get("checkpoint_sha256"):
        raise ArtifactValidationError("artifact_checkpoint_hash_mismatch")
    from experiments.common.checkpoints import load_weights_only_checkpoint
    try:
        checkpoint = load_weights_only_checkpoint(checkpoint_path, map_location="cpu").checkpoint
    except Exception:
        raise ArtifactValidationError("artifact_checkpoint_weights_only_load_failed") from None
    for key, value in summary.items():
        if key not in {"checkpoint_sha256", "final_checkpoint", "training_log"}:
            _same(checkpoint.get(key), value, "artifact_checkpoint_summary_mismatch")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != expected["epochs"]:
        raise ArtifactValidationError("artifact_checkpoint_epoch_mismatch")
    from .models import build_model_from_kwargs, preserved_rng_state
    from .checkpoints import validate_checkpoint_payload
    with preserved_rng_state():
        built = build_model_from_kwargs(spatial_dim=DATASETS[dataset].spatial_dim,
            configuration=expected["base_configuration"], condition=condition, device="cpu")
        restored_optimizer, _ = validate_checkpoint_payload(checkpoint, built.model,
            training_configuration=expected["training_configuration"],
            data_configuration=expected["data_configuration"], epochs=expected["epochs"])
        from .train import sigma_optimizer_record
        _same(summary.get("sigma_optimizer"), sigma_optimizer_record(built.model, restored_optimizer),
              "artifact_sigma_optimizer_metadata_mismatch")
    protocol = pinned_contract(dataset, test_contract=test_contract)
    final = summary.get("final_metrics")
    if not isinstance(final, dict):
        raise ArtifactValidationError("artifact_final_metrics_missing")
    metrics = {public: finite_number(final.get(column)) for column, public in protocol["paper_metrics"].items()}
    log_identity = validate_csv(csv_path, protocol=protocol, final_metrics=final, identity=exact)
    _same(summary.get("training_log_identity"), log_identity, "artifact_training_log_identity_mismatch")
    _same(summary.get("training_log_sha256"), log_identity["sha256"], "artifact_training_log_hash_mismatch")
    finite_number(summary.get("total_training_time_seconds"))
    context = {key: summary[key] for key in ("dataset_identity", "training_configuration", "data_configuration", "evaluation_definition", "source", "runtime")}
    if expected_context:
        for key, value in expected_context.items():
            _same(summary.get(key), value, "artifact_current_context_mismatch")
    return ValidatedArtifact(summary, checkpoint, metrics, summary["comparison_fingerprint"],
                             canonical_sha256(context), digest, log_identity, protocol["evaluation_definition"])


def write_artifact(run_dir, *, metadata, model, optimizer, scheduler, final_metrics,
                   total_training_time_seconds=0.0, test_contract=None):
    """Write an actual weights-only checkpoint; CSV must already be complete."""
    import torch
    from experiments.common.checkpoints import prepare_weights_only_checkpoint
    run_dir = output_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _local_file(run_dir, "training_log.csv", "training_log.csv")
    protocol = pinned_contract(metadata["dataset"], test_contract=test_contract)
    log_identity = validate_csv(csv_path, protocol=protocol, final_metrics=final_metrics)
    from .train import sigma_optimizer_record
    sigma = sigma_optimizer_record(model, optimizer)
    if "sigma_optimizer" in metadata:
        _same(metadata["sigma_optimizer"], sigma, "artifact_sigma_optimizer_metadata_mismatch")
    common = {**metadata, "final_metrics": final_metrics,
              "sigma_optimizer": sigma,
              "total_training_time_seconds": finite_number(total_training_time_seconds),
              "training_log_identity": log_identity, "training_log_sha256": log_identity["sha256"]}
    checkpoint = prepare_weights_only_checkpoint({**common, "epoch": metadata["final_epoch"],
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()})
    checkpoint_path = output_path(run_dir / "final_checkpoint.pt")
    summary_path = output_path(run_dir / "summary.json")
    if checkpoint_path.exists() or summary_path.exists():
        raise ArtifactValidationError("artifact_writer_refuses_overwrite")
    torch.save(checkpoint, checkpoint_path)
    summary = {**common, "final_checkpoint": "final_checkpoint.pt", "training_log": "training_log.csv",
               "checkpoint_sha256": sha256_file(checkpoint_path)}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary_path


def restore_validated_artifact(summary_path, **expected):
    """Checkpoint re-evaluation entry: shared full audit before model restore.

    This loads the saved final model only; it does not claim evaluation or offer
    epoch resume, and never modifies the source run bundle.
    """
    verified = validate_artifact(summary_path, **expected)
    from .models import build_model_from_kwargs, preserved_rng_state
    from .checkpoints import validate_checkpoint_payload
    with preserved_rng_state():
        built = build_model_from_kwargs(spatial_dim=DATASETS[expected["dataset"]].spatial_dim,
            configuration=verified.summary["base_configuration"], condition=expected["condition"], device="cpu")
        validate_checkpoint_payload(verified.checkpoint, built.model,
            training_configuration=verified.summary["training_configuration"],
            data_configuration=verified.summary["data_configuration"], epochs=verified.summary["epochs"])
    return built, verified


def parse_args(argv=None):
    from .public import SafeArgumentParser
    parser = SafeArgumentParser(
        description="Validate a saved final bundle, optionally restore its model; no data evaluation or training.")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model-variant", choices=sorted(MODEL_VARIANTS), required=True)
    parser.add_argument("--condition", choices=sorted(ABLATION_CONDITIONS), required=True)
    parser.add_argument("--seed", type=seed_value, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--restore", action="store_true",
                         help="Restore the audited model on CPU for evaluator integration; no metrics are recalculated.")
    actions.add_argument("--inspect-reference", action="store_true",
                         help="Inspect a main-experiment Full bundle and report whether historical runtime evidence is sufficient.")
    args = parser.parse_args(argv)
    if args.inspect_reference and args.condition != "full":
        parser.error("REFERENCE_REQUIRES_FULL_CONDITION")
    return args


def main(argv=None):
    """Public inspection CLI; no runtime/metadata from a historical run is filled."""
    args = parse_args(argv)
    from .repository import activate_repository, resolve_repository_root
    from .runtime import configure_runtime, apply_torch_runtime
    from .public import emit_json
    activate_repository(resolve_repository_root(args.repository_root))
    # This is a bounded CPU artifact inspection, not the recorded training
    # runtime. Keep caches under results/ so a Git-less release remains valid.
    configure_runtime(cpu=True, profile="formal", threads=1)
    import torch
    apply_torch_runtime()
    identity = {"dataset": args.dataset, "model_variant": args.model_variant,
                "condition": args.condition, "seed": args.seed}
    if args.inspect_reference:
        from .references import validate_full_reference
        reference = validate_full_reference(args.summary, dataset=args.dataset,
            model_variant=args.model_variant, require_comparable=False)
        if reference["seed"] != args.seed:
            raise ArtifactValidationError("artifact_reference_seed_mismatch")
        emit_json({**identity, "status": "REFERENCE_VALIDATED",
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "metrics": reference["metrics"],
            "comparison_eligible": reference["comparison_eligible"],
            "runtime_verification": reference["runtime_verification"],
            "runtime_diagnostics": reference["runtime_diagnostics"],
            "reference_provenance": reference["reference_provenance"],
            "metric_values_reevaluated": False, "training_performed": False,
            "inspection_device": "cpu", "inspection_threads": 1})
        return
    if args.restore:
        built, verified = restore_validated_artifact(args.summary, **identity)
        del built
    else:
        verified = validate_artifact(args.summary, **identity)
    emit_json({**identity, "status": "RESTORED_FOR_EVALUATION" if args.restore else "VALIDATED",
        "checkpoint_sha256": verified.checkpoint_sha256,
        "final_epoch": verified.summary["final_epoch"], "metrics": verified.metrics,
        "comparison_identity": verified.comparison_identity,
        "metric_values_reevaluated": False, "training_performed": False,
        "inspection_device": "cpu", "inspection_threads": 1})


if __name__ == "__main__":
    from .public import public_main
    public_main(main)
