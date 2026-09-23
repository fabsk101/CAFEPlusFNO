"""Read-only import of validated native Full training bundles.

Stored metrics are imported, never described as a new evaluation. Historical
runtime evidence is never filled using the machine performing this validation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .artifacts import ArtifactValidationError, canonical_json, canonical_sha256, finite_number, _read_json, validate_csv
from .contracts import DATASETS, MODEL_VARIANTS, seed_value
from .protocols import TestContract, pinned_contract
from .repository import load_source_manifest, resolve_repository_root, sha256_file, verify_base_sources
from .runtime import contained, workspace_root, validate_runtime_metadata, require_runtime_comparable
from .public import PublicFieldError


def _field_paths(expected, prefix):
    """Enumerate only schema keys from trusted code, never supplied key names."""
    paths = {prefix}
    if isinstance(expected, dict):
        for key, value in expected.items():
            paths.update(_field_paths(value, prefix + "." + str(key)))
    return paths


def _compare_fields(actual, expected, prefix):
    missing, mismatched = [], []
    if actual is None:
        return ([], []) if expected is None else ([prefix], [])
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key, value in expected.items():
            if key not in actual:
                missing.append(prefix + "." + str(key))
            else:
                absent, different = _compare_fields(actual[key], value, prefix + "." + str(key))
                missing.extend(absent)
                mismatched.extend(different)
        if set(actual) - set(expected):
            # A supplied extra key can itself be personal information.
            mismatched.append(prefix)
    elif canonical_json(actual) != canonical_json(expected):
        mismatched.append(prefix)
    return missing, mismatched


def _require_fields(actual, expected, prefix, code):
    missing, mismatched = _compare_fields(actual, expected, prefix)
    if missing or mismatched:
        raise PublicFieldError(code, missing_fields=missing, mismatched_fields=mismatched,
                               field_catalog=_field_paths(expected, prefix))


_RUNTIME_OBSERVATIONS = {
    "python": ("version", "implementation"), "os": ("system", "release", "architecture"),
    "packages": ("torch", "numpy", "h5py", "tensorly", "tensorly_torch", "opt_einsum", "packaging"),
    "operations": ("amp_cpu_enabled", "amp_cuda_enabled", "tf32_matmul", "tf32_cudnn",
                   "float32_matmul_precision", "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic"),
    "threads": ("intra_op", "inter_op", "environment"),
    "cuda": ("build", "cudnn", "gpu_product_names", "nvidia_driver"),
}
_RUNTIME_FIELDS = {
    "runtime", "runtime.schema_version", "runtime.profile", "runtime.device", "runtime.unavailable",
    "runtime.runtime_policy", "runtime.seed_policy", "runtime.comparison_policy",
    *("runtime." + section + "." + name for section, names in _RUNTIME_OBSERVATIONS.items() for name in names),
}


def _runtime_diagnostics(runtime):
    """Describe missing observations without inventing historical values.

    This is advisory field discovery. The existing runtime validators remain
    authoritative for acceptance, including the formal/diagnostic distinction.
    """
    mandatory = {"runtime." + section + "." + name for section, names in _RUNTIME_OBSERVATIONS.items()
                 if section in {"python", "os", "operations"} for name in names}
    mandatory.update({"runtime.packages.torch", "runtime.packages.numpy", "runtime.threads.intra_op", "runtime.threads.inter_op"})
    if isinstance(runtime, dict) and runtime.get("device") == "cuda":
        mandatory.update({"runtime.cuda.build", "runtime.cuda.cudnn", "runtime.cuda.gpu_product_names"})
    missing, mismatched = [], []
    for field in sorted(_RUNTIME_FIELDS - {"runtime"}):
        value = runtime
        absent = False
        for part in field.split(".")[1:]:
            if not isinstance(value, dict) or part not in value:
                absent = True
                break
            value = value[part]
        if absent or value is None and field in mandatory:
            missing.append(field)
        elif value is not None:
            if field.startswith("runtime.operations."):
                if field.endswith(".float32_matmul_precision"):
                    good = isinstance(value, str) and value in {"highest", "high", "medium"}
                else:
                    good = type(value) is bool
                if not good:
                    mismatched.append(field)
            elif field in {"runtime.threads.intra_op", "runtime.threads.inter_op", "runtime.cuda.cudnn"}:
                if type(value) is not int or value <= 0:
                    mismatched.append(field)
            elif field.startswith(("runtime.python.", "runtime.os.", "runtime.packages.")) or field == "runtime.cuda.build":
                if not isinstance(value, str) or not value:
                    mismatched.append(field)
    if runtime is None:
        missing.insert(0, "runtime")
    return {"missing_fields": missing, "mismatched_fields": mismatched}


def _native_failure_fields(reasons, spec):
    """Map only native code-owned audit reasons, never arbitrary error text."""
    known = {
        "artifact.missing:summary": "summary", "artifact.missing:checkpoint": "checkpoint",
        "artifact.missing:training_log": "training_log", "summary.invalid_json": "summary",
        "checkpoint.unreadable": "checkpoint", "checkpoint.not_mapping": "checkpoint",
        "csv.unreadable": "training_log", "checkpoint.strict_model_load_failed": "checkpoint.model_state_dict",
        "summary.run_identity_mismatch": "summary.dataset_identity",
        "checkpoint.run_identity_mismatch": "checkpoint.dataset_identity",
        "summary.model_configuration_mismatch": "summary.model_hyperparameters",
        "checkpoint.model_configuration_mismatch": "checkpoint.model_configuration",
        "summary.nonfinite_value": "summary", "checkpoint.nonfinite_tensor_or_value": "checkpoint",
    }
    fields = ("model_class", "factorization", "rank", "parameter_count", "training_source_commit",
              "source_repository_commit", "training_source_dirty", "source_repository_dirty",
              "sirenfno_commit", "epoch", "final_epoch", "checkpoint_sha256", "environment",
              "model_state", "optimizer_state", "optimizer_param_groups", "scheduler_state",
              "completion_schema_version", "training_protocol_version", "training_protocol_identifier",
              "training_configuration", "data_configuration", "checkpoint_selection_policy")
    for container in ("summary", "checkpoint", "environment"):
        for field in fields:
            for suffix in ("_mismatch", "_missing", "_empty", "_invalid_entry", "_disagreement"):
                known[container + "." + field + suffix] = container + "." + field
            known[container + ".invalid_" + field] = container + "." + field
        for field in spec["metric_columns"]:
            known[container + ".invalid_metric:" + field] = container + "." + field
        for field in spec["training_configuration"]:
            known[container + ".top_level_training_field_mismatch:" + field] = container + "." + field
    selected = [known[reason] for reason in reasons if isinstance(reason, str) and reason in known]
    if not selected:
        selected = ["summary", "checkpoint", "training_log"]
    return selected, set(known.values()) | {"summary", "checkpoint", "training_log"}


def reference_spec(dataset, model_variant, *, test_contract=None):
    """Copy the parent contract and inject the real Full constructor explicitly.

    TestContract overrides apply only to labelled synthetic fixture validation.
    No imported module global or native factory is replaced.
    """
    from scripts import run_paper_sweep as sweep
    from .configs import model_configuration
    from .models import build_model_from_kwargs, preserved_rng_state
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS:
        raise ArtifactValidationError("reference_unknown_identity")
    protocol = pinned_contract(dataset, test_contract=test_contract)
    spec = copy.deepcopy(sweep.DATASETS[DATASETS[dataset].contract_key])
    kwargs = model_configuration(dataset, model_variant) if test_contract is None else test_contract.base_configuration
    canonical_model = MODEL_VARIANTS[model_variant]

    def build(name, device):
        if name != canonical_model:
            raise ArtifactValidationError("reference_model_mismatch")
        with preserved_rng_state():
            built = build_model_from_kwargs(spatial_dim=DATASETS[dataset].spatial_dim,
                configuration=kwargs, condition="full", device=device)
        return SimpleNamespace(model=built.model, configuration=built.model.get_config(),
                               factorization=built.factorization, rank=built.rank)

    spec.update(model_builder=build, parameter_count_function=lambda model: sum(p.numel() for p in model.parameters()),
                model_configuration_factory=lambda name: build(name, "cpu").configuration)
    if test_contract is not None:
        if not isinstance(test_contract, TestContract):
            raise ArtifactValidationError("reference_test_contract_invalid")
        spec.update(epochs=test_contract.epochs, training_configuration=copy.deepcopy(protocol["training_configuration"]),
                    data_configuration=copy.deepcopy(protocol["data_configuration"]),
                    training_protocol_version="TEST_NATIVE_REFERENCE_" + test_contract.label,
                    experiment="TEST_NATIVE_FULL_REFERENCE")
        spec["evaluation_metadata"]["evaluation_horizon"] = test_contract.horizon
    return spec, protocol


def reference_test_source_id(test_contract):
    """A clearly labelled fixture identity, never an asserted experiment commit."""
    return hashlib.sha1(("TEST_ONLY_SYNTHETIC_REFERENCE:" + test_contract.label).encode("utf-8")).hexdigest()


def validate_full_reference(summary_path, *, dataset, model_variant, expected_seed=None,
                            test_contract=None, require_comparable=True):
    """Return one normalized reference row, or reject with a safe reason code.

    ``require_comparable=False`` is a read-only inspection mode for valid
    artifacts with incomplete historical runtime evidence. Such records remain
    ineligible for a paper aggregate. An explicit TestContract always marks
    the result diagnostic; a formal invocation rejects its fixture markers.
    """
    from scripts import run_paper_sweep as sweep
    from .checkpoints import validate_checkpoint_payload
    from .models import preserved_rng_state
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS:
        raise ArtifactValidationError("reference_unknown_identity")
    path = contained(summary_path, workspace_root())
    if path.name != "summary.json":
        raise ArtifactValidationError("reference_summary_name_invalid")
    files = [contained(path.parent / name, workspace_root()) for name in
             ("summary.json", "final_checkpoint.pt", "training_log.csv")]
    labels = ("summary", "checkpoint", "training_log")
    missing = [label for label, item in zip(labels, files) if not item.is_file()]
    shared = [label for label, item in zip(labels, files) if item.is_file() and item.stat().st_nlink != 1]
    if missing or shared:
        raise PublicFieldError("REFERENCE_BUNDLE_MISSING_OR_SHARED", missing_fields=missing,
                               mismatched_fields=shared, field_catalog=labels)
    before = {item.name: sha256_file(item) for item in files}
    summary = _read_json(path)
    seed = seed_value(summary.get("seed"))
    if expected_seed is not None and seed != seed_value(expected_seed):
        raise ArtifactValidationError("reference_seed_mismatch")
    if path.parent.name != f"seed_{seed}" or path.parent.parent.name != MODEL_VARIANTS[model_variant]:
        raise ArtifactValidationError("reference_path_identity_mismatch")
    diagnostic = test_contract is not None
    if diagnostic:
        if summary.get("test_contract") != test_contract.label or summary.get("result_kind") != "diagnostic":
            raise ArtifactValidationError("reference_test_contract_marker_missing")
    elif summary.get("test_contract") or summary.get("result_kind") == "diagnostic":
        raise ArtifactValidationError("reference_diagnostic_not_formal")
    manifest = load_source_manifest()
    verified_base = verify_base_sources(resolve_repository_root())
    training_commit = reference_test_source_id(test_contract) if diagnostic else manifest["base_source_commit"]
    for alias in ("training_source_commit", "source_repository_commit", "evaluation_source_commit"):
        _require_fields(summary.get(alias), training_commit, "summary." + alias, "REFERENCE_SOURCE_COMMIT_MISMATCH")
    spec, protocol = reference_spec(dataset, model_variant, test_contract=test_contract)
    expected_built = spec["model_builder"](MODEL_VARIANTS[model_variant], "cpu")
    for name, expected, code in (
        ("model_hyperparameters", expected_built.configuration, "REFERENCE_CONFIGURATION_MISMATCH"),
        ("training_configuration", spec["training_configuration"], "REFERENCE_TRAINING_CONFIGURATION_MISMATCH"),
        ("data_configuration", spec["data_configuration"], "REFERENCE_DATA_CONFIGURATION_MISMATCH"),
    ):
        _require_fields(summary.get(name), expected, "summary." + name, code)
    if not isinstance(summary.get("dataset_identity"), dict) or not summary["dataset_identity"]:
        raise PublicFieldError("REFERENCE_DATASET_IDENTITY_MISSING", missing_fields=["summary.dataset_identity"],
                               field_catalog=["summary.dataset_identity"])
    try:
        # Verify against the unchanged base source. The add-on's new commit is
        # deliberately not substituted for historical training/evaluation IDs.
        with preserved_rng_state():
            validated = sweep.validate_training_artifacts(path.parent, spec=spec,
                current_source_commit=training_commit)
            columns, rows, reasons = sweep._read_training_log(files[2], epochs=spec["epochs"])
            status, evaluation_reasons = sweep._validate_native_evaluation(
                summary=validated.summary, checkpoint=validated.checkpoint, spec=spec,
                fieldnames=columns, rows=rows, current_evaluation_source_commit=training_commit)
            if reasons or evaluation_reasons or status is not sweep.CompletionStatus.COMPLETE:
                fields, catalog = _native_failure_fields((*reasons, *evaluation_reasons), spec)
                raise PublicFieldError("REFERENCE_EVALUATION_INVALID", mismatched_fields=fields, field_catalog=catalog)
            built = spec["model_builder"](MODEL_VARIANTS[model_variant], "cpu")
            validate_checkpoint_payload(validated.checkpoint, built.model,
                training_configuration=spec["training_configuration"],
                data_configuration=spec["data_configuration"], epochs=spec["epochs"])
    except (ArtifactValidationError, PublicFieldError):
        raise
    except sweep.TrainingArtifactValidationError as exc:
        config_reasons = {
            "checkpoint.model_configuration_mismatch": ("model_configuration", expected_built.configuration),
            "checkpoint.training_configuration_mismatch": ("training_configuration", spec["training_configuration"]),
            "checkpoint.data_configuration_mismatch": ("data_configuration", spec["data_configuration"]),
        }
        if any(reason in exc.audit.reasons for reason in config_reasons):
            from experiments.common.checkpoints import load_weights_only_checkpoint
            checkpoint = load_weights_only_checkpoint(files[1], map_location="cpu").checkpoint
            for reason, (key, expected) in config_reasons.items():
                if reason in exc.audit.reasons:
                    _require_fields(checkpoint.get(key), expected, "checkpoint." + key, "REFERENCE_CONFIGURATION_MISMATCH")
        fields, catalog = _native_failure_fields(exc.audit.reasons, spec)
        raise PublicFieldError("REFERENCE_NATIVE_BUNDLE_VALIDATION_FAILED", mismatched_fields=fields,
                               field_catalog=catalog) from None
    except Exception:
        raise PublicFieldError("REFERENCE_NATIVE_BUNDLE_VALIDATION_FAILED",
            mismatched_fields=["summary", "checkpoint", "training_log"],
            field_catalog=["summary", "checkpoint", "training_log"]) from None
    for name in ("runtime", "test_contract", "result_kind"):
        left, right = summary.get(name), validated.checkpoint.get(name)
        if canonical_json(left) != canonical_json(right):
            fields = ["summary." + name, "checkpoint." + name]
            raise PublicFieldError("REFERENCE_SUMMARY_CHECKPOINT_METADATA_MISMATCH", mismatched_fields=fields,
                                   field_catalog=fields)
    final_metrics = {name: finite_number(rows[-1][name]) if type(rows[-1][name]) in (int, float)
                     else float(rows[-1][name]) for name in protocol["csv_columns"]}
    final_metrics["epoch"] = spec["epochs"]
    # Native metrics are top-level; add-on metrics use canonical CSV names.
    for metric_name, column in spec["metric_columns"].items():
        try:
            final_metrics[column] = finite_number(summary.get(metric_name))
        except ArtifactValidationError:
            fields = ["summary." + metric_name]
            raise PublicFieldError("REFERENCE_METRIC_INVALID", mismatched_fields=fields,
                                   field_catalog=fields) from None
    csv_identity = validate_csv(files[2], protocol=protocol, final_metrics=final_metrics,
                                identity={"seed": seed, "model": MODEL_VARIANTS[model_variant]})
    recorded_log = summary.get("training_log_identity")
    if recorded_log is not None and canonical_json(recorded_log) != canonical_json(csv_identity):
        raise ArtifactValidationError("reference_training_log_hash_mismatch")
    runtime = summary.get("runtime")
    runtime_diagnostics = _runtime_diagnostics(runtime)
    runtime_verified = False
    if runtime is not None:
        try:
            validate_runtime_metadata(runtime, formal=not diagnostic)
            if not diagnostic:
                require_runtime_comparable(runtime)
        except Exception:
            if (not isinstance(runtime, dict) or runtime_diagnostics["mismatched_fields"]
                    or not runtime_diagnostics["missing_fields"]):
                mismatched = runtime_diagnostics["mismatched_fields"] or ["runtime"]
                if isinstance(runtime, dict):
                    from .runtime_metadata import RUNTIME_SCHEMA_VERSION
                    if runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION:
                        mismatched = ["runtime.schema_version"]
                    elif not diagnostic and runtime.get("profile") != "formal":
                        mismatched = ["runtime.profile"]
                    elif not isinstance(runtime.get("device"), str) or runtime["device"] not in {"cpu", "cuda"}:
                        mismatched = ["runtime.device"]
                raise PublicFieldError("REFERENCE_RUNTIME_EVIDENCE_INVALID", mismatched_fields=mismatched,
                                       field_catalog=_RUNTIME_FIELDS) from None
            # Incomplete evidence may be inspected, but never accepted for a
            # formal comparison. No missing setting is supplied from this host.
        else:
            runtime_verified = not bool(runtime_diagnostics["missing_fields"])
    limitations = []
    if not runtime_verified:
        limitations.append("historical_runtime_settings_not_recorded; current environment was not substituted")
    if diagnostic:
        limitations.append("TEST_ONLY: synthetic metadata and tiny CAFE update; no official data or formal training")
    comparison_eligible = runtime_verified and not diagnostic
    if require_comparable and not comparison_eligible:
        if not runtime_verified:
            raise PublicFieldError("REFERENCE_RUNTIME_EVIDENCE_INCOMPLETE", **runtime_diagnostics,
                                   field_catalog=_RUNTIME_FIELDS)
        raise ArtifactValidationError("reference_diagnostic_not_formal")
    metric_definition = protocol["evaluation_definition"]
    metrics = {public: finite_number(final_metrics[column]) for column, public in protocol["paper_metrics"].items()}
    source = {"base_source_commit": manifest["base_source_commit"], "base_source_sha256": verified_base,
              "sirenfno_commit": manifest["sirenfno_commit"],
              "training_source_commit": validated.training_source_commit,
              "source_repository_commit": summary["source_repository_commit"],
              "evaluation_source_commit": summary["evaluation_source_commit"],
              "source_mode": "diagnostic" if diagnostic else "formal",
              "reference_source_verification": "pinned_base_files_and_native_artifact_contract"}
    comparison = {"dataset": dataset, "model_variant": model_variant, "condition": "full",
                  "model_configuration": built.configuration,
                  "training_configuration": protocol["training_configuration"],
                  "data_configuration": protocol["data_configuration"],
                  "evaluation_definition": metric_definition,
                  "dataset_identity": dict(validated.dataset_identity)}
    after = {item.name: sha256_file(item) for item in files}
    if before != after:
        raise ArtifactValidationError("reference_bundle_changed_during_validation")
    return {"seed": seed, "metrics": metrics, "checkpoint_sha256": validated.checkpoint_sha256,
            "parameter_count": summary["parameter_count"], "comparison_identity": canonical_sha256(comparison),
            "context_identity": canonical_sha256({"source": source, "runtime": runtime}),
            "metric_definition": metric_definition, "evaluation_definition": metric_definition,
            "source": source, "runtime": runtime, "dataset_identity": dict(validated.dataset_identity),
            "training_configuration": protocol["training_configuration"],
            "data_configuration": protocol["data_configuration"], "model_configuration": built.configuration,
            "origin": "main_full_reference", "comparison_eligible": comparison_eligible,
            "runtime_verification": "VERIFIED_RECORDED" if runtime_verified else "UNVERIFIED",
            "runtime_diagnostics": runtime_diagnostics,
            "result_kind": "diagnostic" if diagnostic else "formal",
            "reference_provenance": {"kind": "validated_existing_result_reference_import",
                "training_source_commit": validated.training_source_commit,
                "legacy_source_repository_commit": summary["source_repository_commit"],
                "evaluation_source_commit": summary["evaluation_source_commit"],
                "reevaluation_performed": False, "source_files_verified": len(verified_base),
                "artifact_id": "reference-" + before["summary.json"][:16],
                "bundle_sha256": before, "training_log_identity": csv_identity,
                "fixture": diagnostic, "limitations": limitations,
                "runtime_diagnostics": runtime_diagnostics}}
