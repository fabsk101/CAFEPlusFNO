"""Aggregate selected seeds only after common actual-artifact validation."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .contracts import (ABLATION_CONDITIONS, DATASETS, DEFAULT_SEEDS, EXPERIMENT_ID,
                        MODEL_VARIANTS, seed_value, validate_seeds)
from .runtime import contained, output_path, workspace_root
from .public import SafeArgumentParser, PublicError, emit_json, public_main, public_payload


def _run_record(verified):
    summary = verified.summary
    return {"seed": summary["seed"], "origin": "ablation", "metrics": verified.metrics,
            "checkpoint_sha256": verified.checkpoint_sha256,
            "parameter_count": summary["model_dimensions"]["parameter_count"],
            "model_dimensions": summary["model_dimensions"],
            "comparison_identity": verified.comparison_identity,
            "context_identity": verified.context_identity,
            "metric_definition": verified.metric_definition,
            "training_log_identity": verified.training_log_identity,
            **{key: summary[key] for key in ("source", "runtime", "dataset_identity",
                 "training_configuration", "data_configuration", "evaluation_definition")},
            "comparison_eligible": summary["result_kind"] == "formal"}


def _compatible_context(record):
    """Baseline parent and addon commits may differ; verified base must match.

    Addon-to-addon source equality is additionally checked below. Actual source
    commits remain in every run record; nothing is rewritten as a current run.
    """
    source = record["source"]
    return {**{key: record[key] for key in ("runtime", "dataset_identity",
                 "training_configuration", "data_configuration", "evaluation_definition")},
            "base_source": {key: source.get(key) for key in
                            ("base_source_commit", "base_source_sha256", "sirenfno_commit")}}


def _latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value).replace("\n", " "))


def _export(report, destination, *, overwrite=False):
    """Render the entire bundle before its guarded, staged publication."""
    import io
    from .artifacts import canonical_sha256
    from .output_bundle import publish_output_bundle
    report = public_payload(report)
    report = {**report, "ddof": 1,
              "result_label": "FORMAL" if report["result_kind"] == "formal" else "TEST/DIAGNOSTIC",
              "standard_deviation_status": "undefined" if report["seed_count"] == 1 else "defined",
              "completeness_scope": "selected_seed_set_only_not_statistical_or_paper_approval"}
    report["aggregate_id"] = canonical_sha256(report)
    identity = {name: report[name] for name in ("experiment_id", "dataset", "model_variant", "condition")}
    state = {name: report[name] for name in ("status", "result_kind", "result_label", "seed_count", "ddof",
                                           "standard_deviation_status", "aggregate_id", "completeness_scope")}
    state.update({name: json.dumps(report[name]) for name in
                  ("expected_seeds", "selected_seeds", "present_seeds", "missing_seeds")})
    rendered = {"aggregate.json": json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"}
    handle = io.StringIO(newline="")
    fields = [*identity, *state, "metric", "mean", "sample_std", "metric_definition", "context_identity"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for metric, values in report["metrics"].items():
        writer.writerow({**identity, **state, "metric": metric, "mean": values["mean"],
             "sample_std": values["sample_std"],
             "metric_definition": json.dumps(report["metric_definition"], sort_keys=True, allow_nan=False),
             "context_identity": report["context_identity"]})
    rendered["aggregate.csv"] = handle.getvalue()
    handle = io.StringIO(newline="")
    fields = [*identity, *state, "seed", "origin", "metric", "value", "checkpoint_sha256",
              "parameter_count", "source", "comparison_identity", "metric_definition"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for run in report["runs"]:
        for metric, value in run["metrics"].items():
            writer.writerow({**identity, **state, **{key: run[key] for key in
                 ("seed", "origin", "checkpoint_sha256", "parameter_count", "comparison_identity")},
                 "metric": metric, "value": value,
                 "source": json.dumps(run["source"], sort_keys=True, allow_nan=False),
                 "metric_definition": json.dumps(report["metric_definition"], sort_keys=True, allow_nan=False)})
    rendered["seed_values.csv"] = handle.getvalue()
    heading = f"{report['result_label']} | {report['status']} | actual n={report['seed_count']} | ddof=1"
    selection = f"Expected/selected seeds: {report['selected_seeds']}"
    present = f"Present seeds: {report['present_seeds']}; missing seeds: {report['missing_seeds']}"
    interpretation = "COMPLETE describes the selected seed set only; it is not statistical or paper approval."
    if report["status"] == "PARTIAL":
        interpretation = "PARTIAL: selected seeds are missing; this output is not approved as a final paper table. " + interpretation
    sd_note = "Sample SD: undefined when n=1; otherwise ddof=1."
    lines = [report["condition_label"], "", "**" + heading + "**", "", selection + ".", present + ".",
             f"Unselected seeds: {report['unselected_seeds']}.", sd_note, "", interpretation,
             "", "| Metric | Mean | Sample SD |", "|---|---:|---:|"]
    tex = [r"\par\noindent\textbf{" + _latex_escape(report["condition_label"]) + "}",
           r"\par\noindent\textbf{" + _latex_escape(heading) + "}",
           r"\par\noindent " + _latex_escape(selection),
           r"\par\noindent " + _latex_escape(present),
           r"\par\noindent " + _latex_escape(sd_note),
           r"\par\smallskip", r"\begin{tabular}{lrr}", r"Metric & Mean & Sample SD \\", r"\hline"]
    for name, values in report["metrics"].items():
        sd = "undefined" if values["sample_std"] is None else f"{values['sample_std']:.8g}"
        lines.append(f"| {name} | {values['mean']:.8g} | {sd} |")
        tex.append(_latex_escape(name) + f" & {values['mean']:.8g} & {sd}" + r" \\")
    tex.extend([r"\end{tabular}", r"\par\smallskip\noindent " + _latex_escape(interpretation),
                r"\par\noindent Aggregate ID: \texttt{" + _latex_escape(report["aggregate_id"]) + "}",
                r"\par\noindent " + _latex_escape("Source/config/metric evidence: aggregate.json and seed_values.csv.")])
    lines += ["", "Actual constructor parameter counts: " + json.dumps({str(r["seed"]): r["parameter_count"] for r in report["runs"]}),
              "Basis removal changes feature dimensions and parameter counts; no equal-parameter budget is claimed.",
              "", "Source/config/context and exact metric definition:", "", "```json",
              json.dumps({key: report[key] for key in ("aggregate_id", "context_identity", "metric_definition")}, indent=2, sort_keys=True, allow_nan=False),
              "```", "", "Raw seed values and immutable source records: seed_values.csv and aggregate.json."]
    rendered["table.md"] = "\n".join(lines) + "\n"
    rendered["table.tex"] = "\n".join(tex) + "\n"
    publish_output_bundle(destination, {name: text.encode("utf-8") for name, text in rendered.items()}, overwrite=overwrite)
    return report


def aggregate(*, results_root: Path, dataset: str, model_variant: str, condition: str,
              seeds=None, output_dir: Path | None = None, reference_full_summaries=None,
              reference_preference=None, allow_partial=False, test_contract=None,
              overwrite_aggregate=False):
    from .artifacts import validate_artifact, canonical_sha256, canonical_json
    seeds = list(DEFAULT_SEEDS) if seeds is None else seeds
    validate_seeds(seeds)
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS or condition not in ABLATION_CONDITIONS:
        raise PublicError("AGGREGATE_IDENTITY_INVALID")
    if reference_preference not in (None, "reference", "ablation"):
        raise PublicError("REFERENCE_PREFERENCE_INVALID")
    diagnostic = test_contract is not None  # Python-only test injection; no paper CLI override.
    if not diagnostic:
        from .repository import require_dataset_baseline
        require_dataset_baseline(dataset)
    results_root = contained(results_root, workspace_root())
    target = contained(results_root / EXPERIMENT_ID / ("diagnostic" if diagnostic else "formal")
                       / dataset / model_variant / condition, results_root)
    from .output_bundle import assert_publication_idle
    assert_publication_idle(target)
    paths, extra = {}, []
    if target.exists():
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            contained(child, target)
            if child.name in {"aggregate.json", "aggregate.csv", "seed_values.csv", "table.md", "table.tex"} and child.is_file():
                continue
            if not child.name.startswith("seed_"):
                if child.is_dir():
                    raise PublicError("UNEXPECTED_RUN_DIRECTORY")
                continue
            try:
                seed = seed_value(child.name[5:])
            except ValueError:
                raise PublicError("RUN_DIRECTORY_SEED_INVALID") from None
            if child.name != f"seed_{seed}" or not child.is_dir():
                raise PublicError("RUN_DIRECTORY_IDENTITY_INVALID")
            if seed in paths:
                raise PublicError("DUPLICATE_RUN_SEED", seed=seed)
            if seed not in seeds:
                extra.append(seed)
                continue
            # An existing selected directory with no summary is incomplete,
            # not a missing seed eligible for --allow-partial.
            summary_path = contained(child / "summary.json", target)
            if not summary_path.is_file():
                raise PublicError("INCOMPLETE_SELECTED_RUN", seed=seed)
            paths[seed] = summary_path
    references = {}
    for path in reference_full_summaries or []:
        if condition != "full":
            raise PublicError("REFERENCE_REQUIRES_FULL_CONDITION")
        from .references import validate_full_reference
        record = validate_full_reference(contained(path, workspace_root()), dataset=dataset,
                 model_variant=model_variant, test_contract=test_contract, require_comparable=not diagnostic)
        seed = record["seed"]
        if seed not in seeds:
            raise PublicError("REFERENCE_SEED_NOT_SELECTED", seed=seed)
        if seed in references:
            raise PublicError("DUPLICATE_REFERENCE_SEED", seed=seed)
        references[seed] = record
    runs, skipped_preference = [], []
    for seed in seeds:
        ablation = None
        if seed in paths:
            verified = validate_artifact(paths[seed], dataset=dataset, model_variant=model_variant,
                       condition=condition, seed=seed, allow_diagnostic=diagnostic, test_contract=test_contract)
            ablation = _run_record(verified)
        reference = references.get(seed)
        if ablation is not None and reference is not None:
            if reference_preference is None:
                raise PublicError("REFERENCE_AND_ABLATION_SEED_CONFLICT", seed=seed)
            skipped_preference.append({"seed": seed, "excluded_origin":
                                     "ablation" if reference_preference == "reference" else "main_full_reference"})
        chosen = (reference if reference_preference == "reference" else ablation) if ablation and reference else ablation or reference
        if chosen:
            runs.append(chosen)
    present = [run["seed"] for run in runs]
    missing = [seed for seed in seeds if seed not in present]
    if not runs or (missing and not allow_partial):
        raise PublicError("SELECTED_SEEDS_MISSING")
    if not diagnostic and any(run.get("comparison_eligible") is not True for run in runs):
        raise PublicError("REFERENCE_RUNTIME_EVIDENCE_INCOMPLETE")
    if not diagnostic:
        from .runtime import require_runtime_comparable
        for run in runs:
            require_runtime_comparable(run["runtime"])
    if len({run["checkpoint_sha256"] for run in runs}) != len(runs):
        raise PublicError("DUPLICATE_CHECKPOINT_ACROSS_SEEDS")
    contexts = {canonical_json(_compatible_context(run)) for run in runs}
    if len(contexts) != 1:
        raise PublicError("RUN_CONTEXT_INCOMPATIBLE")
    ablations = [run for run in runs if run["origin"] == "ablation"]
    if len({canonical_json(run["source"]) for run in ablations}) > 1 or len({run["comparison_identity"] for run in ablations}) > 1:
        raise PublicError("ABLATION_SOURCE_OR_CONFIGURATION_MIXED")
    metric_names = set(runs[0]["metrics"])
    if not metric_names or any(set(run["metrics"]) != metric_names for run in runs):
        raise PublicError("PAPER_METRIC_SET_MISMATCH")
    metrics = {}
    for name in sorted(metric_names):
        values = [run["metrics"][name] for run in runs]
        metrics[name] = {"mean": statistics.fmean(values),
                         "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                         "n": len(values), "ddof": 1}
    report = {"schema_version": 3, "experiment_id": EXPERIMENT_ID, "dataset": dataset,
        "model_variant": model_variant, "condition": condition,
        "condition_label": ABLATION_CONDITIONS[condition].label,
        "result_kind": "diagnostic_test" if diagnostic else "formal",
        "status": "PARTIAL" if missing else "COMPLETE", "selected_seeds": list(seeds),
        "seed_contract": list(seeds), "expected_seeds": list(seeds), "present_seeds": present,
        "missing_seeds": missing, "unselected_seeds": sorted(extra), "unselected_count": len(extra),
        "seed_count": len(runs), "sample_standard_deviation": "ddof=1; undefined when n=1",
        "context_identity": canonical_sha256(_compatible_context(runs[0])),
        "metric_definition": runs[0]["evaluation_definition"], "runs": runs, "metrics": metrics,
        "preference_exclusions": skipped_preference, "timing_aggregation": "not_performed",
        "metric_values_reevaluated": False, "checkpoint_restore_validated": True}
    return _export(report, output_dir or target, overwrite=overwrite_aggregate)


def parse_args(argv=None):
    parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model-variant", required=True, choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--condition", required=True, choices=sorted(ABLATION_CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=seed_value, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite-aggregate", action="store_true",
                        help="Replace only the five aggregate outputs after full input validation; unrelated to training --overwrite.")
    parser.add_argument("--reference-full-summary", action="append", type=Path, default=[])
    parser.add_argument("--reference-preference", choices=("reference", "ablation"))
    parser.add_argument("--allow-partial", action="store_true", help="Missing selected seeds only; invalid existing runs remain errors.")
    args = parser.parse_args(argv)
    try:
        validate_seeds(args.seeds)
    except ValueError:
        parser.error("INVALID_SEEDS")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .repository import activate_repository, resolve_repository_root
    activate_repository(resolve_repository_root(args.repository_root))
    report = aggregate(results_root=args.results_root, dataset=args.dataset, model_variant=args.model_variant,
        condition=args.condition, seeds=args.seeds, output_dir=args.output_dir,
        reference_full_summaries=args.reference_full_summary, reference_preference=args.reference_preference,
        allow_partial=args.allow_partial, overwrite_aggregate=args.overwrite_aggregate)
    emit_json({key: report[key] for key in ("status", "selected_seeds", "present_seeds", "missing_seeds",
                                          "unselected_seeds", "seed_count", "metrics")})


if __name__ == "__main__":
    public_main(main)
