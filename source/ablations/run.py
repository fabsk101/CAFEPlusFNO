"""Run the default seed set for one explicit dataset/model/condition tuple."""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

from .contracts import validate_condition_variant
from .contracts import ABLATION_CONDITIONS, DATASETS, DEFAULT_SEEDS, MODEL_VARIANTS, seed_value, validate_seeds
from .runtime import contained, output_path, workspace_root
from .repository import PACKAGE_ROOT
from .public import SafeArgumentParser, PublicError, emit_json, public_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = SafeArgumentParser(
        description=(
            "Run one selected ablation tuple. This command never expands across "
            "datasets, model variants, or conditions."
        )
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model-variant", required=True, choices=sorted(MODEL_VARIANTS)
    )
    parser.add_argument(
        "--condition", required=True, choices=sorted(ABLATION_CONDITIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=seed_value, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostic-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_condition_variant(args.condition, args.model_variant)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_seeds(args.seeds)
    except ValueError:
        parser.error("INVALID_SEEDS")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .repository import (baseline_profile, load_source_manifest, require_dataset_baseline,
                             resolve_repository_root, verify_base_sources)
    require_dataset_baseline(args.dataset)
    verify_base_sources(resolve_repository_root(args.repository_root))
    contained(args.data_root, workspace_root())
    if args.results_root:
        output_path(args.results_root)
    emit_json({"status": "DRY_RUN" if args.dry_run else "PLAN", "dataset": args.dataset,
               "baseline_profile": baseline_profile(),
               "base_source_commit": load_source_manifest()["base_source_commit"],
               "model_variant": args.model_variant, "condition": args.condition,
               "requested_seeds": args.seeds, "process_count": len(args.seeds),
               "device": args.device, "thread_override": args.threads,
               "epoch_resume_supported": False, "completed_run_reuse": "strict_validation"})
    for seed in args.seeds:
        command = [
            sys.executable,
            "-I", "-S", str(PACKAGE_ROOT / "isolated.py"),
            "--module", "train", "--workspace-root", str(workspace_root()),
            "--dataset",
            args.dataset,
            "--model-variant",
            args.model_variant,
            "--condition",
            args.condition,
            "--seed",
            str(seed),
            "--requested-seeds", *map(str, args.seeds),
            "--data-root",
            str(args.data_root),
            "--device",
            args.device,
        ]
        selected_root = args.repository_root or os.environ.get("CAFE_ABLATION_REPOSITORY_ROOT")
        if selected_root:
            command.extend(("--repository-root", str(selected_root)))
        if args.results_root:
            command.extend(("--results-root", str(args.results_root)))
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        if args.overwrite:
            command.append("--overwrite")
        if args.diagnostic_source:
            command.append("--diagnostic-source")
        if not args.dry_run:
            # Child launcher sanitizes adapter progress/errors; never publish
            # the raw command, whose interpreter/data paths may be personal.
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise PublicError("SEED_PROCESS_FAILED", dataset=args.dataset,
                                  model_variant=args.model_variant, condition=args.condition, seed=seed)


if __name__ == "__main__":
    public_main(main)
