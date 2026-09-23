"""Read-only installation, source, import, and model-contract checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import validate_condition_variant
from .contracts import ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS
from .public import SafeArgumentParser, public_main, emit_json
from .runtime import configure_runtime, apply_torch_runtime
from .runtime_metadata import capture_runtime_metadata
from .repository import (
    activate_repository,
    import_audit,
    resolve_repository_root,
    source_state,
)


def main(argv: list[str] | None = None) -> None:
    parser = SafeArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model-variant", required=True, choices=sorted(MODEL_VARIANTS)
    )
    parser.add_argument(
        "--condition", required=True, choices=sorted(ABLATION_CONDITIONS)
    )
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_condition_variant(args.condition, args.model_variant)
    except ValueError as exc:
        parser.error(str(exc))
    root = resolve_repository_root(args.repository_root)
    activate_repository(root)
    state = source_state(root, formal=args.formal)
    profile = "formal" if args.formal else "diagnostic"
    # This is a CPU inspection, never a formal training runtime override.
    configure_runtime(cpu=True, profile=profile, threads=1)

    from .models import build_ablation_model, describe_model, preserved_rng_state
    apply_torch_runtime()
    with preserved_rng_state():
        built = build_ablation_model(
            args.dataset, args.model_variant, args.condition, device="cpu"
        )
    report = {
        "repository_id": "selected-source",
        "dataset": args.dataset,
        "model_variant": args.model_variant,
        "condition": args.condition,
        "model": describe_model(built.model),
        "source": state,
        "imports": import_audit(root),
        "runtime": capture_runtime_metadata(profile=profile, device="cpu"),
    }
    emit_json(report)


if __name__ == "__main__":
    public_main(main)
