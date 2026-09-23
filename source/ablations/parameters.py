"""Report exact dimensions and parameter deltas without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS, conditions_for_variant
from .repository import activate_repository, resolve_repository_root, baseline_datasets
from .runtime import output_path, configure_runtime
from .public import SafeArgumentParser, emit_json, public_main, public_path


def main(argv: list[str] | None = None) -> None:
    parser = SafeArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument(
        "--model-variant", choices=sorted(MODEL_VARIANTS)
    )
    parser.add_argument("--all", action="store_true", help="Report all seven datasets and four factorizations on the reviewed source")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.all and (args.dataset is None or args.model_variant is None):
        parser.error("Select --dataset and --model-variant, or --all")
    if args.all and (args.dataset or args.model_variant):
        parser.error("--all cannot be combined with dataset/model selection")
    root = resolve_repository_root(args.repository_root)
    activate_repository(root)
    configure_runtime(cpu=True, profile="diagnostic")
    import torch
    torch.set_num_threads(1)
    from .models import build_ablation_model, describe_model, preserved_rng_state

    reports = []
    for dataset in (baseline_datasets() if args.all else [args.dataset]):
        for variant in (MODEL_VARIANTS if args.all else [args.model_variant]):
            rows = {}
            for condition in conditions_for_variant(variant):
                with preserved_rng_state():
                    built = build_ablation_model(dataset, variant, condition, device="cpu")
                rows[condition] = describe_model(built.model)
                del built
            full_count = rows["full"]["parameter_count"]
            for row in rows.values():
                row["parameter_delta_from_full"] = row["parameter_count"] - full_count
            reports.append({"dataset": dataset, "model_variant": variant, "conditions": rows})
    payload = reports if args.all else reports[0]
    if args.output:
        path = output_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        emit_json({"status": "PASS", "model_descriptions": sum(len(item["conditions"]) for item in reports),
                   "artifact": public_path(path, artifact_id="parameter-report")})
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    public_main(main)
