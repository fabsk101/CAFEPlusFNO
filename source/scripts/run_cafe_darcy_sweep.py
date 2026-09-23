"""Run only the four CAFE+FNO Darcy variants in isolated subprocesses.

The default output root is deliberately separate from completed baseline runs.
Each run is accepted as complete only when its CSV, summary, and checkpoint all
report epoch 500.  Incomplete artifacts are fail-closed unless ``--overwrite``
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.configs.seeds import PAPER_SEEDS  # noqa: E402
from scripts.run_paper_sweep import (  # noqa: E402
    DATASETS,
    is_complete,
    resolve_current_source_commit,
)


CAFE_MODELS = (
    "cafe_plus_fno",
    "cp_cafe_plus_fno",
    "tt_cafe_plus_fno",
    "tucker_cafe_plus_fno",
)
DEFAULT_RESULTS_ROOT = (
    Path("results") / "darcy128_cafe_compact_v1_sirenfno_81918ec"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow train_darcy to replace known files for these CAFE runs.",
    )
    return parser.parse_args(argv)


def build_command(
    *,
    model: str,
    seed: int,
    data_root: Path,
    results_root: Path,
    device: str,
    overwrite: bool,
) -> list[str]:
    if model not in CAFE_MODELS:
        raise ValueError(f"Refusing non-CAFE model: {model}")
    command = [
        sys.executable,
        "-B",
        "-m",
        "experiments.train_darcy",
        "--model",
        model,
        "--seed",
        str(seed),
        "--data-root",
        str(data_root),
        "--results-root",
        str(results_root),
        "--device",
        device,
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    completion_spec = DATASETS["darcy128"]
    current_source_commit = resolve_current_source_commit(completion_spec)
    for model in CAFE_MODELS:
        for seed in PAPER_SEEDS:
            run_dir = args.results_root / model / f"seed_{seed}"
            complete = is_complete(
                run_dir,
                spec=completion_spec,
                model=model,
                seed=seed,
                current_source_commit=current_source_commit,
            )
            if complete and not args.overwrite:
                print(f"COMPLETE: {model}/seed_{seed}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
                raise SystemExit(
                    f"Incomplete run exists for {model}/seed_{seed}; refusing to "
                    "overwrite it. Review it or rerun with explicit --overwrite."
                )
            command = build_command(
                model=model,
                seed=seed,
                data_root=args.data_root,
                results_root=args.results_root,
                device=args.device,
                overwrite=args.overwrite,
            )
            completed = subprocess.run(
                command, cwd=REPOSITORY_ROOT, check=False
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            if not is_complete(
                run_dir,
                spec=completion_spec,
                model=model,
                seed=seed,
                current_source_commit=current_source_commit,
            ):
                raise SystemExit(
                    f"Completion audit failed for {model}/seed_{seed}"
                )


if __name__ == "__main__":
    main()
