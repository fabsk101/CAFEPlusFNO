"""Initialize (when requested) and verify the pinned SirenFNO checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.sirenfno_backend import (  # noqa: E402
    BackendVerificationError,
    SIRENFNO_ROOT,
    verify_sirenfno_checkout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Initialize a missing submodule or clone the exact public revision.",
    )
    return parser.parse_args()


def _run_git(args: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError("Git could not initialize the pinned SirenFNO source.")


def install_missing_checkout() -> None:
    if SIRENFNO_ROOT.exists():
        raise RuntimeError(
            "third_party/SirenFNO exists but is not verifiable; it will not be overwritten."
        )
    manifest = json.loads(
        (REPOSITORY_ROOT / "third_party" / "UPSTREAM_VERSIONS.json").read_text(
            encoding="utf-8"
        )
    )["SirenFNO"]
    if (REPOSITORY_ROOT / ".git").exists() and (
        REPOSITORY_ROOT / ".gitmodules"
    ).is_file():
        _run_git(
            ["submodule", "update", "--init", "--recursive", "--", "third_party/SirenFNO"],
            REPOSITORY_ROOT,
        )
    else:
        SIRENFNO_ROOT.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            ["clone", "--no-checkout", manifest["repository"], str(SIRENFNO_ROOT)],
            REPOSITORY_ROOT,
        )
        _run_git(["checkout", "--detach", manifest["commit"]], SIRENFNO_ROOT)


def main() -> None:
    args = parse_args()
    if not SIRENFNO_ROOT.exists() and args.install:
        install_missing_checkout()
    try:
        verified = verify_sirenfno_checkout()
    except BackendVerificationError as exc:
        print(f"SirenFNO setup: FAIL ({exc})")
        raise SystemExit(1) from exc

    print("SirenFNO setup: PASS")
    print(f"Repository: {verified['repository']}")
    print(f"Commit: {verified['commit']}")
    print(f"Bundled NeuralOperator: {verified['neuraloperator_source']}")
    print(f"License: {verified['license']}")


if __name__ == "__main__":
    main()
