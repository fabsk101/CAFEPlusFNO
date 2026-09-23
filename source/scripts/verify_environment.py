"""Verify the pinned ICLR paper execution environment and backend."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.sirenfno_backend import (  # noqa: E402
    BackendVerificationError,
    bootstrap_sirenfno_backend,
)


EXPECTED_PYTHON = "3.13.11"
EXPECTED_CUDA = "12.8"
EXPECTED_DISTRIBUTIONS = {
    "h5py": "3.15.1",
    "numpy": "2.4.6",
    "torch": "2.8.0",
    "tensorly": "0.9.0",
    "tensorly-torch": "0.5.0",
    "opt_einsum": "3.4.0",
    "requests": "2.32.5",
}


def base_version(version: str) -> str:
    return version.split("+", 1)[0]


def main() -> None:
    failures: list[str] = []
    try:
        backend = bootstrap_sirenfno_backend()
    except BackendVerificationError as exc:
        print(f"SirenFNO backend: FAIL ({exc})")
        raise SystemExit(1) from exc

    import neuralop
    import torch

    print("ENVIRONMENT VERIFICATION")
    print("=" * 36)

    python_actual = platform.python_version()
    python_ok = python_actual == EXPECTED_PYTHON
    print(f"Python {EXPECTED_PYTHON:<18} {'PASS' if python_ok else 'FAIL'} ({python_actual})")
    if not python_ok:
        failures.append("Python version")

    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        ok = base_version(actual) == expected
        print(f"{distribution + ' ' + expected:<26} {'PASS' if ok else 'FAIL'} ({actual})")
        if not ok:
            failures.append(distribution)

    cuda_ok = torch.version.cuda == EXPECTED_CUDA
    available_ok = torch.cuda.is_available()
    neuralop_ok = str(neuralop.__version__) == backend["neuraloperator_version"]
    print(f"CUDA runtime {EXPECTED_CUDA:<13} {'PASS' if cuda_ok else 'FAIL'} ({torch.version.cuda})")
    print(f"CUDA available{'':<14} {'PASS' if available_ok else 'FAIL'} ({available_ok})")
    print(
        f"Bundled neuralop {backend['neuraloperator_version']:<8} "
        f"{'PASS' if neuralop_ok else 'FAIL'} ({neuralop.__version__})"
    )
    print(f"SirenFNO commit{'':<11} PASS ({backend['commit']})")
    print(f"NeuralOperator source{'':<4} PASS ({backend['neuraloperator_source']})")
    if not cuda_ok:
        failures.append("CUDA runtime")
    if not available_ok:
        failures.append("CUDA availability")
    if not neuralop_ok:
        failures.append("bundled neuralop version")

    if failures:
        print("Environment result: FAIL")
        raise SystemExit(1)
    print("Environment result: PASS")


if __name__ == "__main__":
    main()
