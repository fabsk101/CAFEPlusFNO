"""Dataset checksum validation with public-safe metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.common.sirenfno_backend import REPOSITORY_ROOT


DATASET_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "DATASETS.json"


def dataset_manifest() -> dict[str, Any]:
    return json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(dataset_id: str, data_root: Path) -> dict[str, Any]:
    """Verify exact files and return metadata containing no local path."""

    specification = dataset_manifest()[dataset_id]
    verified_files: dict[str, dict[str, Any]] = {}
    for filename, identity in specification["files"].items():
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required dataset file is missing: data/{filename}. "
                "Run the documented dataset preparation command."
            )
        if path.stat().st_size != identity["size_bytes"]:
            raise RuntimeError(f"Dataset size mismatch: data/{filename}")
        actual = sha256_file(path)
        if actual != identity["sha256"]:
            raise RuntimeError(f"Dataset SHA256 mismatch: data/{filename}")
        verified_files[filename] = {
            "path": f"data/{filename}",
            "sha256": actual,
            "size_bytes": identity["size_bytes"],
        }

    return {
        "dataset_id": dataset_id,
        "source": specification["source"],
        "resolution": specification["resolution"],
        "n_train": specification["n_train"],
        "n_test": specification["n_test"],
        "files": verified_files,
    }


__all__ = [
    "DATASET_MANIFEST_PATH",
    "dataset_manifest",
    "sha256_file",
    "verify_dataset",
]
