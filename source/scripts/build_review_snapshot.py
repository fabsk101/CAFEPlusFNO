"""Build a curated, explicitly non-release snapshot of the current worktree."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.configs import burgers1d, cfd1d, cfd2d, reacdiff1d  # noqa: E402
from experiments.common.sirenfno_backend import (  # noqa: E402
    SIRENFNO_ROOT,
    upstream_manifest,
    verify_sirenfno_checkout,
)
from scripts.build_anonymous_release import archive_git_tree_sha1  # noqa: E402


ARCHIVE_NAME = "CAFEPlusFNO_full_review.zip"
FILE_LIST_NAME = "CAFEPlusFNO_full_review_files.txt"
SHA256_LIST_NAME = "CAFEPlusFNO_full_review_sha256.txt"
NOTICE_NAME = "REVIEW_SNAPSHOT_NOTICE.txt"

TOP_LEVEL_FILES = {
    ".gitattributes",
    ".gitignore",
    ".gitmodules",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "environment.yml",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements.txt",
}
SOURCE_DIRECTORIES = ("models", "experiments", "scripts", "tests", "third_party")
DATA_FILES = {
    "data/DATASETS.json",
    "data/README.md",
    "data/download_data.py",
}
FORBIDDEN_PARTS = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "env",
    "venv",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".coverage",
    ".env",
    ".h5",
    ".hdf5",
    ".joblib",
    ".key",
    ".log",
    ".mat",
    ".nc",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".zip",
    ".zarr",
}
FORBIDDEN_FILENAMES = {
    "credentials.json",
    "release_provenance.json",
    "secrets.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing review artifacts after verification.",
    )
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def source_payload(relative: str, source: Path) -> bytes:
    """Read worktree bytes, using canonical Git bytes for the pinned upstream."""

    upstream_prefix = "third_party/SirenFNO/"
    if not relative.startswith(upstream_prefix):
        return source.read_bytes()
    upstream_relative = relative[len(upstream_prefix) :]
    completed = subprocess.run(
        ["git", "show", f"HEAD:{upstream_relative}"],
        cwd=SIRENFNO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Pinned upstream source bytes could not be read from Git.")
    return completed.stdout


def _safe_source_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    lowered_parts = {part.casefold() for part in path.parts}
    lowered_name = path.name.casefold()
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in relative
        and not lowered_parts.intersection(FORBIDDEN_PARTS)
        and lowered_name not in FORBIDDEN_FILENAMES
        and not lowered_name.startswith(".env")
        and path.suffix.casefold() not in FORBIDDEN_SUFFIXES
        and relative not in {ARCHIVE_NAME, FILE_LIST_NAME, SHA256_LIST_NAME}
    )


def selected_source_files() -> list[tuple[str, Path]]:
    candidates: set[Path] = {
        REPOSITORY_ROOT / relative for relative in TOP_LEVEL_FILES
    }
    candidates.update(REPOSITORY_ROOT / relative for relative in DATA_FILES)
    for directory in SOURCE_DIRECTORIES:
        root = REPOSITORY_ROOT / directory
        if not root.is_dir():
            raise RuntimeError(f"Required review source directory is missing: {directory}")
        candidates.update(path for path in root.rglob("*") if path.is_file())

    selected: list[tuple[str, Path]] = []
    for source in candidates:
        if not source.exists():
            continue
        if source.is_symlink():
            raise RuntimeError("Review source contains a symbolic link.")
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError("Review source resolved outside the repository.") from exc
        if _safe_source_path(relative):
            selected.append((relative, resolved))

    selected.sort(key=lambda item: item[0])
    names = [relative for relative, _ in selected]
    if len(names) != len(set(names)):
        raise RuntimeError("Review source selection contains duplicate paths.")
    missing = sorted(
        relative
        for relative in TOP_LEVEL_FILES | DATA_FILES
        if relative not in set(names)
    )
    if missing:
        raise RuntimeError(
            f"Required review source file selection is incomplete ({len(missing)} missing)."
        )
    return selected


def worktree_is_dirty() -> bool:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=dirty",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Git could not inspect the review worktree state.")
    return bool(completed.stdout.strip())


def review_notice(*, dirty: bool) -> bytes:
    return (
        "CAFEPlusFNO WORKTREE REVIEW SNAPSHOT\n"
        "\n"
        "This archive is for source review only. It is not the official anonymous "
        "release.\n"
        f"Source worktree dirty when packaged: {str(dirty).lower()}\n"
        "RELEASE_PROVENANCE.json is intentionally absent. Review and commit the "
        "intended files,\n"
        "then use scripts/build_anonymous_release.py to create the official archive.\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_archive(
    archive: Path,
    selected: list[tuple[str, Path]],
    *,
    dirty: bool,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, source in selected:
            payload = source_payload(relative, source)
            bundle.writestr(_zip_info(relative), payload)
            hashes[relative] = _sha256(payload)
        notice = review_notice(dirty=dirty)
        bundle.writestr(_zip_info(NOTICE_NAME), notice)
        hashes[NOTICE_NAME] = _sha256(notice)
    return dict(sorted(hashes.items()))


def verify_archive(archive: Path, expected_hashes: dict[str, str]) -> None:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if bundle.testzip() is not None:
            raise RuntimeError("Review archive contains a corrupt member.")
        if len(names) != len(set(names)) or set(names) != set(expected_hashes):
            raise RuntimeError("Review archive inventory differs from its selection.")
        if "RELEASE_PROVENANCE.json" in names:
            raise RuntimeError("Review archive must not impersonate an official release.")
        for info in infos:
            if info.is_dir() or not _safe_source_path(info.filename):
                if info.filename != NOTICE_NAME:
                    raise RuntimeError("Review archive contains a forbidden member.")
            if _sha256(bundle.read(info)) != expected_hashes[info.filename]:
                raise RuntimeError("Review archive member hash mismatch.")

        expected_upstream = upstream_manifest()["SirenFNO"]
        actual_tree = archive_git_tree_sha1(bundle, "third_party/SirenFNO/")
        if actual_tree != expected_upstream["git_tree_sha1"]:
            raise RuntimeError("Bundled SirenFNO tree differs from the pinned source.")
        expected_blobs: dict[str, str] = {}
        for source_blobs in (
            burgers1d.BURGERS_SOURCE_BLOBS,
            cfd1d.CFD1D_SOURCE_BLOBS,
            cfd2d.CFD2D_SOURCE_BLOBS,
            reacdiff1d.REACDIFF_SOURCE_BLOBS,
        ):
            for relative, expected_blob in source_blobs.items():
                previous = expected_blobs.setdefault(relative, expected_blob)
                if previous != expected_blob:
                    raise RuntimeError("Conflicting pinned upstream blob identities.")
        for relative, expected_blob in expected_blobs.items():
            payload = bundle.read(f"third_party/SirenFNO/{relative}")
            if _git_blob_sha1(payload) != expected_blob:
                raise RuntimeError("Bundled upstream source blob differs from its pin.")


_GITLESS_REVIEW_SMOKE = r"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
assert not (root / ".git").exists()
assert not (root / "RELEASE_PROVENANCE.json").exists()
assert "not the official anonymous release" in (
    root / "REVIEW_SNAPSHOT_NOTICE.txt"
).read_text(encoding="utf-8")
for source in root.rglob("*.py"):
    compile(source.read_bytes(), source.as_posix(), "exec")
for source in root.rglob("*.json"):
    json.loads(source.read_text(encoding="utf-8"))
for module in (
    "experiments.configs.airfoil",
    "experiments.configs.burgers1d",
    "experiments.configs.cfd1d",
    "experiments.configs.cfd2d",
    "experiments.configs.darcy",
    "experiments.configs.ns2d",
    "experiments.configs.reacdiff1d",
    "models.Cafe_Plus_FNO1D",
    "models.Cafe_Plus_FNO2D",
):
    importlib.import_module(module)
from experiments.common.sirenfno_backend import git_tree_sha1, upstream_manifest

manifest = upstream_manifest()["SirenFNO"]
upstream = root / "third_party" / "SirenFNO"
assert git_tree_sha1(upstream) == manifest["git_tree_sha1"]
expected_blobs = {}
from experiments.configs import burgers1d, cfd1d, cfd2d, reacdiff1d
for source_blobs in (
    burgers1d.BURGERS_SOURCE_BLOBS,
    cfd1d.CFD1D_SOURCE_BLOBS,
    cfd2d.CFD2D_SOURCE_BLOBS,
    reacdiff1d.REACDIFF_SOURCE_BLOBS,
):
    for relative, expected in source_blobs.items():
        assert relative not in expected_blobs or expected_blobs[relative] == expected
        expected_blobs[relative] = expected
for relative, expected in expected_blobs.items():
    payload = (upstream / relative).read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    assert hashlib.sha1(header + payload).hexdigest() == expected
print("GITLESS_REVIEW_SMOKE_OK")
"""


def verify_extracted_snapshot(
    archive: Path,
    expected_hashes: dict[str, str],
) -> None:
    """Verify a fresh Git-less extraction without treating it as a release."""

    with tempfile.TemporaryDirectory(prefix="review-snapshot-smoke-") as temp:
        extracted = Path(temp) / "review"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        extracted_hashes = {
            path.relative_to(extracted).as_posix(): _sha256(path.read_bytes())
            for path in extracted.rglob("*")
            if path.is_file()
        }
        if extracted_hashes != expected_hashes:
            raise RuntimeError("Fresh review extraction differs from the ZIP inventory.")
        if any(path.name.casefold() == ".git" for path in extracted.rglob("*")):
            raise RuntimeError("Fresh review extraction contains Git metadata.")

        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _GITLESS_REVIEW_SMOKE,
                str(extracted),
            ],
            cwd=temp,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            completed.returncode != 0
            or "GITLESS_REVIEW_SMOKE_OK" not in completed.stdout
        ):
            diagnostic = _sha256(
                (completed.stdout + "\n" + completed.stderr).encode(
                    "utf-8", errors="replace"
                )
            )[:12]
            raise RuntimeError(
                "Fresh Git-less review extraction smoke failed "
                f"(exit {completed.returncode}; diagnostic {diagnostic})."
            )


def write_inventory(path: Path, hashes: dict[str, str]) -> None:
    lines = [
        "# CAFEPlusFNO_full_review.zip file inventory",
        "# Review snapshot only; not an official anonymous release.",
        f"# regular_files={len(hashes)}",
        *hashes,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_sha256_list(
    path: Path,
    archive: Path,
    hashes: dict[str, str],
) -> None:
    lines = [
        f"{_sha256(archive.read_bytes())}  {ARCHIVE_NAME}",
        *(f"{digest}  {ARCHIVE_NAME}/{relative}" for relative, digest in hashes.items()),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_archive = output_dir / ARCHIVE_NAME
    final_inventory = output_dir / FILE_LIST_NAME
    final_sha256 = output_dir / SHA256_LIST_NAME
    final_outputs = (final_archive, final_inventory, final_sha256)
    if not args.overwrite and any(path.exists() for path in final_outputs):
        raise SystemExit("Review artifact already exists; refusing to overwrite it.")

    selected = selected_source_files()
    verify_sirenfno_checkout()
    dirty = worktree_is_dirty()
    with tempfile.TemporaryDirectory(prefix=".review-snapshot-", dir=output_dir) as temp:
        temporary_root = Path(temp)
        archive = temporary_root / ARCHIVE_NAME
        inventory = temporary_root / FILE_LIST_NAME
        sha256_list = temporary_root / SHA256_LIST_NAME
        hashes = build_archive(archive, selected, dirty=dirty)
        verify_archive(archive, hashes)
        verify_extracted_snapshot(archive, hashes)
        write_inventory(inventory, hashes)
        write_sha256_list(sha256_list, archive, hashes)
        for temporary, final in zip(
            (archive, inventory, sha256_list),
            final_outputs,
            strict=True,
        ):
            os.replace(temporary, final)

    print(f"Review snapshot: {ARCHIVE_NAME}")
    print(f"File inventory: {FILE_LIST_NAME}")
    print(f"SHA256 inventory: {SHA256_LIST_NAME}")
    print(f"Regular files: {len(hashes)}")
    print(f"Source worktree dirty: {dirty}")
    print("Official RELEASE_PROVENANCE.json included: False")
    print("Fresh Git-less review extraction: PASS")


if __name__ == "__main__":
    main()
