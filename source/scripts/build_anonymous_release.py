"""Build and smoke-test a Git-less anonymous ZIP from a clean committed ref."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.sirenfno_backend import (  # noqa: E402
    BackendVerificationError,
    RELEASE_FORMAT_VERSION,
    RELEASE_HASH_SCOPE,
    RELEASE_PROVENANCE_FILENAME,
    SIRENFNO_ROOT,
    source_repository_state,
    upstream_manifest,
    verify_sirenfno_checkout,
)
from scripts.audit_anonymity import archive_anonymity_findings  # noqa: E402


ARCHIVE_NAME = "anonymous-cafe-plus-fno.zip"
UPSTREAM_PREFIX = "third_party/SirenFNO/"
REQUIRED_ARCHIVE_MEMBERS = {
    ".gitmodules",
    "LICENSE",
    "README.md",
    RELEASE_PROVENANCE_FILENAME,
    "THIRD_PARTY_NOTICES.md",
    "scripts/setup_sirenfno.py",
    "third_party/UPSTREAM_VERSIONS.json",
    f"{UPSTREAM_PREFIX}LICENSE",
    f"{UPSTREAM_PREFIX}neuralop/__init__.py",
}
GENERATED_OR_DATA_SUFFIXES = (".pt", ".hdf5", ".ckpt", ".pth")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--verify-archive",
        type=Path,
        help=(
            "Verify an existing ZIP and a fresh isolated extraction instead of "
            "building."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an older archive only after all checks pass.",
    )
    return parser.parse_args()


def run_checked(command: list[str], *, cwd: Path = REPOSITORY_ROOT) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def git_output(args: list[str], *, cwd: Path = REPOSITORY_ROOT) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit("Git could not identify the clean release snapshot.")
    return completed.stdout.strip()


def _safe_member_path(member: str) -> bool:
    path = PurePosixPath(member)
    normalized = path.as_posix() + ("/" if member.endswith("/") else "")
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in member
        and "\x00" not in member
        and ":" not in path.parts[0]
        and normalized == member
    )


def _is_source_member(member: str) -> bool:
    """The provenance hash scope is every regular member except itself."""

    return member != RELEASE_PROVENANCE_FILENAME


def archive_source_hashes(archive: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not _is_source_member(info.filename):
                continue
            hashes[info.filename] = hashlib.sha256(bundle.read(info)).hexdigest()
    return hashes


def _upstream_identity() -> dict[str, str]:
    manifest = upstream_manifest()["SirenFNO"]
    return {
        "path": "third_party/SirenFNO",
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "git_tree_sha1": manifest["git_tree_sha1"],
    }


def add_release_provenance(
    archive: Path,
    *,
    source_repository_commit: str,
    source_tree_sha1: str,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "release_format_version": RELEASE_FORMAT_VERSION,
        "source_repository_commit": source_repository_commit,
        "source_tree_sha1": source_tree_sha1,
        "source_repository_dirty": False,
        "bundled_upstream": _upstream_identity(),
        "hash_scope": RELEASE_HASH_SCOPE,
        "files": archive_source_hashes(archive),
    }
    payload = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    info = zipfile.ZipInfo(
        RELEASE_PROVENANCE_FILENAME, date_time=(1980, 1, 1, 0, 0, 0)
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(archive, mode="a") as bundle:
        if RELEASE_PROVENANCE_FILENAME in bundle.namelist():
            raise RuntimeError("Release provenance is already present in the archive.")
        bundle.writestr(info, payload)
    return provenance


def _clone_zip_info(source: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(filename, date_time=source.date_time)
    copied.compress_type = zipfile.ZIP_DEFLATED
    copied.comment = source.comment
    copied.create_system = source.create_system
    copied.external_attr = source.external_attr
    copied.extra = source.extra
    copied.internal_attr = source.internal_attr
    return copied


def append_pinned_upstream_archive(
    release_archive: Path,
    upstream_archive: Path,
) -> None:
    """Merge a separately created upstream Git archive under its public path."""

    with zipfile.ZipFile(release_archive, mode="a") as destination, zipfile.ZipFile(
        upstream_archive
    ) as source:
        existing = set(destination.namelist())
        for info in source.infolist():
            member = UPSTREAM_PREFIX + info.filename
            if not _safe_member_path(member):
                raise RuntimeError("The pinned upstream archive contains an unsafe path.")
            if any(
                part.casefold() == ".git" for part in PurePosixPath(member).parts
            ):
                raise RuntimeError("The pinned upstream archive contains Git metadata.")
            if member in existing:
                raise RuntimeError("The pinned upstream archive contains a duplicate path.")
            destination.writestr(
                _clone_zip_info(info, member),
                b"" if info.is_dir() else source.read(info),
            )
            existing.add(member)


def _git_object_sha1(kind: str, payload: bytes) -> str:
    return hashlib.sha1(
        f"{kind} {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def archive_git_tree_sha1(
    bundle: zipfile.ZipFile,
    prefix: str,
) -> str:
    """Reconstruct a regular-file Git tree identity from ZIP member bytes."""

    tree: dict[str, Any] = {}
    seen = False
    for info in bundle.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        relative = info.filename[len(prefix) :]
        if not relative or not _safe_member_path(relative):
            raise RuntimeError("The bundled upstream tree contains an unsafe path.")
        seen = True
        node = tree
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise RuntimeError("The bundled upstream tree has a path collision.")
            node = child
        if parts[-1] in node:
            raise RuntimeError("The bundled upstream tree has a duplicate path.")
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            mode = "120000"
        elif file_type not in {0, stat.S_IFREG}:
            raise RuntimeError("The bundled upstream tree has an unsupported entry.")
        elif unix_mode & stat.S_IXUSR:
            mode = "100755"
        else:
            mode = "100644"
        node[parts[-1]] = (mode, bundle.read(info))
    if not seen:
        raise RuntimeError("The bundled upstream source is missing.")

    def hash_node(node: dict[str, Any]) -> str:
        entries: list[tuple[bytes, bytes]] = []
        for name, value in node.items():
            encoded_name = name.encode("utf-8")
            if isinstance(value, dict):
                mode = b"40000"
                object_id = hash_node(value)
                sort_name = encoded_name + b"/"
            else:
                mode_text, data = value
                mode = mode_text.encode("ascii")
                object_id = _git_object_sha1("blob", data)
                sort_name = encoded_name
            entry = mode + b" " + encoded_name + b"\0" + bytes.fromhex(object_id)
            entries.append((sort_name, entry))
        payload = b"".join(entry for _, entry in sorted(entries))
        return _git_object_sha1("tree", payload)

    return hash_node(tree)


def _redacted_locations(locations: set[str]) -> str:
    identities = sorted(
        hashlib.sha256(location.encode("utf-8", errors="replace")).hexdigest()[:12]
        for location in locations
    )
    return ", ".join(f"location:{identity}" for identity in identities)


def verify_archive_anonymity(archive: Path) -> None:
    findings = archive_anonymity_findings(archive)
    failures = sorted(
        f"{label} ({len(paths)}): {_redacted_locations(paths)}"
        for label, paths in findings.items()
        if paths
    )
    if failures:
        raise RuntimeError("Archive anonymity scan failed: " + "; ".join(failures))


def _load_provenance(bundle: zipfile.ZipFile) -> dict[str, Any] | None:
    try:
        return json.loads(
            bundle.read(RELEASE_PROVENANCE_FILENAME),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def verify_archive(archive: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            corrupt = bundle.testzip()
            infos = bundle.infolist()
            members = [info.filename for info in infos]
            member_set = set(members)
            required = set(REQUIRED_ARCHIVE_MEMBERS)
            required.update(
                identity["path"]
                for identity in upstream_manifest()["SirenFNO"]["source_files"].values()
            )
            missing = sorted(required - member_set)
            duplicate = sorted(
                name
                for name, count in collections.Counter(members).items()
                if count > 1
            )
            unsafe = sorted(name for name in members if not _safe_member_path(name))
            git_metadata = sorted(
                name
                for name in members
                if any(
                    part.casefold() == ".git"
                    for part in PurePosixPath(name).parts
                )
            )
            symlinks = sorted(
                info.filename
                for info in infos
                if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
            )
            forbidden = sorted(
                name
                for name in members
                if name.lower().endswith(GENERATED_OR_DATA_SUFFIXES)
            )
            provenance = _load_provenance(bundle)

            provenance_error = not isinstance(provenance, dict)
            if isinstance(provenance, dict):
                expected_hashes = archive_source_hashes(archive)
                provenance_error = (
                    provenance.get("release_format_version")
                    != RELEASE_FORMAT_VERSION
                    or provenance.get("source_repository_dirty") is not False
                    or provenance.get("hash_scope") != RELEASE_HASH_SCOPE
                    or provenance.get("bundled_upstream") != _upstream_identity()
                    or not isinstance(provenance.get("source_repository_commit"), str)
                    or not SHA1_RE.fullmatch(provenance["source_repository_commit"])
                    or not isinstance(provenance.get("source_tree_sha1"), str)
                    or not SHA1_RE.fullmatch(provenance["source_tree_sha1"])
                    or provenance.get("files") != expected_hashes
                )

            upstream_tree_error = False
            upstream_blob_error = False
            try:
                actual_tree = archive_git_tree_sha1(bundle, UPSTREAM_PREFIX)
                expected_tree = upstream_manifest()["SirenFNO"]["git_tree_sha1"]
                upstream_tree_error = actual_tree != expected_tree
                for identity in upstream_manifest()["SirenFNO"][
                    "source_files"
                ].values():
                    data = bundle.read(identity["path"])
                    if _git_object_sha1("blob", data) != identity["git_blob_sha1"]:
                        upstream_blob_error = True
            except (KeyError, RuntimeError):
                upstream_tree_error = True
                upstream_blob_error = True
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Release archive is not a readable ZIP.") from exc

    problems = []
    if corrupt:
        problems.append("corrupt ZIP member")
    if missing:
        problems.append(f"missing {len(missing)} required file(s)")
    if duplicate:
        problems.append(f"duplicate ZIP members ({len(duplicate)})")
    if unsafe:
        problems.append(f"unsafe archive paths ({len(unsafe)})")
    if git_metadata:
        problems.append(f"Git metadata present ({len(git_metadata)})")
    if symlinks:
        problems.append(f"symbolic links present ({len(symlinks)})")
    if forbidden:
        problems.append(f"generated/data artifacts present ({len(forbidden)})")
    if provenance_error:
        problems.append("RELEASE_PROVENANCE.json schema or hashes do not match")
    if upstream_tree_error:
        problems.append("bundled SirenFNO Git tree does not match the pinned manifest")
    if upstream_blob_error:
        problems.append("bundled canonical upstream blob does not match the manifest")
    if problems:
        raise RuntimeError("Release archive verification failed: " + "; ".join(problems))
    verify_archive_anonymity(archive)


_GITLESS_SMOKE = r"""
import importlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
from experiments.common.sirenfno_backend import (
    bootstrap_sirenfno_backend,
    source_repository_state,
    verify_pinned_source_blobs,
)
from experiments.configs import burgers1d, cfd1d, cfd2d, reacdiff1d

state = source_repository_state(allow_dirty=False)
assert state["source_repository_dirty"] is False
backend = bootstrap_sirenfno_backend()
assert backend["verification"] == "release_manifest_and_git_tree"
expected_blobs = {}
for source_blobs in (
    burgers1d.BURGERS_SOURCE_BLOBS,
    cfd1d.CFD1D_SOURCE_BLOBS,
    cfd2d.CFD2D_SOURCE_BLOBS,
    reacdiff1d.REACDIFF_SOURCE_BLOBS,
):
    for path, blob in source_blobs.items():
        assert path not in expected_blobs or expected_blobs[path] == blob
        expected_blobs[path] = blob
assert verify_pinned_source_blobs(expected_blobs) == expected_blobs
for module in (
    "experiments.configs.airfoil",
    "experiments.configs.burgers1d",
    "experiments.configs.cfd1d",
    "experiments.configs.cfd2d",
    "experiments.configs.darcy",
    "experiments.configs.ns2d",
    "experiments.configs.reacdiff1d",
):
    importlib.import_module(module)
for source in root.rglob("*.py"):
    compile(source.read_bytes(), source.as_posix(), "exec")
if sys.argv[2] == "runtime":
    import neuralop
    from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D
    from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D
    expected = (root / "third_party" / "SirenFNO" / "neuralop").resolve()
    assert Path(neuralop.__file__).resolve().is_relative_to(expected)
    assert callable(CAFEPlusFNO1D) and callable(CAFEPlusFNO2D)
print("GITLESS_RELEASE_SMOKE_OK")
"""


def _redacted_smoke_failure(completed: subprocess.CompletedProcess[str]) -> str:
    """Return a useful child-process diagnostic without echoing its output."""

    combined = (completed.stdout + "\n" + completed.stderr).encode(
        "utf-8", errors="replace"
    )
    diagnostic = hashlib.sha256(combined).hexdigest()[:12]
    allowed_categories = (
        "AssertionError",
        "BackendVerificationError",
        "ImportError",
        "ModuleNotFoundError",
        "OSError",
        "RuntimeError",
        "SyntaxError",
    )
    category = next(
        (name for name in allowed_categories if name in completed.stderr),
        "unknown",
    )
    return f"category {category}, diagnostic {diagnostic}"


def verify_extracted_release(
    archive: Path,
    *,
    runtime_imports: bool = True,
) -> None:
    """Extract afresh and verify without inheriting PYTHONPATH/editable imports."""

    verify_archive(archive)
    with tempfile.TemporaryDirectory(prefix="anonymous-release-smoke-") as temp:
        extracted = Path(temp) / "release"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        if any(path.name.casefold() == ".git" for path in extracted.rglob("*")):
            raise RuntimeError("Extracted release contains forbidden Git metadata.")
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
                _GITLESS_SMOKE,
                str(extracted),
                "runtime" if runtime_imports else "source-only",
            ],
            cwd=temp,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or "GITLESS_RELEASE_SMOKE_OK" not in completed.stdout:
            # Subprocess output may contain a local path, so do not echo it.
            raise RuntimeError(
                "Isolated Git-less release smoke failed "
                f"(exit {completed.returncode}; {_redacted_smoke_failure(completed)})."
            )


def release_source_identity() -> tuple[str, str, str]:
    """Fail before building unless HEAD and its pinned gitlink are clean."""

    try:
        source_state = source_repository_state(allow_dirty=False)
    except BackendVerificationError as exc:
        raise RuntimeError(
            "Official release creation requires a clean committed source tree; "
            "source identity verification failed."
        ) from exc
    source_commit = source_state["source_repository_commit"]
    source_tree = git_output(["rev-parse", "HEAD^{tree}"])
    expected_upstream_commit = upstream_manifest()["SirenFNO"]["commit"]
    recorded_gitlink = git_output(["rev-parse", "HEAD:third_party/SirenFNO"])
    if recorded_gitlink != expected_upstream_commit:
        raise SystemExit(
            "The committed SirenFNO gitlink does not match the pinned manifest."
        )
    verified = verify_sirenfno_checkout()
    if verified["commit"] != expected_upstream_commit:
        raise SystemExit("The checked-out SirenFNO source is not the pinned commit.")
    return source_commit, source_tree, expected_upstream_commit


def main() -> None:
    args = parse_args()

    if args.verify_archive is not None:
        if args.overwrite:
            raise SystemExit("--overwrite cannot be used with --verify-archive.")
        verify_extracted_release(args.verify_archive, runtime_imports=True)
        print("Archive content/anonymity verification: PASS")
        print("Fresh Git-less extraction smoke: PASS")
        return

    # This gate intentionally precedes every expensive audit: an official
    # archive may never silently omit dirty or untracked work in favor of HEAD.
    source_commit, source_tree, upstream_commit = release_source_identity()

    run_checked([sys.executable, "-B", "scripts/audit_anonymity.py", "--ref", "HEAD"])
    run_checked(
        [
            sys.executable,
            "-B",
            "scripts/audit_reproducibility.py",
            "--data-root",
            str(args.data_root),
        ]
    )
    run_checked([sys.executable, "-B", "scripts/verify_environment.py"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_archive = output_dir / ARCHIVE_NAME
    if final_archive.exists() and not args.overwrite:
        raise SystemExit("Release archive already exists; refusing to overwrite it.")

    with tempfile.TemporaryDirectory(prefix="anonymous-release-", dir=output_dir) as temp:
        temporary_archive = Path(temp) / ARCHIVE_NAME
        upstream_archive = Path(temp) / "pinned-sirenfno.zip"
        run_checked(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "archive",
                "--format=zip",
                f"--output={temporary_archive}",
                "HEAD",
            ]
        )
        run_checked(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "archive",
                "--format=zip",
                f"--output={upstream_archive}",
                upstream_commit,
            ],
            cwd=SIRENFNO_ROOT,
        )
        append_pinned_upstream_archive(temporary_archive, upstream_archive)
        add_release_provenance(
            temporary_archive,
            source_repository_commit=source_commit,
            source_tree_sha1=source_tree,
        )
        verify_archive(temporary_archive)
        run_checked(
            [
                sys.executable,
                "-B",
                "scripts/audit_anonymity.py",
                "--ref",
                "HEAD",
                "--archive",
                str(temporary_archive),
            ]
        )
        verify_extracted_release(temporary_archive, runtime_imports=True)
        os.replace(temporary_archive, final_archive)

    print(f"Anonymous supplementary archive: dist/{ARCHIVE_NAME}")
    print("Clean committed source gate: PASS")
    print("Pinned bundled SirenFNO verification: PASS")
    print("RELEASE_PROVENANCE generation: PASS")
    print("Archive content/anonymity verification: PASS")
    print("Fresh Git-less extraction smoke: PASS")


if __name__ == "__main__":
    try:
        main()
    except (BackendVerificationError, RuntimeError) as exc:
        # Library errors are intentionally path/value-free; suppress traceback
        # frames because their absolute source paths can identify a workstation.
        print(f"Release build: FAIL ({exc})", file=sys.stderr)
        raise SystemExit(1) from None
