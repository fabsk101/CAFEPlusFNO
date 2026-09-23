"""Pinned SirenFNO backend bootstrap and public-safe provenance helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPOSITORY_ROOT / "third_party"
SIRENFNO_ROOT = (THIRD_PARTY_ROOT / "SirenFNO").resolve()
NEURALOP_ROOT = (SIRENFNO_ROOT / "neuralop").resolve()
MANIFEST_PATH = THIRD_PARTY_ROOT / "UPSTREAM_VERSIONS.json"
RELEASE_PROVENANCE_FILENAME = "RELEASE_PROVENANCE.json"
RELEASE_FORMAT_VERSION = 2
RELEASE_HASH_SCOPE = {
    "algorithm": "sha256",
    "members": "all_regular_archive_members_except_RELEASE_PROVENANCE.json",
}


class BackendVerificationError(RuntimeError):
    """Raised when the pinned paper backend cannot be verified."""


def upstream_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise BackendVerificationError("Missing third_party/UPSTREAM_VERSIONS.json")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise BackendVerificationError("Git is required to verify source revisions.") from exc


def _git(args: list[str], cwd: Path) -> str:
    completed = _run_git(args, cwd)
    if completed.returncode != 0:
        raise BackendVerificationError(
            "Git source verification failed for the anonymous experiment snapshot."
        )
    return completed.stdout.strip()


def _git_quiet(args: list[str], cwd: Path) -> bool:
    completed = _run_git(args, cwd)
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise BackendVerificationError(
        "Git could not verify the anonymous experiment snapshot."
    )


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_object_sha1(kind: str, payload: bytes) -> str:
    header = f"{kind} {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def git_tree_sha1(directory: Path) -> str:
    """Compute a Git tree identity from a plain directory without using Git.

    Release archives preserve the pinned SirenFNO snapshot without its Git
    control files.  This routine provides the same object identity check in a
    Git-less extraction.  The pinned upstream tree contains regular 100644
    files; executable and symlink modes are handled for fail-closed reuse.
    """

    root = directory.resolve()
    if not root.is_dir():
        raise BackendVerificationError("The bundled upstream source is missing.")

    def hash_directory(current: Path) -> str:
        entries: list[tuple[bytes, bytes]] = []
        try:
            children = list(current.iterdir())
        except OSError as exc:
            raise BackendVerificationError(
                "The bundled upstream source could not be inspected."
            ) from exc
        for child in children:
            if child.name.casefold() == ".git":
                raise BackendVerificationError(
                    "Git metadata is forbidden in the bundled upstream source."
                )
            name = os.fsencode(child.name)
            try:
                if child.is_symlink():
                    mode = b"120000"
                    payload = os.fsencode(os.readlink(child))
                    object_id = _git_object_sha1("blob", payload)
                    sort_name = name
                elif child.is_dir():
                    mode = b"40000"
                    object_id = hash_directory(child)
                    sort_name = name + b"/"
                elif child.is_file():
                    executable = bool(child.stat().st_mode & stat.S_IXUSR)
                    mode = b"100755" if executable else b"100644"
                    object_id = _git_object_sha1("blob", child.read_bytes())
                    sort_name = name
                else:
                    raise BackendVerificationError(
                        "The bundled upstream source contains an unsupported entry."
                    )
            except OSError as exc:
                raise BackendVerificationError(
                    "The bundled upstream source could not be inspected."
                ) from exc
            encoded = mode + b" " + name + b"\0" + bytes.fromhex(object_id)
            entries.append((sort_name, encoded))
        payload = b"".join(encoded for _, encoded in sorted(entries))
        return _git_object_sha1("tree", payload)

    return hash_directory(root)


def public_source_path(path: Path) -> str:
    """Return a repository-relative path suitable for public run artifacts."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise BackendVerificationError(
            "A runtime implementation resolved outside the anonymous repository."
        ) from exc


def _module_source(obj: Any, label: str) -> Path:
    try:
        source = inspect.getfile(obj)
    except (TypeError, OSError) as exc:
        raise BackendVerificationError(
            f"Cannot inspect the implementation source for {label}."
        ) from exc
    return Path(source).resolve()


def verify_git_checkout_cleanliness(checkout_root: Path) -> None:
    """Reject staged/untracked changes and tracked changes beyond EOL style."""

    semantic_tracked_clean = _git_quiet(
        ["diff", "--ignore-space-at-eol", "--quiet", "HEAD", "--"],
        checkout_root,
    )
    staged_clean = _git_quiet(
        ["diff", "--cached", "--quiet", "HEAD", "--"],
        checkout_root,
    )
    untracked = _git(
        ["ls-files", "--others", "--exclude-standard"],
        checkout_root,
    )
    if not semantic_tracked_clean or not staged_clean or bool(untracked):
        raise BackendVerificationError(
            "The pinned third-party checkout has semantic, staged, or untracked "
            "changes. EOL-only working-tree differences are permitted."
        )


def verify_sirenfno_checkout() -> dict[str, Any]:
    """Verify the pinned checkout or its Git-less release representation."""

    manifest = upstream_manifest()["SirenFNO"]
    if not (SIRENFNO_ROOT / ".git").exists():
        if (REPOSITORY_ROOT / ".git").exists():
            raise BackendVerificationError(
                "third_party/SirenFNO is not an initialized Git checkout."
            )
        release = verify_release_provenance(REPOSITORY_ROOT, allow_dirty=False)
        expected_tree = manifest.get("git_tree_sha1")
        if not isinstance(expected_tree, str) or not re.fullmatch(
            r"[0-9a-f]{40}", expected_tree
        ):
            raise BackendVerificationError(
                "The upstream manifest lacks a valid SirenFNO tree identity."
            )
        actual_tree = git_tree_sha1(SIRENFNO_ROOT)
        neuralop_tree = git_tree_sha1(NEURALOP_ROOT)
        if actual_tree != expected_tree:
            raise BackendVerificationError(
                "The bundled Git-less SirenFNO tree does not match the manifest."
            )
        expected_neuralop_tree = manifest["bundled_neuraloperator"][
            "git_tree_sha1"
        ]
        if neuralop_tree != expected_neuralop_tree:
            raise BackendVerificationError(
                "The bundled Git-less NeuralOperator tree does not match the manifest."
            )
        for identity in manifest["source_files"].values():
            relative = PurePosixPath(identity["path"])
            source = REPOSITORY_ROOT.joinpath(*relative.parts)
            try:
                blob = _git_object_sha1("blob", source.read_bytes())
            except OSError as exc:
                raise BackendVerificationError(
                    "A pinned upstream source file could not be inspected."
                ) from exc
            if blob != identity.get("git_blob_sha1"):
                raise BackendVerificationError(
                    f"Pinned canonical Git blob mismatch: {identity['path']}"
                )
        bundled = release.get("bundled_upstream")
        if not isinstance(bundled, dict) or bundled.get("commit") != manifest["commit"]:
            raise BackendVerificationError(
                "Release provenance does not identify the pinned SirenFNO commit."
            )
        return {
            "repository": manifest["repository"],
            "commit": manifest["commit"],
            "license": manifest["license"],
            "neuraloperator_version": manifest["bundled_neuraloperator"][
                "declared_version"
            ],
            "neuraloperator_git_tree_sha1": neuralop_tree,
            "neuraloperator_source": manifest["bundled_neuraloperator"]["path"],
            "verification": "release_manifest_and_git_tree",
        }

    actual_commit = _git(["rev-parse", "HEAD"], SIRENFNO_ROOT)
    if actual_commit != manifest["commit"]:
        raise BackendVerificationError(
            "third_party/SirenFNO is at the wrong commit: "
            f"expected {manifest['commit']}, got {actual_commit}."
        )

    actual_tree = _git(["rev-parse", "HEAD:neuralop"], SIRENFNO_ROOT)
    expected_tree = manifest["bundled_neuraloperator"]["git_tree_sha1"]
    if actual_tree != expected_tree:
        raise BackendVerificationError(
            "The bundled NeuralOperator Git tree does not match the manifest."
        )

    remote = _git(["remote", "get-url", "origin"], SIRENFNO_ROOT)
    normalized_remote = remote.removesuffix("/").removesuffix(".git")
    normalized_expected = manifest["repository"].removesuffix("/").removesuffix(".git")
    if normalized_remote != normalized_expected:
        raise BackendVerificationError(
            "third_party/SirenFNO origin is not the public manifest repository."
        )

    submodule_prefix = Path("third_party") / "SirenFNO"
    for identity in manifest["source_files"].values():
        try:
            relative = Path(identity["path"]).relative_to(submodule_prefix).as_posix()
        except (KeyError, ValueError) as exc:
            raise BackendVerificationError(
                "A pinned source identity is outside third_party/SirenFNO."
            ) from exc
        actual_blob = _git(["rev-parse", f"HEAD:{relative}"], SIRENFNO_ROOT)
        if actual_blob != identity.get("git_blob_sha1"):
            raise BackendVerificationError(
                f"Pinned canonical Git blob mismatch: {identity['path']}"
            )

    verify_git_checkout_cleanliness(SIRENFNO_ROOT)

    return {
        "repository": manifest["repository"],
        "commit": actual_commit,
        "license": manifest["license"],
        "neuraloperator_version": manifest["bundled_neuraloperator"][
            "declared_version"
        ],
        "neuraloperator_git_tree_sha1": actual_tree,
        "neuraloperator_source": manifest["bundled_neuraloperator"]["path"],
    }


def verify_pinned_source_blobs(
    expected_blobs: Mapping[str, str],
) -> dict[str, str]:
    """Verify experiment-specific files against the pinned upstream commit.

    A normal checkout compares committed blob identities after the shared
    verifier enforces the pinned commit/tree and checkout-cleanliness policy.
    A Git-less release first verifies its complete release manifest and bundled
    upstream trees, then derives each blob identity from the actual archived
    bytes.  This keeps the existing EOL-only developer-checkout allowance while
    making the release check depend on the files that were really bundled.
    """

    verify_sirenfno_checkout()
    git_checkout = (SIRENFNO_ROOT / ".git").exists()
    sha1_pattern = re.compile(r"[0-9a-f]{40}\Z")
    actual_blobs: dict[str, str] = {}

    for relative, expected_blob in expected_blobs.items():
        if not isinstance(relative, str) or not isinstance(expected_blob, str):
            raise BackendVerificationError("A pinned source identity is invalid.")
        pure_path = PurePosixPath(relative)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or ".." in pure_path.parts
            or "\\" in relative
            or any(part.casefold() == ".git" for part in pure_path.parts)
            or not sha1_pattern.fullmatch(expected_blob)
        ):
            raise BackendVerificationError("A pinned source identity is invalid.")

        source = SIRENFNO_ROOT.joinpath(*pure_path.parts).resolve()
        if not _is_below(source, SIRENFNO_ROOT) or not source.is_file():
            raise BackendVerificationError(
                f"Pinned upstream source file is missing: {relative}"
            )
        if git_checkout:
            actual_blob = _git(["rev-parse", f"HEAD:{relative}"], SIRENFNO_ROOT)
        else:
            try:
                actual_blob = _git_object_sha1("blob", source.read_bytes())
            except OSError as exc:
                raise BackendVerificationError(
                    f"Pinned upstream source file could not be inspected: {relative}"
                ) from exc
        if actual_blob != expected_blob:
            raise BackendVerificationError(
                f"Pinned canonical Git blob mismatch: {relative}"
            )
        actual_blobs[relative] = actual_blob

    return actual_blobs


def bootstrap_sirenfno_backend() -> dict[str, Any]:
    """Put the verified official checkout first without reloading modules."""

    verified = verify_sirenfno_checkout()
    entry = str(SIRENFNO_ROOT)
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

    preloaded = sys.modules.get("neuralop")
    if preloaded is not None:
        source = _module_source(preloaded, "preloaded neuralop")
        if not _is_below(source, NEURALOP_ROOT):
            raise BackendVerificationError(
                "neuralop was preloaded outside third_party/SirenFNO/neuralop. "
                "Start a fresh Python process; implementations will not be mixed."
            )
    return verified


def audit_runtime_sources(
    official_components: Mapping[str, Any],
    proposed_components: Mapping[str, Any],
) -> dict[str, str]:
    """Validate implementation objects and return sanitized source paths."""

    audited: dict[str, str] = {}
    for label, obj in official_components.items():
        source = _module_source(obj, label)
        if not _is_below(source, SIRENFNO_ROOT):
            raise BackendVerificationError(
                f"{label} was not imported from third_party/SirenFNO."
            )
        audited[label] = public_source_path(source)

    proposed_root = (REPOSITORY_ROOT / "models").resolve()
    for label, obj in proposed_components.items():
        source = _module_source(obj, label)
        if not _is_below(source, proposed_root):
            raise BackendVerificationError(f"{label} was not imported from models/.")
        audited[label] = public_source_path(source)
    return audited


def verify_release_provenance(
    repository_root: Path,
    *,
    allow_dirty: bool,
) -> dict[str, Any]:
    """Verify a Git-less release and every immutable archived source file."""

    root = repository_root.resolve()
    provenance_path = root / RELEASE_PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise BackendVerificationError(
            "No superproject Git metadata or RELEASE_PROVENANCE.json was found."
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendVerificationError("RELEASE_PROVENANCE.json is invalid.") from exc

    sha1_pattern = re.compile(r"[0-9a-f]{40}\Z")
    sha256_pattern = re.compile(r"[0-9a-f]{64}\Z")
    commit = provenance.get("source_repository_commit")
    source_tree = provenance.get("source_tree_sha1")
    files = provenance.get("files")
    bundled_upstream = provenance.get("bundled_upstream")
    expected_upstream = upstream_manifest()["SirenFNO"]
    expected_upstream_identity = {
        "path": "third_party/SirenFNO",
        "repository": expected_upstream["repository"],
        "commit": expected_upstream["commit"],
        "git_tree_sha1": expected_upstream.get("git_tree_sha1"),
    }
    if (
        provenance.get("release_format_version") != RELEASE_FORMAT_VERSION
        or not isinstance(commit, str)
        or not sha1_pattern.fullmatch(commit)
        or not isinstance(source_tree, str)
        or not sha1_pattern.fullmatch(source_tree)
        or provenance.get("source_repository_dirty") is not False
        or provenance.get("hash_scope") != RELEASE_HASH_SCOPE
        or bundled_upstream != expected_upstream_identity
        or not isinstance(files, dict)
        or not files
    ):
        raise BackendVerificationError("RELEASE_PROVENANCE.json has an invalid schema.")

    dirty = False
    recorded_paths: set[str] = set()
    for relative, expected_sha256 in files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise BackendVerificationError(
                "RELEASE_PROVENANCE.json contains an invalid file identity."
            )
        pure_path = PurePosixPath(relative)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or ".." in pure_path.parts
            or "\\" in relative
            or any(part.casefold() == ".git" for part in pure_path.parts)
            or relative == RELEASE_PROVENANCE_FILENAME
            or not sha256_pattern.fullmatch(expected_sha256)
        ):
            raise BackendVerificationError(
                "RELEASE_PROVENANCE.json contains an unsafe file identity."
            )
        recorded_paths.add(relative)
        source = (root / Path(*pure_path.parts)).resolve()
        if not _is_below(source, root) or not source.is_file():
            dirty = True
            continue
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            dirty = True

    upstream_prefix = "third_party/SirenFNO/"
    if not any(relative.startswith(upstream_prefix) for relative in recorded_paths):
        raise BackendVerificationError(
            "RELEASE_PROVENANCE.json does not cover the bundled SirenFNO source."
        )

    def allowed_runtime_addition(relative: str) -> bool:
        parts = PurePosixPath(relative).parts
        return bool(parts) and (
            parts[0] in {"data", "results"}
            or any(
                part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                or part.endswith(".egg-info")
                for part in parts
            )
        )

    try:
        for source in root.rglob("*"):
            relative = source.relative_to(root).as_posix()
            if any(
                part.casefold() == ".git" for part in PurePosixPath(relative).parts
            ):
                dirty = True
                if source.is_dir() and not source.is_symlink():
                    continue
            if source.is_dir() and not source.is_symlink():
                continue
            if relative == RELEASE_PROVENANCE_FILENAME or relative in recorded_paths:
                continue
            if not allowed_runtime_addition(relative):
                dirty = True
    except OSError:
        dirty = True

    upstream_root = root / "third_party" / "SirenFNO"
    try:
        if git_tree_sha1(upstream_root) != expected_upstream_identity["git_tree_sha1"]:
            dirty = True
        if (
            git_tree_sha1(upstream_root / "neuralop")
            != expected_upstream["bundled_neuraloperator"]["git_tree_sha1"]
        ):
            dirty = True
    except BackendVerificationError:
        dirty = True

    if dirty and not allow_dirty:
        raise BackendVerificationError(
            "The anonymous release source differs from RELEASE_PROVENANCE.json. "
            "Pass --allow-dirty-source only for an explicit diagnostic run."
        )
    return {
        "source_repository_commit": commit,
        "source_repository_dirty": dirty,
        "bundled_upstream": bundled_upstream,
    }


def verify_git_repository_state(
    repository_root: Path,
    *,
    allow_dirty: bool,
) -> dict[str, Any]:
    """Verify the commit and public source state of a superproject checkout."""

    commit = _git(["rev-parse", "HEAD"], repository_root)
    status = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=dirty",
        ],
        repository_root,
    )
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise BackendVerificationError(
            "The source repository is dirty. Commit the audited snapshot or pass "
            "--allow-dirty-source for an explicitly non-release diagnostic run."
        )
    return {
        "source_repository_commit": commit,
        "source_repository_dirty": dirty,
    }


def source_repository_state(*, allow_dirty: bool) -> dict[str, Any]:
    """Verify either a Git checkout or a Git-less anonymous release snapshot."""

    if (REPOSITORY_ROOT / ".git").exists():
        return verify_git_repository_state(
            REPOSITORY_ROOT,
            allow_dirty=allow_dirty,
        )
    return verify_release_provenance(REPOSITORY_ROOT, allow_dirty=allow_dirty)


__all__ = [
    "BackendVerificationError",
    "MANIFEST_PATH",
    "NEURALOP_ROOT",
    "REPOSITORY_ROOT",
    "RELEASE_FORMAT_VERSION",
    "RELEASE_HASH_SCOPE",
    "RELEASE_PROVENANCE_FILENAME",
    "SIRENFNO_ROOT",
    "audit_runtime_sources",
    "bootstrap_sirenfno_backend",
    "git_tree_sha1",
    "public_source_path",
    "source_repository_state",
    "upstream_manifest",
    "verify_git_checkout_cleanliness",
    "verify_git_repository_state",
    "verify_pinned_source_blobs",
    "verify_release_provenance",
    "verify_sirenfno_checkout",
]
