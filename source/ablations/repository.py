"""Repository selection, import auditing, and source provenance checks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import configparser
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .public import PublicError
from .contracts import DATASETS
from .runtime import absolute, contained, workspace_root

PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_MANIFEST_NAME = "source_manifest_baseline_c.json"
_SOURCE_COMPATIBILITY_NAME = "source_compatibility_anonymity_release_v1.json"
_EXPECTED_SOURCE_COMPATIBILITY = {
    "schema_version": 1,
    "compatibility_id": "anonymity-audit-release-v1",
    "baseline_manifest": _SOURCE_MANIFEST_NAME,
    "baseline_manifest_sha256": "647cbaaf1132c8bc0a32936081d33b6c266e89c85e0c4801ec4e818d2e9eb47a",
    "base_source_commit": "5d1ec68dfe2774b74eec45f402b6f7f028a47b3e",
    "normalization": "CRLF to LF; all other bytes preserved",
    "provenance_claim": (
        "The baseline remains the training source. These exact deployment-only "
        "changes are not claimed as training source."
    ),
    "approved_changes": [
        {
            "path": "scripts/audit_anonymity.py",
            "baseline_sha256": "56cc2b92ac67228a0ef9b362ce42f73d75e251aaffa6562c168c6fde207b6769",
            "deployment_sha256": "ddf92ec7d735b22a88ef26d01e22cab45623cffeb6995c4ed4ff4c36b2280dc2",
            "scope": "anonymous release false-positive handling only",
            "training_or_evaluation_effect": False,
        },
        {
            "path": "tests/test_release.py",
            "baseline_sha256": "0b9197314f553b2ae17a63b366f6f9e35dfbfe27ab97db3e7b629b91c53afc2c",
            "deployment_sha256": "0010c3501ff24e7a70eaddf6fbfbc22e105ebda3e258eea9004fa7dc07fb488f",
            "scope": "anonymous release regression coverage only",
            "training_or_evaluation_effect": False,
        },
    ],
}
_BASELINE_DATASETS = tuple(DATASETS)
REQUIRED_PATHS = (
    Path("models/Cafe_Plus_FNO1D.py"),
    Path("models/Cafe_Plus_FNO2D.py"),
    Path("experiments/configs/seeds.py"),
    Path("experiments/common/checkpoints.py"),
)
AUDITED_PREFIXES = (
    "models", "experiments", "scripts", "baseline", "neuralop", "utils",
    "SirenFNO1D", "SirenFNO2D", "train_Burgers", "train_CFD", "train_CFD2D",
    "train_ReacDiff", "train_Darcy128", "train_NS", "ablations",
)


def inspect_repository_paths(root: Path) -> None:
    """Reject filesystem and Git indirection before source code is imported."""
    root = contained(root, workspace_root())
    pending = [root]
    while pending:
        current = pending.pop()
        for child in current.iterdir():
            contained(child, root)
            if child.is_dir():
                # Dataset/result contents are not needed to check source imports.
                if child.name not in {"data", "results", "checkpoints", "logs", "tmp", "__pycache__"}:
                    pending.append(child)
            elif child.stat().st_nlink > 1:
                raise RuntimeError("SOURCE_SHARED_HARDLINK: independent regular files are required")
    for checkout in (root, root / "third_party" / "SirenFNO"):
        control = checkout / ".git"
        if not control.exists():
            continue
        if control.is_file():
            text = control.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir: "):
                raise RuntimeError("SOURCE_GIT_CONTROL_INVALID")
            control = contained(checkout / text[8:], root)
        for name in ("commondir", "objects/info/alternates", "objects/info/http-alternates"):
            link = control / name
            if link.exists():
                raise RuntimeError("SOURCE_SHARED_GIT_METADATA: an independent checkout is required")
        config = configparser.ConfigParser(interpolation=None)
        config.read(control / "config", encoding="utf-8")
        if any(section.lower().startswith("include") for section in config.sections()):
            raise RuntimeError("Git configuration includes are not allowed in a reference copy")
        if config.has_option("core", "worktree"):
            contained(control / config.get("core", "worktree"), root)


class RepositoryImportGuard(importlib.abc.MetaPathFinder):
    """Resolve project modules only in their selected source roots, before execution."""
    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        prefix = fullname.split(".")[0]
        if prefix not in AUDITED_PREFIXES:
            return None
        owner = PACKAGE_ROOT if prefix == "ablations" else self.root
        if path is None:
            if prefix == "ablations":
                locations = [str(PACKAGE_ROOT.parent)]
            elif prefix in {"models", "experiments", "scripts"}:
                locations = [str(self.root)]
            else:
                locations = [str(self.root / "third_party" / "SirenFNO")]
        else:
            locations = [str(contained(item, owner)) for item in path]
        spec = importlib.machinery.PathFinder.find_spec(fullname, locations)
        if spec is None:
            raise ImportError(f"Selected repository has no module {fullname}; external fallback blocked")
        if spec.origin:
            contained(spec.origin, owner)
        for location in spec.submodule_search_locations or ():
            contained(location, owner)
        return spec


def _below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def baseline_profile() -> str:
    """Identify the sole approved source; reject stale selection settings."""

    if os.environ.get("CAFE_ABLATION_BASELINE_PROFILE", "C") != "C":
        raise PublicError("BASELINE_PROFILE_INVALID")
    return "C"


def baseline_datasets() -> tuple[str, ...]:
    """Return the seven supported datasets on the single approved source."""

    baseline_profile()
    return _BASELINE_DATASETS


def require_dataset_baseline(dataset: str) -> None:
    """Reject a dataset outside the selected profile before parent imports."""

    if dataset not in baseline_datasets():
        raise PublicError("DATASET_BASELINE_NOT_APPROVED", dataset=dataset)


def source_manifest_path() -> Path:
    """Use the reviewed all-dataset manifest, independently of HEAD/root."""

    baseline_profile()
    return PACKAGE_ROOT / _SOURCE_MANIFEST_NAME


def load_source_manifest() -> dict:
    return json.loads(source_manifest_path().read_text(encoding="utf-8"))


def source_compatibility_path() -> Path:
    """Return the immutable deployment-only compatibility record."""

    baseline_profile()
    return PACKAGE_ROOT / _SOURCE_COMPATIBILITY_NAME


def _canonical_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_source_compatibility(record: object, manifest: dict) -> dict:
    """Accept only the reviewed two-file deployment transition.

    The baseline manifest remains authoritative.  Keeping the complete expected
    record here means editing the JSON alone cannot authorize a new path or hash.
    """

    if record != _EXPECTED_SOURCE_COMPATIBILITY:
        raise RuntimeError("SOURCE_COMPATIBILITY_RECORD_INVALID")
    if manifest.get("base_source_commit") != record["base_source_commit"]:
        raise RuntimeError("SOURCE_COMPATIBILITY_BASE_COMMIT_MISMATCH")
    if _canonical_source_sha256(source_manifest_path()) != record["baseline_manifest_sha256"]:
        raise RuntimeError("SOURCE_COMPATIBILITY_BASE_MANIFEST_MISMATCH")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("SOURCE_COMPATIBILITY_BASE_MANIFEST_INVALID")
    for change in record["approved_changes"]:
        if files.get(change["path"]) != change["baseline_sha256"]:
            raise RuntimeError("SOURCE_COMPATIBILITY_BASE_HASH_MISMATCH")
        if change["training_or_evaluation_effect"] is not False:
            raise RuntimeError("SOURCE_COMPATIBILITY_SCOPE_INVALID")
    return record


def load_source_compatibility() -> dict:
    try:
        record = json.loads(source_compatibility_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("SOURCE_COMPATIBILITY_RECORD_INVALID") from None
    return validate_source_compatibility(record, load_source_manifest())


def resolve_repository_root(explicit: str | Path | None = None) -> Path:
    """Resolve an explicit test repository or the parent of integrated ablations."""

    candidate = Path(explicit or os.environ.get("CAFE_ABLATION_REPOSITORY_ROOT", PACKAGE_ROOT.parent))
    candidate = contained(candidate, workspace_root())
    inspect_repository_paths(candidate)
    missing = [str(item) for item in REQUIRED_PATHS if not (candidate / item).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SOURCE_REQUIRED_FILES_MISSING: {', '.join(missing)}"
        )
    return candidate


def activate_repository(root: Path) -> None:
    """Put exactly one selected repository first on sys.path, or fail closed."""

    root = root.resolve()
    import_audit(root)
    # Reject already installed editable mappings before resolving any project module.
    for finder in sys.meta_path:
        module = sys.modules.get(getattr(finder, "__module__", ""))
        mapping = getattr(module, "MAPPING", {})
        for name, location in mapping.items():
            if name.split(".")[0] in AUDITED_PREFIXES:
                owner = PACKAGE_ROOT if name.split(".")[0] == "ablations" else root
                if not absolute(location).is_relative_to(owner):
                    raise RuntimeError("External project editable finder detected; use an independent environment")
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in AUDITED_PREFIXES:
            continue
        source = getattr(module, "__file__", None)
        owner = PACKAGE_ROOT if name.split(".", 1)[0] == "ablations" else root
        if source and not _below(Path(source), owner):
            raise RuntimeError(
                f"Module {name!r} was already imported from outside the selected "
                "repository (external location withheld)"
            )
    root_text = str(root)
    sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != root]
    sys.path.insert(0, root_text)
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, RepositoryImportGuard)]
    sys.meta_path.insert(0, RepositoryImportGuard(root))
    importlib.invalidate_caches()


def import_audit(root: Path, modules: Iterable[ModuleType] | None = None) -> dict[str, str]:
    """Return source paths and reject repository-owned imports from elsewhere."""

    root = root.resolve()
    selected = modules if modules is not None else tuple(sys.modules.values())
    audit: dict[str, str] = {}
    for module in selected:
        name = getattr(module, "__name__", "")
        if name.split(".", 1)[0] not in AUDITED_PREFIXES:
            continue
        source = getattr(module, "__file__", None)
        owner = PACKAGE_ROOT if name.split(".", 1)[0] == "ablations" else root
        locations = [source] if source else list(getattr(module, "__path__", ()))
        for location in locations:
            resolved = contained(location, owner)
            audit[name] = ("ablations/" if owner == PACKAGE_ROOT else "") + resolved.relative_to(owner).as_posix()
    return dict(sorted(audit.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_base_sources(root: Path) -> tuple[dict[str, str], dict]:
    """Verify the baseline plus one exact, atomic deployment-only transition."""

    manifest = load_source_manifest()
    expected = manifest["files"]
    compatibility = load_source_compatibility()
    approved = {change["path"]: change for change in compatibility["approved_changes"]}
    mismatches: list[str] = []
    applied: list[dict] = []
    for relative, expected_hash in expected.items():
        path = contained(root / Path(relative), root)
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        digest = _canonical_source_sha256(path)
        if digest == expected_hash:
            continue
        change = approved.get(relative)
        if (change is not None and change["baseline_sha256"] == expected_hash
                and change["deployment_sha256"] == digest):
            applied.append(change)
        else:
            mismatches.append(f"hash:{relative}:{expected_hash}:{digest}")
    applied_paths = {change["path"] for change in applied}
    approved_paths = set(approved)
    if applied_paths and applied_paths != approved_paths:
        missing = ",".join(sorted(approved_paths - applied_paths))
        mismatches.append(f"compatibility-partial:{missing}")
    if mismatches:
        raise RuntimeError("Base source verification failed: " + "; ".join(mismatches))
    state = {
        "compatibility_id": compatibility["compatibility_id"],
        "record": source_compatibility_path().name,
        "record_sha256": sha256_file(source_compatibility_path()),
        "baseline_manifest": compatibility["baseline_manifest"],
        "baseline_manifest_sha256": compatibility["baseline_manifest_sha256"],
        "status": "APPLIED" if applied else "BASELINE",
        "approved_changes": compatibility["approved_changes"],
        "applied_paths": sorted(applied_paths),
        "provenance_claim": compatibility["provenance_claim"],
        "training_source_equivalence_claim": False,
    }
    # Return the unchanged baseline hashes separately from the deployment state.
    return dict(expected), state


def verify_base_sources(root: Path) -> dict[str, str]:
    """Verify stable source/config files without rewriting their baseline record."""

    expected, _ = _verify_base_sources(root)
    return expected


def ablation_source_files(package_root: Path | None = None) -> dict[str, Path]:
    """Enumerate immutable add-on files, rejecting links before reading them.

    Results belong outside the package. Only Python bytecode caches may be
    added to an installed package without changing its source identity.
    """

    package_root = contained(package_root or PACKAGE_ROOT, workspace_root())
    found: dict[str, Path] = {}
    pending = [package_root]
    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir()):
            child = contained(child, package_root)
            if child.name == "__pycache__":
                continue
            if child.is_dir():
                if child.name.casefold() == ".git":
                    raise RuntimeError("SOURCE_NESTED_GIT_IN_ABLATIONS")
                pending.append(child)
            elif child.is_file():
                if child.suffix.lower() in {".pyc", ".pyo"}:
                    continue
                if child.stat().st_nlink > 1:
                    raise RuntimeError("SOURCE_SHARED_HARDLINK")
                found[child.relative_to(package_root).as_posix()] = child
            else:
                raise RuntimeError("SOURCE_UNSUPPORTED_FILE_TYPE")
    if not found:
        raise RuntimeError("SOURCE_ABLATIONS_EMPTY")
    return dict(sorted(found.items()))


def ablation_tree_sha256(package_root: Path | None = None) -> str:
    """Hash distributable ablation sources without caches or generated outputs."""

    digest = hashlib.sha256()
    for name, path in ablation_source_files(package_root).items():
        relative = name.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0")
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root, text=True, encoding="utf-8", env=env,
            errors="replace", capture_output=True, check=False,
        )
    except OSError:
        raise RuntimeError("SOURCE_GIT_UNAVAILABLE") from None
    if completed.returncode:
        raise RuntimeError("SOURCE_GIT_VERIFICATION_FAILED")
    return completed.stdout.strip()


def _parent_backend(root: Path):
    """Use the selected, already hash-verified parent's unchanged verifier."""

    activate_repository(root)
    backend = importlib.import_module("experiments.common.sirenfno_backend")
    if contained(backend.__file__, root) != root / "experiments/common/sirenfno_backend.py":
        raise RuntimeError("SOURCE_BACKEND_LOCATION_MISMATCH")
    return backend


def _release_state(root: Path, files: dict[str, Path], base_files: dict[str, str]) -> dict:
    """Verify the parent's release format plus complete integrated add-on coverage.

    A release declares a source commit; without Git objects it supplies no
    independently checked ancestry evidence. The hash check is explicitly
    recorded as such, never as a Git ancestry check or training evidence.
    """

    provenance_path = contained(root / "RELEASE_PROVENANCE.json", root)
    if not provenance_path.is_file():
        raise RuntimeError("SOURCE_RELEASE_MANIFEST_MISSING")
    try:
        backend = _parent_backend(root)
        verified = backend.verify_release_provenance(root, allow_dirty=False)
        manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
        recorded = manifest["files"]
        expected_paths = set(base_files) | {f"ablations/{name}" for name in files}
        if not expected_paths.issubset(recorded):
            raise RuntimeError("SOURCE_RELEASE_COVERAGE_INCOMPLETE")
        # The parent verifier hashes every recorded file. Repeat just add-on
        # bytes here so coverage and content are inseparable in our contract.
        if any(recorded[f"ablations/{name}"] != sha256_file(path) for name, path in files.items()):
            raise RuntimeError("SOURCE_RELEASE_ABLATIONS_HASH_MISMATCH")
        if verified.get("source_repository_dirty") is not False:
            raise RuntimeError("SOURCE_RELEASE_DIRTY")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise RuntimeError("SOURCE_RELEASE_VERIFICATION_FAILED") from None
    return {
        "source_verification_method": "release_manifest_hashes",
        "actual_ablation_source_commit": verified["source_repository_commit"],
        "source_commit_evidence": "declared_by_verified_release_manifest",
        "source_repository_dirty": verified["source_repository_dirty"],
        "git_ancestry_verified": False,
        "git_ancestry_status": "not_available_without_git_objects",
        "release_provenance_sha256": sha256_file(provenance_path),
        "release_source_tree_sha1": manifest["source_tree_sha1"],
        "release_hash_scope": manifest["hash_scope"],
        "release_verified_file_count": len(recorded),
    }


def source_state(root: Path, *, formal: bool) -> dict:
    """Separate the captured base source from the actual ablation execution source."""

    root = contained(root, workspace_root())
    inspect_repository_paths(root)
    manifest = load_source_manifest()
    verified, compatibility = _verify_base_sources(root)
    files = ablation_source_files()
    file_hashes = {name: sha256_file(path) for name, path in files.items()}
    package_integrated = _below(PACKAGE_ROOT, root) and PACKAGE_ROOT.parent == root
    state = {
        "baseline_profile": baseline_profile(),
        "allowed_datasets": list(baseline_datasets()),
        "base_source_commit": manifest["base_source_commit"],
        "base_source_verification": manifest["verification"],
        "source_manifest": source_manifest_path().name,
        "base_source_sha256": verified,
        "deployment_source_compatibility": compatibility,
        "sirenfno_commit": manifest["sirenfno_commit"],
        "ablation_source_sha256": ablation_tree_sha256(),
        "ablation_source_files_sha256": file_hashes,
        "ablation_file_manifest_sha256": hashlib.sha256(
            json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "ablation_package_integrated": package_integrated,
        "source_mode": "formal" if formal else "diagnostic",
    }
    if not formal:
        state["actual_ablation_source_commit"] = "UNCOMMITTED_DIAGNOSTIC"
        state["source_verification_method"] = "diagnostic"
        state["source_repository_dirty"] = None
        state["git_ancestry_verified"] = False
        state["git_ancestry_status"] = "not_checked_diagnostic"
        return state

    if not package_integrated:
        raise RuntimeError(
            "Formal runs require this ablations directory to be copied directly "
            "under the selected repository root."
        )
    if not (root / ".git").exists():
        state.update(_release_state(root, files, verified))
        return state
    _git(root, "merge-base", "--is-ancestor", manifest["base_source_commit"], "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("Formal runs require a clean committed source tree.")
    tracked = set(_git(root, "ls-files", "--", "ablations").splitlines())
    distributable = {f"ablations/{name}" for name in files}
    missing = sorted(distributable - tracked)
    if missing:
        raise RuntimeError(
            "Formal runs require every ablation source to be committed: "
            + str(len(missing)) + " uncommitted file(s)"
        )
    for name, path in files.items():
        committed_blob = _git(root, "rev-parse", f"HEAD:ablations/{name}")
        payload = path.read_bytes()
        candidates = (payload, payload.replace(b"\r\n", b"\n"))
        actual_blobs = {
            hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            for data in candidates
        }
        if committed_blob not in actual_blobs:
            # A status-only check can miss assume-unchanged/skip-worktree files.
            raise RuntimeError("SOURCE_ABLATIONS_COMMIT_CONTENT_MISMATCH")
    try:
        _parent_backend(root).verify_sirenfno_checkout()
    except RuntimeError:
        raise RuntimeError("SOURCE_PINNED_UPSTREAM_VERIFICATION_FAILED") from None
    state["actual_ablation_source_commit"] = _git(root, "rev-parse", "HEAD")
    state["source_verification_method"] = "git_clean_checkout"
    state["source_commit_evidence"] = "verified_local_git_head"
    state["source_repository_dirty"] = bool(status)
    state["git_ancestry_verified"] = True
    state["git_ancestry_status"] = "base_is_ancestor_of_actual_ablation_commit"
    return state
