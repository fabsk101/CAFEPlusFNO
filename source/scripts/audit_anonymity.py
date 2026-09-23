"""Audit a public ref, release candidate, and optional ZIP for anonymity.

Public Git history and archive contents are release gates. Local Git control
files are reported separately because Git does not publish reflogs, worktree
pointers, or local configuration during an ordinary push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANONYMOUS_AUTHOR = "Anonymous Author"
ALLOWED_EMAILS = {"anonymous@users.noreply.github.com"}
NON_IDENTITY_EMAIL_DOMAINS = {
    "example.com",
    "example.invalid",
    "example.net",
    "example.org",
}
ALLOWED_GITHUB = {
    "https://github.com/pengqingshi/sirenfno",
    "https://github.com/pengqingshi/sirenfno.git",
    # Exact attribution embedded in the pinned NeuralOperator source tree.
    "https://github.com/lucidrains/x-transformers",
}
CONTENT_LABELS = (
    "tracked personal name",
    "tracked personal email",
    "institutional affiliation",
    "local absolute paths",
    "hostname / machine id",
    "personal GitHub references",
    "credentials / secrets",
)
PRIVATE_DENYLIST_ENV = {
    "names": "ANONYMITY_PRIVATE_NAMES",
    "emails": "ANONYMITY_PRIVATE_EMAILS",
    "affiliations": "ANONYMITY_PRIVATE_AFFILIATIONS",
    "githubs": "ANONYMITY_PRIVATE_GITHUBS",
}
MAX_INSPECTABLE_BYTES = 8 * 1024 * 1024
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[A-Z0-9._-]+[\\/]"
    r"|/(?:Users|home)/[A-Z0-9._-]+/|/mnt/"
    r"[a-z]/Users/[A-Z0-9._-]+/)"
)

# Narrow release-scan exceptions for reviewed privacy-test fixtures.  These are
# not account-name or directory allowlists: the relative file path, complete
# UTF-8 content SHA-256, ordered count-preserving SHA-256 values of every regex
# match, and (for history) complete Git blob ID must all agree.  Raw path-like
# fixture values remain solely in the tests that exercise privacy redaction.
_REVIEWED_SYNTHETIC_PATH_FIXTURES = {
    "ablations/tests/test_references.py": (
        "45cdeb8c2426a4210053376925ebfe43b3bc0d46d40c73769648ec1412cd34b5",
        ("5e1b2c78f6d15ff9d283df66abadbdbbf845ae1c93e996161efaef05d91886df",),
    ),
    "ablations/tests/test_runtime_privacy.py": (
        "187e1285a20a9cca44c90468d694ceb19c9d317dbae2e43c01754d4a4f51b2cb",
        (
            "7a9ae5cf8e42280034b308bd59b26488e4be2bc4c08d99999fb386619c661d38",
            "fa5c9a47f89ccbdbe50687e61723f8ac7a242b0cf16fb5f0c42c18b6357fcbcb",
            "58df4c2ca66934ac054975fada237ecf6bbf6965f615c68a3feca6356b8b093d",
        ),
    ),
    "ablations/tests/test_validation_accounting.py": (
        "c34acf72c177c5f8eb1cf9432b9c32d5e09f165c8de0e16238d69a39b0f0d605",
        (
            "64bc0512534216292935be3aed4ed64f9f86830f8e78846c3688cc4a55a66110",
            "64bc0512534216292935be3aed4ed64f9f86830f8e78846c3688cc4a55a66110",
        ),
    ),
}
_REVIEWED_SYNTHETIC_HISTORY_BLOBS = {
    "2781faa6b6e27f95c4816463c2cffe03be9704b3": (
        "ablations/tests/test_validation_accounting.py",
        "c34acf72c177c5f8eb1cf9432b9c32d5e09f165c8de0e16238d69a39b0f0d605",
    ),
    "73e7d97e62badbc031f6333d7d9a0c4c884e9f87": (
        "ablations/tests/test_runtime_privacy.py",
        "bcff95d83d965ed030106764ec343b217697672a396370bd2c2e24cb443adbd1",
    ),
    "b269cb10f93d1a7cb24b447164f56b15cd67b2f3": (
        "ablations/tests/test_references.py",
        "45cdeb8c2426a4210053376925ebfe43b3bc0d46d40c73769648ec1412cd34b5",
    ),
    "fa8fd2e7ce17a948259ce29f772ad28c73511185": (
        "ablations/tests/test_runtime_privacy.py",
        "187e1285a20a9cca44c90468d694ceb19c9d317dbae2e43c01754d4a4f51b2cb",
    ),
}
FORBIDDEN_ARCHIVE_DIRECTORY_NAMES = {
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "debug",
    "dist",
    "env",
    "logs",
    "tmp",
    "venv",
    "wandb",
}
FORBIDDEN_ARCHIVE_FILE_NAMES = {".ds_store", "thumbs.db"}
GENERATED_REVIEW_ARTIFACTS = {"CAFEPlusFNO_full_review.zip"}
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".bak",
    ".ckpt",
    ".hdf5",
    ".log",
    ".nsys-rep",
    ".onnx",
    ".orig",
    ".pem",
    ".prof",
    ".pt",
    ".pth",
    ".pfx",
    ".p12",
    ".rej",
    ".sqlite",
    ".swp",
    ".tmp",
    ".trace",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Public superproject ref whose reachable history is audited.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Also audit the bytes of this already-created release ZIP.",
    )
    return parser.parse_args()


def _run_git(
    args: list[str],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=text,
    )


def git_output(args: list[str], repository_root: Path = REPOSITORY_ROOT) -> str:
    completed = _run_git(args, repository_root=repository_root)
    if completed.returncode != 0:
        return ""
    return completed.stdout


def release_candidate_files(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[Path]:
    completed = _run_git(
        ["ls-files", "--cached", "--others", "--exclude-standard"],
        repository_root=repository_root,
    )
    if completed.returncode != 0:
        raise RuntimeError("Git could not enumerate the release candidate.")
    paths: list[Path] = []
    for relative in sorted(set(completed.stdout.splitlines())):
        path = repository_root / relative
        # The separately pinned upstream is inspected in the final archive.
        if (
            path.is_file()
            and relative not in GENERATED_REVIEW_ARTIFACTS
            and not relative.startswith("third_party/SirenFNO/")
        ):
            paths.append(path)
    return paths


def _decode_files(
    paths: Iterable[Path],
    *,
    label_root: Path,
) -> tuple[list[tuple[str, str]], set[str]]:
    decoded: list[tuple[str, str]] = []
    uninspectable: set[str] = set()
    for path in paths:
        try:
            relative = path.relative_to(label_root).as_posix()
            size = path.stat().st_size
            data = path.read_bytes() if size <= MAX_INSPECTABLE_BYTES else b""
        except (OSError, ValueError):
            uninspectable.add("candidate-file")
            continue
        if size > MAX_INSPECTABLE_BYTES or b"\x00" in data:
            uninspectable.add(relative)
            continue
        try:
            decoded.append((relative, data.decode("utf-8")))
        except UnicodeDecodeError:
            uninspectable.add(relative)
    return decoded, uninspectable


def text_files() -> list[tuple[str, str]]:
    """Compatibility helper returning inspectable release-candidate text."""

    decoded, _ = _decode_files(
        release_candidate_files(),
        label_root=REPOSITORY_ROOT,
    )
    return decoded


def files_matching(
    pattern: re.Pattern[str], files: list[tuple[str, str]]
) -> set[str]:
    return {relative for relative, text in files if pattern.search(text)}


def private_denylists_from_environment() -> dict[str, set[str]]:
    """Read optional private values from JSON arrays without logging values."""

    denylists: dict[str, set[str]] = {}
    for category, variable in PRIVATE_DENYLIST_ENV.items():
        raw = os.environ.get(variable, "").strip()
        if not raw:
            denylists[category] = set()
            continue
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{variable} must be a JSON array of strings.") from exc
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise RuntimeError(f"{variable} must be a JSON array of strings.")
        denylists[category] = {value.strip().casefold() for value in values}
    return denylists


def content_anonymity_findings(
    files: list[tuple[str, str]],
    *,
    private_denylists: dict[str, set[str]] | None = None,
    hostname: str | None = None,
) -> dict[str, set[str]]:
    """Return content findings; matched private values are never returned."""

    results: dict[str, set[str]] = {}
    denylists = (
        private_denylists_from_environment()
        if private_denylists is None
        else {
            category: {
                value.casefold() for value in private_denylists.get(category, set())
            }
            for category in PRIVATE_DENYLIST_ENV
        }
    )
    results["tracked personal name"] = {
        relative
        for relative, text in files
        if any(name in text.casefold() for name in denylists["names"])
    }

    email_re = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    results["tracked personal email"] = {
        relative
        for relative, text in files
        if any(
            match.casefold() not in ALLOWED_EMAILS
            and match.rpartition("@")[2].casefold() not in NON_IDENTITY_EMAIL_DOMAINS
            for match in email_re.findall(text)
        )
        or any(email in text.casefold() for email in denylists["emails"])
    }

    affiliation_re = re.compile(
        r"(?i)(?:@[A-Z0-9.-]+\.(?:edu|ac\.[A-Z]{2})\b|\baffiliation\s*:)"
    )
    results["institutional affiliation"] = files_matching(
        affiliation_re, files
    ) | {
        relative
        for relative, text in files
        if any(value in text.casefold() for value in denylists["affiliations"])
    }

    results["local absolute paths"] = files_matching(ABSOLUTE_PATH_RE, files)

    machine_id = (platform.node() if hostname is None else hostname).strip().casefold()
    results["hostname / machine id"] = {
        relative
        for relative, text in files
        if len(machine_id) >= 4 and machine_id in text.casefold()
    }

    github_re = re.compile(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
    )
    results["personal GitHub references"] = {
        relative
        for relative, text in files
        if any(
            url.casefold().rstrip("/.,)") not in ALLOWED_GITHUB
            for url in github_re.findall(text)
        )
        or any(value in text.casefold() for value in denylists["githubs"])
    }

    secret_re = re.compile(
        r"(?i)(?:-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
        r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
        r"|\bhf_[A-Za-z0-9]{20,}\b"
        r"|\bAKIA[0-9A-Z]{16}\b"
        r"|https?://[^\s/:]+:[^\s/@]+@[^\s/]+"
        r"|(?:api[_-]?key|wandb[_-]?api[_-]?key|secret[_-]?key|password)"
        r"\s*[:=]\s*['\"][^'\"]+['\"]"
        r")"
    )
    results["credentials / secrets"] = files_matching(secret_re, files)
    return results


def _reviewed_synthetic_absolute_path_fixture(label: str, text: str) -> bool:
    """Match one exact reviewed fixture without weakening the base detector."""

    specification = _REVIEWED_SYNTHETIC_PATH_FIXTURES.get(label)
    if label.startswith("history-blob:"):
        object_id = label.removeprefix("history-blob:")
        history_specification = _REVIEWED_SYNTHETIC_HISTORY_BLOBS.get(object_id)
        if history_specification is None:
            specification = None
        else:
            reviewed_path, content_hash = history_specification
            specification = (
                content_hash,
                _REVIEWED_SYNTHETIC_PATH_FIXTURES[reviewed_path][1],
            )
    if specification is None:
        return False
    expected_content, expected_matches = specification
    actual_matches = tuple(
        hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
        for match in ABSOLUTE_PATH_RE.finditer(text)
    )
    return (
        bool(actual_matches)
        and hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_content
        and actual_matches == expected_matches
    )


def release_content_anonymity_findings(
    files: list[tuple[str, str]],
    *,
    private_denylists: dict[str, set[str]] | None = None,
    hostname: str | None = None,
    applied_exemptions: set[str] | None = None,
) -> dict[str, set[str]]:
    """Apply exact fixture review only to release-scan absolute-path findings.

    Every other privacy category remains active.  Runtime privacy and low-level
    detector tests continue to call ``content_anonymity_findings`` directly and
    therefore still reject or redact these path-shaped synthetic inputs.
    """

    results = content_anonymity_findings(
        files,
        private_denylists=private_denylists,
        hostname=hostname,
    )
    for label, text in files:
        if (
            label in results["local absolute paths"]
            and _reviewed_synthetic_absolute_path_fixture(label, text)
        ):
            results["local absolute paths"].remove(label)
            if applied_exemptions is not None:
                applied_exemptions.add(label)
    return results


def _merge_findings(
    destination: dict[str, set[str]], source: dict[str, set[str]]
) -> None:
    for label, locations in source.items():
        destination.setdefault(label, set()).update(locations)


def _archive_artifact_name(member: str) -> bool:
    path = PurePosixPath(member)
    lowered_parts = tuple(part.casefold() for part in path.parts)
    if any(part in FORBIDDEN_ARCHIVE_DIRECTORY_NAMES for part in lowered_parts):
        return True
    if lowered_parts and lowered_parts[-1] in FORBIDDEN_ARCHIVE_FILE_NAMES:
        return True
    lowered = member.casefold()
    return lowered.endswith("~") or any(
        lowered.endswith(suffix) for suffix in FORBIDDEN_ARCHIVE_SUFFIXES
    )


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def archive_anonymity_findings(
    archive: Path,
    *,
    applied_exemptions: set[str] | None = None,
) -> dict[str, set[str]]:
    """Inspect every archive member without trusting its filename extension."""

    results: dict[str, set[str]] = {
        "archive Git metadata": set(),
        "archive local/generated artifacts": set(),
        "archive uninspectable files": set(),
    }
    decoded: list[tuple[str, str]] = []
    member_names: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                label = info.filename
                parts = tuple(part.casefold() for part in PurePosixPath(label).parts)
                if ".git" in parts:
                    results["archive Git metadata"].add(label)
                if _archive_artifact_name(label):
                    results["archive local/generated artifacts"].add(label)
                if info.is_dir():
                    continue
                member_names.append((f"member-name:{_stable_id(label)}", label))
                if info.flag_bits & 0x1 or info.file_size > MAX_INSPECTABLE_BYTES:
                    results["archive uninspectable files"].add(label)
                    continue
                try:
                    data = bundle.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    results["archive uninspectable files"].add(label)
                    continue
                if b"\x00" in data:
                    results["archive uninspectable files"].add(label)
                    continue
                try:
                    decoded.append((label, data.decode("utf-8")))
                except UnicodeDecodeError:
                    results["archive uninspectable files"].add(label)
    except (OSError, zipfile.BadZipFile):
        results["archive uninspectable files"].add("archive")
        return results

    _merge_findings(
        results,
        release_content_anonymity_findings(
            decoded + member_names,
            applied_exemptions=applied_exemptions,
        ),
    )
    return results


def _identity_is_anonymous(name: str, email: str) -> bool:
    return name == ANONYMOUS_AUTHOR and email.casefold() in ALLOWED_EMAILS


def _history_blob_text(
    repository_root: Path,
    ref: str,
) -> tuple[list[tuple[str, str]], set[str]]:
    listed = _run_git(
        ["rev-list", "--objects", "--no-object-names", ref],
        repository_root=repository_root,
    )
    if listed.returncode != 0:
        return [], {"history"}
    object_ids = list(dict.fromkeys(listed.stdout.splitlines()))
    decoded: list[tuple[str, str]] = []
    uninspectable: set[str] = set()
    try:
        process = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return [], {"history"}
    assert process.stdin is not None and process.stdout is not None
    try:
        for object_id in object_ids:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n").split()
            if len(header) != 3 or header[1] == b"missing":
                uninspectable.add(f"object:{object_id[:12]}")
                continue
            try:
                size = int(header[2])
            except ValueError:
                uninspectable.add(f"object:{object_id[:12]}")
                continue
            data = process.stdout.read(size)
            terminator = process.stdout.read(1)
            if len(data) != size or terminator != b"\n":
                uninspectable.add(f"object:{object_id[:12]}")
                break
            if header[1] != b"blob":
                continue
            # Preserve the complete object ID internally so a reviewed history
            # exception cannot collide on the public 12-character locator.
            label = f"history-blob:{object_id}"
            if size > MAX_INSPECTABLE_BYTES or b"\x00" in data:
                uninspectable.add(label)
                continue
            try:
                decoded.append((label, data.decode("utf-8")))
            except UnicodeDecodeError:
                uninspectable.add(label)
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        process.stdout.close()
    return decoded, uninspectable


def git_history_anonymity_findings(
    repository_root: Path = REPOSITORY_ROOT,
    ref: str = "HEAD",
    *,
    applied_exemptions: set[str] | None = None,
) -> dict[str, set[str]]:
    """Audit identities, messages, tags, and all blobs reachable from ``ref``."""

    results: dict[str, set[str]] = {
        "Git commit author metadata": set(),
        "Git commit committer metadata": set(),
        "Git commit messages": set(),
        "Git annotated tag metadata": set(),
        "Git reachable history contents": set(),
        "Git history uninspectable objects": set(),
    }
    log = _run_git(
        [
            "log",
            ref,
            "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e",
        ],
        repository_root=repository_root,
    )
    if log.returncode != 0 or not log.stdout.strip():
        results["Git history uninspectable objects"].add("history")
        return results
    message_inputs: list[tuple[str, str]] = []
    for record in log.stdout.split("\x1e"):
        record = record.strip("\r\n")
        if not record:
            continue
        fields = record.split("\x1f", 5)
        if len(fields) != 6:
            results["Git history uninspectable objects"].add("commit-record")
            continue
        commit, author, author_email, committer, committer_email, message = fields
        location = f"commit:{commit[:12]}"
        if not _identity_is_anonymous(author, author_email):
            results["Git commit author metadata"].add(location)
        if not _identity_is_anonymous(committer, committer_email):
            results["Git commit committer metadata"].add(location)
        message_inputs.append((location, message))
    message_results = content_anonymity_findings(message_inputs)
    results["Git commit messages"].update(
        location for locations in message_results.values() for location in locations
    )

    tags = _run_git(
        ["for-each-ref", "--format=%(refname:short)", "refs/tags"],
        repository_root=repository_root,
    )
    if tags.returncode != 0:
        results["Git history uninspectable objects"].add("tags")
    else:
        for tag in tags.stdout.splitlines():
            reachable = _run_git(
                ["merge-base", "--is-ancestor", f"{tag}^{{}}", ref],
                repository_root=repository_root,
            )
            if reachable.returncode == 1:
                continue
            if reachable.returncode != 0:
                results["Git history uninspectable objects"].add("tag")
                continue
            object_type = git_output(
                ["cat-file", "-t", tag], repository_root
            ).strip()
            if object_type != "tag":
                continue
            raw = _run_git(
                ["cat-file", "tag", tag],
                repository_root=repository_root,
            )
            tag_label = f"tag:{_stable_id(tag)}"
            if raw.returncode != 0:
                results["Git history uninspectable objects"].add(tag_label)
                continue
            tagger = next(
                (line for line in raw.stdout.splitlines() if line.startswith("tagger ")),
                "",
            )
            match = re.match(r"tagger (.*) <([^>]*)> [0-9]+ [+-][0-9]{4}$", tagger)
            if match is None or not _identity_is_anonymous(match.group(1), match.group(2)):
                results["Git annotated tag metadata"].add(tag_label)
            tag_results = content_anonymity_findings([(tag_label, raw.stdout)])
            if any(tag_results.values()):
                results["Git annotated tag metadata"].add(tag_label)

    history_text, uninspectable = _history_blob_text(repository_root, ref)
    history_results = release_content_anonymity_findings(
        history_text,
        applied_exemptions=applied_exemptions,
    )
    results["Git reachable history contents"].update(
        location for locations in history_results.values() for location in locations
    )
    results["Git history uninspectable objects"].update(uninspectable)
    return results


def local_git_metadata_findings(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, set[str]]:
    """Inspect non-published Git control files for a separate local warning."""

    results: dict[str, set[str]] = {"local Git metadata uninspectable": set()}
    git_dir_result = _run_git(
        ["rev-parse", "--absolute-git-dir"], repository_root=repository_root
    )
    if git_dir_result.returncode != 0:
        results["local Git metadata uninspectable"].add("git-dir")
        return results
    git_dir = Path(git_dir_result.stdout.strip())
    candidates: list[Path] = []
    for path in git_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(git_dir).as_posix()
        parts = PurePosixPath(relative).parts
        if (
            relative == "config"
            or parts[:1] in {("logs",), ("worktrees",)}
            or (parts[:1] == ("modules",) and parts[-1] == "config")
            or (parts[:1] == ("modules",) and "logs" in parts)
        ):
            candidates.append(path)
    decoded, uninspectable = _decode_files(candidates, label_root=git_dir)
    _merge_findings(results, content_anonymity_findings(decoded))
    results["local Git metadata uninspectable"].update(uninspectable)
    return results


def _redacted_location(location: str) -> str:
    """Return a stable locator without echoing a possibly private path."""

    if re.fullmatch(r"(?:commit|history-blob|object):[0-9a-f]{12}", location):
        return location
    full_object = re.fullmatch(r"(history-blob|object):([0-9a-f]{40})", location)
    if full_object:
        return f"{full_object.group(1)}:{full_object.group(2)[:12]}"
    return f"location:{_stable_id(location)}"


def _print_results(title: str, results: dict[str, set[str]], *, gate: bool) -> bool:
    print(title)
    print("=" * len(title))
    failed = False
    for label in sorted(results):
        findings = results[label]
        status = "FAIL" if findings and gate else "REVIEW" if findings else "PASS"
        print(f"{label:<40} {status}")
        for finding in sorted(findings):
            print(f"  - {_redacted_location(finding)}")
        failed = failed or bool(findings)
    return failed


def main() -> None:
    args = parse_args()
    applied_exemptions: set[str] = set()
    candidate_files = release_candidate_files()
    candidate_text, candidate_uninspectable = _decode_files(
        candidate_files,
        label_root=REPOSITORY_ROOT,
    )
    public_results = release_content_anonymity_findings(
        candidate_text,
        applied_exemptions=applied_exemptions,
    )
    public_results["release-candidate uninspectable files"] = candidate_uninspectable
    raw_suffixes = {
        ".log",
        ".prof",
        ".trace",
        ".nsys-rep",
        ".ckpt",
        ".pt",
        ".pth",
    }
    public_results["raw local logs/results"] = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in candidate_files
        if path.suffix.casefold() in raw_suffixes
    }
    _merge_findings(
        public_results,
        git_history_anonymity_findings(
            REPOSITORY_ROOT,
            args.ref,
            applied_exemptions=applied_exemptions,
        ),
    )
    if args.archive is not None:
        _merge_findings(
            public_results,
            archive_anonymity_findings(
                args.archive,
                applied_exemptions=applied_exemptions,
            ),
        )

    public_failed = _print_results(
        "PUBLIC REF / RELEASE ANONYMITY GATE", public_results, gate=True
    )
    print("REVIEWED SYNTHETIC ABSOLUTE-PATH FIXTURES")
    print("=========================================")
    print(
        f"exact path + content hash + match hashes  "
        f"{'APPLIED' if applied_exemptions else 'NOT APPLIED'}"
    )
    for label in sorted(applied_exemptions):
        print(f"  - {_redacted_location(label)}")
    local_results = local_git_metadata_findings()
    _print_results(
        "LOCAL-ONLY GIT METADATA (NOT EXPORTED BY PUSH OR GIT ARCHIVE)",
        local_results,
        gate=False,
    )
    print(
        "Local-only findings are review warnings for raw-directory copies; they "
        "are not claims about what an ordinary Git push publishes."
    )
    if public_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ANONYMITY AUDIT: FAIL ({exc})")
        raise SystemExit(1) from None
