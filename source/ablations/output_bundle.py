"""Publish the five aggregate files with staging, exclusion and rollback.

Publication uses individual replacements, not an atomic five-file snapshot.
An interrupted writer or incomplete rollback leaves a lock/transaction marker;
subsequent aggregation refuses that destination pending explicit inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from .public import PublicError
from .runtime import output_path

OWNED_OUTPUT_NAMES = ("aggregate.json", "aggregate.csv", "seed_values.csv", "table.md", "table.tex")
LOCK_NAME = ".aggregate.lock"
TRANSACTION_NAME = ".aggregate.transaction"


def _file(path: Path) -> Path | None:
    path = output_path(path)  # Checks every ancestor with lstat before access.
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PublicError("AGGREGATE_OUTPUT_UNSAFE")
    return path


def assert_publication_idle(destination: Path) -> None:
    """Do not consume an interrupted publication, even with overwrite set."""
    destination = output_path(destination)
    for name in (LOCK_NAME, TRANSACTION_NAME):
        marker = output_path(destination / name)
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        raise PublicError("AGGREGATE_PUBLICATION_PENDING")


def _snapshot(destination: Path) -> dict[str, bytes | None]:
    return {name: path.read_bytes() if (path := _file(destination / name)) else None
            for name in OWNED_OUTPUT_NAMES}


def _write_new(path: Path, payload: bytes) -> None:
    """Exclusive files: a failed write can only leave our staging file."""
    path = output_path(path)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _journal(transaction: Path, record: dict) -> None:
    temporary = transaction / "journal.next"
    _write_new(temporary, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    _file(temporary)
    _file(transaction / "journal.json")
    os.replace(temporary, transaction / "journal.json")


def _clean_transaction(transaction: Path) -> None:
    """Delete only known files created in our exclusively created directory."""
    transaction = output_path(transaction)
    if not transaction.exists():
        return
    allowed = {f"{prefix}-{name}" for prefix in ("new", "old") for name in OWNED_OUTPUT_NAMES}
    allowed.update(("journal.next", "journal.json"))
    children = list(transaction.iterdir())
    if any(child.name not in allowed for child in children):
        raise PublicError("AGGREGATE_PUBLICATION_RECOVERY_REQUIRED")
    # Keep the recovery journal until staging/backup cleanup is complete.
    for child in sorted(children, key=lambda path: path.name == "journal.json"):
        _file(child)
        child.unlink()
    transaction.rmdir()


def _release_lock(lock: Path, token: bytes) -> None:
    if _file(lock) is None or lock.read_bytes() != token:
        raise PublicError("AGGREGATE_PUBLICATION_RECOVERY_REQUIRED")
    lock.unlink()


def _rollback(destination: Path, transaction: Path, previous, contents) -> list[str]:
    failures = []
    for name in OWNED_OUTPUT_NAMES:
        try:
            target = destination / name
            present = _file(target)
            current = present.read_bytes() if present else None
            if current == previous[name]:
                continue
            # Never remove or replace unexpected contents from another writer.
            if current is not None and current != contents[name]:
                raise PublicError("AGGREGATE_OUTPUT_CHANGED_DURING_PUBLICATION")
            if previous[name] is None:
                if present:
                    present.unlink()
            else:
                backup = _file(transaction / f"old-{name}")
                if backup is None or backup.read_bytes() != previous[name]:
                    raise PublicError("AGGREGATE_BACKUP_INVALID")
                os.replace(backup, target)
        except Exception:
            failures.append(name)
    return failures


def publish_output_bundle(destination: Path, contents: dict[str, bytes], *, overwrite=False) -> None:
    """Publish fully serialized bytes, preserving prior outputs on known failure.

    All five targets are checked before staging, all stages/backups are complete
    before replacement, and a cooperating writer must acquire the exclusive
    lock. A process/power interruption cannot be rolled back by this process;
    the persistent markers instead make subsequent writes fail closed.
    """
    if (not isinstance(contents, dict) or set(contents) != set(OWNED_OUTPUT_NAMES)
            or any(type(value) is not bytes for value in contents.values()) or type(overwrite) is not bool):
        raise PublicError("AGGREGATE_OUTPUT_BUNDLE_INVALID")
    destination = output_path(destination)
    assert_publication_idle(destination)
    if destination.exists() and not destination.is_dir():
        raise PublicError("AGGREGATE_OUTPUT_DIRECTORY_INVALID")
    initial = _snapshot(destination)
    if not overwrite and any(value is not None for value in initial.values()):
        raise PublicError("AGGREGATE_OUTPUT_EXISTS")
    destination.mkdir(parents=True, exist_ok=True)
    lock = output_path(destination / LOCK_NAME)
    token = (json.dumps({"schema_version": 1, "writer_token": uuid.uuid4().hex}) + "\n").encode("utf-8")
    try:
        _write_new(lock, token)
    except FileExistsError:
        raise PublicError("AGGREGATE_PUBLICATION_PENDING") from None
    except OSError:
        raise PublicError("AGGREGATE_PUBLICATION_LOCK_FAILED") from None
    transaction = output_path(destination / TRANSACTION_NAME)
    publishing = False
    committed = False
    created_transaction = False
    previous = None
    record = None
    try:
        # The second preflight is protected by the lock. A preexisting marker
        # can belong only to an interrupted publication and is never removed.
        if transaction.exists():
            raise PublicError("AGGREGATE_PUBLICATION_PENDING")
        previous = _snapshot(destination)
        if previous != initial:
            raise PublicError("AGGREGATE_OUTPUT_CHANGED_DURING_PUBLICATION")
        transaction.mkdir(exist_ok=False)
        created_transaction = True
        for name in OWNED_OUTPUT_NAMES:
            _write_new(transaction / f"new-{name}", contents[name])
            if previous[name] is not None:
                _write_new(transaction / f"old-{name}", previous[name])
        record = {"schema_version": 1, "status": "STAGED",
                  "files": list(OWNED_OUTPUT_NAMES),
                  "prior_sha256": {name: hashlib.sha256(value).hexdigest() if value is not None else None
                                   for name, value in previous.items()},
                  "new_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}}
        _journal(transaction, record)
        if _snapshot(destination) != previous:
            raise PublicError("AGGREGATE_OUTPUT_CHANGED_DURING_PUBLICATION")
        publishing = True
        for name in OWNED_OUTPUT_NAMES:
            _file(destination / name)
            staged = _file(transaction / f"new-{name}")
            if staged is None or staged.read_bytes() != contents[name]:
                raise PublicError("AGGREGATE_STAGING_INVALID")
            os.replace(staged, destination / name)
        if _snapshot(destination) != contents:
            raise PublicError("AGGREGATE_PUBLICATION_VERIFICATION_FAILED")
        committed = True
        record["status"] = "COMMITTED"
        _journal(transaction, record)
        _clean_transaction(transaction)
        _release_lock(lock, token)
    except Exception as error:
        if committed:
            # All targets are the new bundle, but failed cleanup is still an
            # explicit pending state. Never silently accept/rewrite it later.
            raise PublicError("AGGREGATE_PUBLICATION_CLEANUP_PENDING") from None
        if publishing:
            failures = _rollback(destination, transaction, previous, contents)
            if failures:
                if record is not None:
                    record.update(status="RECOVERY_REQUIRED", rollback_failed_files=failures)
                    try:
                        _journal(transaction, record)
                    except Exception:
                        pass  # Existing STAGED journal and lock remain evidence.
                raise PublicError("AGGREGATE_PUBLICATION_RECOVERY_REQUIRED") from None
        try:
            if created_transaction:
                _clean_transaction(transaction)
            _release_lock(lock, token)
        except Exception:
            raise PublicError("AGGREGATE_PUBLICATION_RECOVERY_REQUIRED") from None
        if not created_transaction and isinstance(error, PublicError):
            raise error
        raise PublicError("AGGREGATE_PUBLICATION_REPLACEMENT_FAILED" if publishing
                          else "AGGREGATE_PUBLICATION_STAGING_FAILED") from None
