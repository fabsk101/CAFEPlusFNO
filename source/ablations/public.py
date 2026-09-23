"""Public output identifiers and failure handling; never publish raw exceptions.

Paths remain available to callers for I/O.  Public artifacts contain relative
identifiers below an explicitly allowed root, or a fixed external artifact ID.
No exception text, subprocess stderr, hostname, or external basename is needed
to report a failed operation. Detailed private debugging is not enabled here.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable


_WINDOWS_PATH = re.compile(r"(?i)(?<![\w:])(?:[a-z]:[\\/]|\\\\|//)[^\r\n\"'<>|]*")
_POSIX_PATH = re.compile(r"(?<![\w.:/])/(?:[^\s\"'<>|,;\]\[{}()]+/?)+")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


class PublicError(RuntimeError):
    """A stable reason code with deliberately restricted public identities."""

    def __init__(self, code: str, **identity: Any):
        self.reason_code = code if re.fullmatch(r"[A-Z][A-Z0-9_]*", code) else "OPERATION_FAILED"
        self.identity = {
            key: value for key, value in identity.items()
            if key in {"dataset", "model_variant", "condition", "seed", "epoch", "artifact_id"}
            and ((isinstance(value, int) and not isinstance(value, bool))
                 or (isinstance(value, str) and _IDENTIFIER.fullmatch(value)))
        }
        super().__init__(self.reason_code)


class PublicFieldError(ValueError):
    """Field diagnostics drawn only from a caller's code-owned schema catalog.

    Never pass field names or catalogs copied from an untrusted artifact.
    Unknown fields and actual/expected values are deliberately not published.
    """

    def __init__(self, code: str, *, missing_fields=(), mismatched_fields=(), field_catalog=()):
        self.reason_code = code if re.fullmatch(r"[A-Z][A-Z0-9_]*", code) else "INVALID_CONTRACT"
        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
        catalog = {field for field in field_catalog if isinstance(field, str) and pattern.fullmatch(field)}
        self.missing_fields = sorted({field for field in missing_fields if isinstance(field, str) and field in catalog})
        self.mismatched_fields = sorted({field for field in mismatched_fields if isinstance(field, str) and field in catalog})
        super().__init__(self.reason_code)


def public_path(path: str | Path, root: str | Path | None = None,
                artifact_id: str = "external-artifact") -> str:
    """Lexical only: this function never resolves/stats an untrusted path."""
    text = str(path)
    external = artifact_id if _IDENTIFIER.fullmatch(artifact_id) else "external-artifact"
    if root is None:
        return external
    text_root = str(root)
    windows = bool(re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", text_root))
    path_type = PureWindowsPath if windows else PurePosixPath
    candidate, base = path_type(text), path_type(text_root)
    if ".." in candidate.parts:
        return external
    # A Windows or UNC path is external on POSIX, and vice versa.
    if not windows and (PureWindowsPath(text).is_absolute() or text.startswith("//")):
        return external
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return external
    return relative.as_posix() or "."


def safe_text(value: str) -> str:
    """Redact absolute Windows/POSIX/WSL/UNC paths in upstream progress text."""
    value = _WINDOWS_PATH.sub("<external-artifact>", value)
    return _POSIX_PATH.sub("<external-artifact>", value)


def public_payload(value: Any, root: str | Path | None = None) -> Any:
    if isinstance(value, Path):
        return public_path(value, root)
    if isinstance(value, dict):
        return {safe_text(str(key)): public_payload(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_payload(item, root) for item in value]
    if isinstance(value, str):
        # Public upstream authorship/license URLs are not filesystem paths.
        if re.match(r"^https?://", value):
            return value
        absolute_like = bool(re.match(r"^(?:[A-Za-z]:[\\/]|/|\\\\)", value))
        return public_path(value, root) if absolute_like else safe_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    # Do not stringify arbitrary objects: repr can contain paths/tokens.
    return "<unsupported-public-value>"


def error_record(exc: BaseException) -> dict[str, Any]:
    code = getattr(exc, "reason_code", None)
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", code):
        code = {FileNotFoundError: "ARTIFACT_NOT_FOUND", FileExistsError: "OUTPUT_EXISTS",
                PermissionError: "ACCESS_DENIED", ImportError: "DEPENDENCY_IMPORT_FAILED",
                ModuleNotFoundError: "DEPENDENCY_UNAVAILABLE", ValueError: "INVALID_CONTRACT",
                TypeError: "INVALID_TYPE", KeyboardInterrupt: "INTERRUPTED"}.get(type(exc), "OPERATION_FAILED")
    record: dict[str, Any] = {"status": "FAIL", "reason_code": code}
    if isinstance(exc, PublicError):
        record.update(exc.identity)
    if isinstance(exc, PublicFieldError):
        record.update(missing_fields=exc.missing_fields, mismatched_fields=exc.mismatched_fields)
    return record


def emit_json(value: Any, *, root: str | Path | None = None, stream=None) -> None:
    print(json.dumps(public_payload(value, root), indent=2, sort_keys=True, allow_nan=False),
          file=stream or sys.stdout, flush=True)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's message can contain arbitrary user-provided paths/secrets.
        self.print_usage(sys.stderr)
        emit_json({"status": "FAIL", "reason_code": "INVALID_ARGUMENTS"}, stream=sys.stderr)
        raise SystemExit(2)


class PublicStream:
    """Line-buffered progress stream: keep numeric progress and scrub paths."""

    def __init__(self, stream):
        self.stream, self.pending = stream, ""

    def write(self, value):
        self.pending += str(value)
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.stream.write(safe_text(line) + "\n")
        return len(value)

    def flush(self):
        if self.pending:
            self.stream.write(safe_text(self.pending))
            self.pending = ""
        self.stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def public_main(main: Callable, argv=None) -> None:
    """Guard an entrypoint; failures retain nonzero exit without tracebacks."""
    out, err = PublicStream(sys.stdout), PublicStream(sys.stderr)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            if argv is None:
                main()
            else:
                main(argv)
        except SystemExit as exc:
            if exc.code is None or isinstance(exc.code, int):
                raise
            emit_json({"status": "FAIL", "reason_code": "INVALID_INVOCATION"}, stream=sys.stderr)
            raise SystemExit(2) from None
        except (Exception, KeyboardInterrupt) as exc:
            emit_json(error_record(exc), stream=sys.stderr)
            raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1) from None
        finally:
            out.flush()
            err.flush()
