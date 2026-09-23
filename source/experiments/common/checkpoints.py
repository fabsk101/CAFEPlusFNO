"""Safe, reproducible checkpoint construction and loading helpers.

PyTorch exposes ``torch.__version__`` as a ``TorchVersion`` string subclass.
Serializing that object directly records a Python global which the restricted
``weights_only=True`` loader correctly rejects.  The writers use the helpers
in this module to keep checkpoint metadata to exact built-in container and
scalar types while leaving model, optimizer, and scheduler state dictionaries
untouched.

This module intentionally has no top-level PyTorch import.  Experiment entry
points must be able to verify and bootstrap the pinned SirenFNO backend before
any torch-dependent import is evaluated.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any


DEFAULT_STATE_DICT_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
)
"""Checkpoint fields whose tensor/container structure is preserved verbatim."""

_LEGACY_TORCH_VERSION_GLOBAL = "torch.torch_version.TorchVersion"
MODEL_INITIALIZATION_METADATA_KEY = "model_initialization_metadata"
_MODEL_STATE_METADATA_KEY = "_metadata"
_KNOWN_NEURALOP_INITIALIZATION_IDENTIFIERS = {
    "spectral_conv": "neuralop.layers.spectral_convolution.SpectralConv",
    "gelu": "torch._C._nn.gelu",
}
_LEGACY_NEURALOP_UNSAFE_GLOBALS = tuple(
    _KNOWN_NEURALOP_INITIALIZATION_IDENTIFIERS.values()
)


def _child_path(path: str, key: object) -> str:
    if isinstance(key, str):
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def normalize_builtin_metadata(value: Any, *, _path: str = "metadata") -> Any:
    """Return metadata composed only of exact, restricted-loader-safe built-ins.

    Supported leaves are exact ``None``, ``str``, ``bool``, ``int``, and
    finite ``float`` values.  String and numeric subclasses (including
    ``torch.torch_version.TorchVersion`` and NumPy scalar numbers) are copied
    into their exact built-in counterparts.  Mappings, lists, and tuples are
    rebuilt recursively; path-like values and enums are reduced to their
    built-in representations.  Unsupported objects fail closed instead of
    being silently stringified.
    """

    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{_path} must be finite, got {value!r}")
        return value

    # Order matters: bool is an Integral and TorchVersion is a str subclass.
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{_path} must be finite, got {value!r}")
        return result
    if isinstance(value, Enum):
        return normalize_builtin_metadata(value.value, _path=_path)
    if isinstance(value, os.PathLike):
        result = os.fspath(value)
        if not isinstance(result, str):
            raise TypeError(f"{_path} path representation must be text")
        return str(result)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = normalize_builtin_metadata(
                key, _path=f"{_path}.<key>"
            )
            if type(normalized_key) is not str:
                raise TypeError(
                    f"{_path} keys must normalize to exact str, got "
                    f"{type(normalized_key).__name__}"
                )
            if normalized_key in result:
                raise ValueError(
                    f"{_path} contains duplicate key {normalized_key!r} "
                    "after normalization"
                )
            result[normalized_key] = normalize_builtin_metadata(
                item, _path=_child_path(_path, normalized_key)
            )
        return result
    if isinstance(value, list):
        return [
            normalize_builtin_metadata(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            normalize_builtin_metadata(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        )

    raise TypeError(
        f"{_path} has unsupported metadata type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def normalize_environment_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one environment record to exact built-in metadata types."""

    normalized = normalize_builtin_metadata(metadata, _path="environment")
    if type(normalized) is not dict:  # Defensive; Mapping always normalizes to dict.
        raise TypeError("environment metadata must be a mapping")
    return normalized


def _known_neuralop_initialization_identifier(
    value: Any,
    *,
    path: str,
) -> str | None:
    """Return a fixed identifier for the two known upstream init objects."""

    import torch

    if value is torch._C._nn.gelu and path.endswith(".non_linearity"):
        return _KNOWN_NEURALOP_INITIALIZATION_IDENTIFIERS["gelu"]

    # Resolve the class only from the verified, pinned SirenFNO checkout. This
    # prevents accepting an arbitrary same-named class from site-packages.
    from experiments.common.sirenfno_backend import bootstrap_sirenfno_backend

    bootstrap_sirenfno_backend()
    from neuralop.layers.spectral_convolution import SpectralConv

    if value is SpectralConv and path.endswith(".conv_module"):
        return _KNOWN_NEURALOP_INITIALIZATION_IDENTIFIERS["spectral_conv"]
    return None


def normalize_neuralop_initialization_metadata(
    value: Any,
    *,
    _path: str = "model_initialization_metadata",
) -> Any:
    """Normalize upstream init data without evaluating stored identifiers."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = normalize_builtin_metadata(
                key, _path=f"{_path}.<key>"
            )
            if type(normalized_key) is not str:
                raise TypeError(
                    f"{_path} keys must normalize to exact str, got "
                    f"{type(normalized_key).__name__}"
                )
            if normalized_key in result:
                raise ValueError(
                    f"{_path} contains duplicate key {normalized_key!r} "
                    "after normalization"
                )
            result[normalized_key] = normalize_neuralop_initialization_metadata(
                item,
                _path=_child_path(_path, normalized_key),
            )
        return result
    if isinstance(value, list):
        return [
            normalize_neuralop_initialization_metadata(
                item, _path=f"{_path}[{index}]"
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            normalize_neuralop_initialization_metadata(
                item, _path=f"{_path}[{index}]"
            )
            for index, item in enumerate(value)
        )

    try:
        return normalize_builtin_metadata(value, _path=_path)
    except TypeError:
        identifier = _known_neuralop_initialization_identifier(
            value,
            path=_path,
        )
        if identifier is not None:
            return identifier
        raise


def _prepare_model_state_dict(
    value: Any,
    *,
    checkpoint_key: str,
) -> tuple[MutableMapping[str, Any], dict[str, Any] | None]:
    """Copy model weights and separate neuraloperator's mapping metadata key."""

    import torch

    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint.{checkpoint_key} must be a mapping")

    prepared_state = copy.copy(value)
    if not isinstance(prepared_state, MutableMapping):
        raise TypeError(
            f"checkpoint.{checkpoint_key} must be mutable after a shallow copy"
        )

    initialization_metadata: dict[str, Any] | None = None
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError(
                f"checkpoint.{checkpoint_key} state keys must be exact str"
            )
        if key == _MODEL_STATE_METADATA_KEY:
            normalized = normalize_neuralop_initialization_metadata(
                item,
                _path=(
                    f"checkpoint.{checkpoint_key}.{_MODEL_STATE_METADATA_KEY}"
                ),
            )
            if type(normalized) is not dict:
                raise TypeError(
                    f"checkpoint.{checkpoint_key}.{_MODEL_STATE_METADATA_KEY} "
                    "must be a mapping"
                )
            initialization_metadata = normalized
            del prepared_state[key]
            continue
        if not torch.is_tensor(item):
            raise TypeError(
                f"checkpoint.{checkpoint_key}.{key} must be a Tensor, got "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    # OrderedDict._metadata is a separate attribute containing module-version
    # data. Preserve it; do not confuse it with the mapping key removed above.
    if hasattr(value, "_metadata"):
        setattr(prepared_state, "_metadata", getattr(value, "_metadata"))
    return prepared_state, initialization_metadata


def prepare_weights_only_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    state_dict_keys: frozenset[str] = DEFAULT_STATE_DICT_KEYS,
) -> dict[str, Any]:
    """Normalize metadata and isolate neuraloperator init metadata from weights.

    The model state mapping is shallow-copied so the upstream mapping key named
    ``_metadata`` can be removed without mutating the caller. Its safe,
    normalized contents are stored separately. The distinct OrderedDict
    ``._metadata`` attribute and every Tensor are preserved. Optimizer and
    scheduler states retain their exact objects and structure.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if not isinstance(state_dict_keys, frozenset) or not all(
        type(key) is str for key in state_dict_keys
    ):
        raise TypeError("state_dict_keys must be a frozenset of exact str values")

    prepared: dict[str, Any] = {}
    model_initialization_metadata: dict[str, Any] | None = None
    for key, value in checkpoint.items():
        normalized_key = normalize_builtin_metadata(
            key, _path="checkpoint.<key>"
        )
        if type(normalized_key) is not str:
            raise TypeError("checkpoint keys must normalize to exact str")
        if normalized_key in prepared:
            raise ValueError(
                f"checkpoint contains duplicate key {normalized_key!r} "
                "after normalization"
            )
        if normalized_key == "model_state_dict" and normalized_key in state_dict_keys:
            prepared_state, extracted_metadata = _prepare_model_state_dict(
                value,
                checkpoint_key=normalized_key,
            )
            prepared[normalized_key] = prepared_state
            model_initialization_metadata = extracted_metadata
        elif normalized_key in state_dict_keys:
            if not isinstance(value, Mapping):
                raise TypeError(f"checkpoint.{normalized_key} must be a mapping")
            prepared[normalized_key] = value
        elif normalized_key == MODEL_INITIALIZATION_METADATA_KEY:
            normalized_metadata = normalize_neuralop_initialization_metadata(
                value,
                _path=_child_path("checkpoint", normalized_key),
            )
            if type(normalized_metadata) is not dict:
                raise TypeError(
                    f"checkpoint.{normalized_key} must be a mapping"
                )
            prepared[normalized_key] = normalized_metadata
        else:
            prepared[normalized_key] = normalize_builtin_metadata(
                value, _path=_child_path("checkpoint", normalized_key)
            )

    if model_initialization_metadata is not None:
        if MODEL_INITIALIZATION_METADATA_KEY in prepared:
            raise ValueError(
                "checkpoint contains both model_state_dict['_metadata'] and "
                f"top-level {MODEL_INITIALIZATION_METADATA_KEY!r}"
            )
        prepared[MODEL_INITIALIZATION_METADATA_KEY] = (
            model_initialization_metadata
        )
    return prepared


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Restricted-load payload plus explicit compatibility-path provenance."""

    checkpoint: dict[str, Any]
    legacy_torch_version_compatibility_used: bool
    legacy_neuraloperator_metadata_compatibility_used: bool

    @property
    def load_metadata(self) -> dict[str, str | bool]:
        """Return exact built-ins suitable for an audit or evaluation artifact."""

        return {
            "checkpoint_load_mode": "weights_only",
            "legacy_torch_version_compatibility_used": (
                self.legacy_torch_version_compatibility_used
            ),
            "legacy_neuraloperator_metadata_compatibility_used": (
                self.legacy_neuraloperator_metadata_compatibility_used
            ),
        }


def load_weights_only_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
    allow_legacy_torch_version: bool = False,
    allow_legacy_neuraloperator_metadata: bool = False,
) -> CheckpointLoadResult:
    """Load a checkpoint without ever falling back to unrestricted pickle.

    By default this is a plain ``torch.load(..., weights_only=True)``.  The
    optional legacy modes are deliberately narrow and independent. They retry
    only for explicitly recognized globals and allowlist only the opted-in
    ``TorchVersion`` and/or the two objects emitted by the verified pinned
    neuraloperator BaseModel. Returned metadata is normalized in memory; the
    source file is never changed.
    """

    import torch

    torch_version_compatibility_used = False
    neuralop_metadata_compatibility_used = False
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        error_text = str(error)
        torch_version_required = _LEGACY_TORCH_VERSION_GLOBAL in error_text
        neuralop_metadata_required = any(
            name in error_text for name in _LEGACY_NEURALOP_UNSAFE_GLOBALS
        )
        if not (
            (allow_legacy_torch_version and torch_version_required)
            or (
                allow_legacy_neuraloperator_metadata
                and neuralop_metadata_required
            )
        ):
            raise

        safe_global_objects: list[Any] = []
        if allow_legacy_torch_version:
            from torch.torch_version import TorchVersion

            safe_global_objects.append(TorchVersion)
            torch_version_compatibility_used = True
        if allow_legacy_neuraloperator_metadata:
            from experiments.common.sirenfno_backend import (
                bootstrap_sirenfno_backend,
            )

            bootstrap_sirenfno_backend()
            from neuralop.layers.spectral_convolution import SpectralConv

            safe_global_objects.extend([SpectralConv, torch._C._nn.gelu])
            neuralop_metadata_compatibility_used = True

        with torch.serialization.safe_globals(safe_global_objects):
            loaded = torch.load(
                path, map_location=map_location, weights_only=True
            )

    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    return CheckpointLoadResult(
        checkpoint=prepare_weights_only_checkpoint(loaded),
        legacy_torch_version_compatibility_used=(
            torch_version_compatibility_used
        ),
        legacy_neuraloperator_metadata_compatibility_used=(
            neuralop_metadata_compatibility_used
        ),
    )


__all__ = [
    "CheckpointLoadResult",
    "DEFAULT_STATE_DICT_KEYS",
    "MODEL_INITIALIZATION_METADATA_KEY",
    "load_weights_only_checkpoint",
    "normalize_builtin_metadata",
    "normalize_environment_metadata",
    "normalize_neuralop_initialization_metadata",
    "prepare_weights_only_checkpoint",
]
