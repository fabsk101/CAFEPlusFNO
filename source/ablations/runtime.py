"""Workspace-contained paths and process-local runtime settings."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_RUNTIME_POLICY: dict = {}


def absolute(path: str | Path) -> Path:
    # Lexical normalization deliberately precedes any filesystem resolution.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def contained(path: str | Path, root: str | Path) -> Path:
    root, path = absolute(root), absolute(path)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path must stay within the allowed workspace") from exc
    # lstat never follows a final link. Inspect each ancestor before its child.
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Linked path is not allowed")
    return path


def workspace_root() -> Path:
    return absolute(os.environ.get("CAFE_ABLATION_WORKSPACE_ROOT", Path(__file__).parent.parent))


def output_path(path: str | Path) -> Path:
    path = contained(path, workspace_root())
    repository = os.environ.get("CAFE_ABLATION_REPOSITORY_ROOT")
    # A detached development source copy remains read-only.
    if repository and absolute(repository) != workspace_root():
        try:
            path.relative_to(absolute(repository))
        except ValueError:
            pass
        else:
            raise ValueError("Output may not enter the reference snapshot")
    return path


def configure_runtime(*, cpu: bool, profile: str = "formal", threads: int | None = None,
                      cache_area: str | None = None) -> Path:
    """Contain caches without silently changing a formal numerical policy.

    Only diagnostic CPU work receives the one-thread development default.
    Explicit thread overrides apply identically to all conditions in a sweep.
    CUDA visibility is changed only for an explicitly CPU-only diagnostic.
    Determinism, TF32, AMP, and cuDNN policy are never changed here.
    """
    global _RUNTIME_POLICY
    if profile not in {"formal", "diagnostic", "diagnostic_cpu"}:
        raise ValueError("Unknown execution profile")
    if cache_area not in (None, "tmp", "results"):
        raise ValueError("Unknown workspace cache area")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        raise ValueError("Thread override must be a positive integer")
    diagnostic_cpu = profile != "formal" and cpu
    if diagnostic_cpu and threads not in (None, 1):
        raise ValueError("CPU diagnostic tests require exactly one thread")
    effective_threads = 1 if diagnostic_cpu else threads
    inherited = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    normalized_profile = "diagnostic_cpu" if diagnostic_cpu else profile
    if (_RUNTIME_POLICY.get("profile") == normalized_profile
            and _RUNTIME_POLICY.get("requested_device") == ("cpu" if cpu else "cuda")
            and _RUNTIME_POLICY.get("thread_override") == threads
            and _RUNTIME_POLICY.get("applied_thread_environment") == inherited):
        inherited = dict(_RUNTIME_POLICY["inherited_thread_environment"])
    # Parent release provenance permits generated results/, but treats tmp/
    # source-adjacent files as unexpected additions in a Git-less release.
    selected_cache_area = cache_area or ("tmp" if diagnostic_cpu else "results")
    runtime = output_path(workspace_root() / selected_cache_area / "ablations_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    locations = {
        "TMP": runtime / "temp", "TEMP": runtime / "temp", "TMPDIR": runtime / "temp",
        "XDG_CACHE_HOME": runtime / "cache", "TORCH_HOME": runtime / "torch",
        "TORCHINDUCTOR_CACHE_DIR": runtime / "inductor", "TRITON_CACHE_DIR": runtime / "triton",
        "MPLCONFIGDIR": runtime / "matplotlib", "HF_HOME": runtime / "huggingface",
        "WANDB_DIR": runtime / "wandb", "CUDA_CACHE_PATH": runtime / "cuda",
    }
    for name, destination in locations.items():
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    tempfile.tempdir = str(locations["TMPDIR"])
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if effective_threads is not None:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = str(effective_threads)
    if diagnostic_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _RUNTIME_POLICY = {
        "profile": normalized_profile,
        "requested_device": "cpu" if cpu else "cuda",
        "thread_override": threads,
        "effective_thread_override": effective_threads,
        "inherited_thread_environment": inherited,
        "applied_thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "numerical_policy": "inherited_without_override",
        "cache_policy": "workspace_local",
    }
    if cache_area is not None:
        _RUNTIME_POLICY["cache_area"] = cache_area
    return runtime


def apply_torch_runtime() -> None:
    """Apply a chosen thread override only after guarded torch import."""
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("Import torch before applying its runtime policy")
    threads = _RUNTIME_POLICY.get("effective_thread_override")
    if threads is None:
        return
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != threads:
        torch.set_num_interop_threads(threads)


def get_runtime_policy() -> dict:
    from copy import deepcopy
    return deepcopy(_RUNTIME_POLICY)


def validate_runtime_metadata(value: dict, *, formal: bool = True) -> None:
    """Require the observed schema; never fill absent historical information."""
    import math
    import re
    from .public import PublicError, public_payload
    from .runtime_metadata import PACKAGE_DISTRIBUTIONS, RUNTIME_SCHEMA_VERSION

    def require(ok):
        if not ok:
            raise PublicError("RUNTIME_METADATA_INVALID")

    require(isinstance(value, dict))
    require(value.get("schema_version") == RUNTIME_SCHEMA_VERSION)
    require(value.get("profile") in {"formal", "diagnostic", "diagnostic_cpu"})
    require(value.get("device") in {"cpu", "cuda"})
    if formal:
        require(value["profile"] == "formal")
    unavailable = value.get("unavailable")
    require(isinstance(unavailable, dict) and all(isinstance(k, str) and isinstance(v, str) and v
                                               for k, v in unavailable.items()))

    def field(section, key, predicate):
        mapping = value.get(section)
        require(isinstance(mapping, dict) and key in mapping)
        item = mapping[key]
        if item is None:
            require(bool(unavailable.get(section + "." + key)))
        else:
            require(predicate(item))

    text = lambda item: isinstance(item, str) and bool(item)
    positive = lambda item: type(item) is int and item > 0
    for key in ("version", "implementation"):
        field("python", key, text)
    for key in ("system", "release", "architecture"):
        field("os", key, text)
    for key in PACKAGE_DISTRIBUTIONS:
        field("packages", key, text)
    for key in ("amp_cpu_enabled", "amp_cuda_enabled", "tf32_matmul", "tf32_cudnn",
                "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic"):
        field("operations", key, lambda item: type(item) is bool)
    field("operations", "float32_matmul_precision", lambda item: item in {"highest", "high", "medium"})
    field("cuda", "build", text)
    field("cuda", "cudnn", positive)
    field("cuda", "gpu_product_names", lambda item: isinstance(item, list) and bool(item) and all(text(x) for x in item))
    field("cuda", "nvidia_driver", lambda item: isinstance(item, list) and bool(item) and all(isinstance(x, str) and re.fullmatch(r"[0-9.]+", x) for x in item))
    field("threads", "intra_op", positive)
    field("threads", "inter_op", positive)
    thread_environment = value["threads"].get("environment")
    require(isinstance(thread_environment, dict) and set(thread_environment) == set(THREAD_ENVIRONMENT))
    require(all(item is None or isinstance(item, str) for item in thread_environment.values()))
    policy = value.get("runtime_policy")
    require(isinstance(policy, dict) and policy.get("profile") in {"formal", "diagnostic", "diagnostic_cpu"})
    if formal:
        require(policy["profile"] == "formal")
    require(policy.get("requested_device") == value["device"])
    require(policy.get("numerical_policy") == "inherited_without_override")
    require(policy.get("cache_policy") == "workspace_local")
    for key in ("thread_override", "effective_thread_override"):
        require(key in policy and (policy[key] is None or positive(policy[key])))
    for key in ("inherited_thread_environment", "applied_thread_environment"):
        require(isinstance(policy.get(key), dict) and set(policy[key]) == set(THREAD_ENVIRONMENT))
        require(all(item is None or isinstance(item, str) for item in policy[key].values()))
    require(policy["applied_thread_environment"] == thread_environment)
    require(isinstance(value.get("seed_policy"), dict))
    require(value["seed_policy"].get("selection") == "explicit_run_seed")
    require(value["seed_policy"].get("range") == "0 <= seed < 2**32")
    require(value["seed_policy"].get("one_process_per_seed") is True)
    require(isinstance(value.get("comparison_policy"), dict))

    def finite_tree(item):
        if isinstance(item, float):
            require(math.isfinite(item))
        elif isinstance(item, dict):
            for child in item.values():
                finite_tree(child)
        elif isinstance(item, list):
            for child in item:
                finite_tree(child)
    finite_tree(value)
    require(public_payload(value) == value)


def require_runtime_comparable(value: dict) -> None:
    """Unknown mandatory observations cannot establish formal comparability.

    Driver discovery is optional evidence; missing driver alone does not change
    a verified accuracy record into a failure, but cannot support strict timing
    comparisons. A diagnostic synthetic runtime is never promoted by this API.
    """
    from .public import PublicError
    validate_runtime_metadata(value, formal=True)
    required = {
        "python": ("version", "implementation"),
        "os": ("system", "release", "architecture"),
        "packages": ("torch", "numpy"),
        "operations": ("amp_cpu_enabled", "amp_cuda_enabled", "tf32_matmul", "tf32_cudnn",
                       "float32_matmul_precision", "deterministic_algorithms", "cudnn_benchmark",
                       "cudnn_deterministic"),
        "threads": ("intra_op", "inter_op"),
    }
    if value["device"] == "cuda":
        required["cuda"] = ("build", "cudnn", "gpu_product_names")
    if any(value[section][key] is None for section, keys in required.items() for key in keys):
        raise PublicError("RUNTIME_ACCURACY_EVIDENCE_INCOMPLETE")


def runtime_timing_evidence(value: dict) -> dict:
    require_runtime_comparable(value)
    has_driver = value["device"] != "cuda" or value["cuda"]["nvidia_driver"] is not None
    return {"accuracy_runtime_observations_complete": True,
            "timing_runtime_observations_complete": has_driver,
            "timing_unavailable_reason": None if has_driver else "nvidia_driver_unavailable",
            "timing_comparison_requires_identical_hardware_and_settings": True}
