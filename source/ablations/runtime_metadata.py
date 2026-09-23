"""Observed execution environment, separate from requested runtime policy."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys

from .public import public_payload
from .runtime import THREAD_ENVIRONMENT, get_runtime_policy

RUNTIME_SCHEMA_VERSION = 1
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch", "numpy": "numpy", "h5py": "h5py", "tensorly": "tensorly",
    "tensorly_torch": "tensorly-torch", "opt_einsum": "opt-einsum", "packaging": "packaging",
}


def capture_runtime_metadata(*, profile: str, device: str, seed: int | None = None) -> dict:
    """Read actual settings; unavailable values are null with explicit reasons.

    This never changes precision/determinism and never initializes a GPU during
    CPU validation. Individual run seed is intentionally kept in run identity,
    not in this environment compatibility record.
    """
    unavailable: dict[str, str] = {}

    def observe(key, fn):
        try:
            value = fn()
            if value is None:
                unavailable[key] = "not_reported_by_runtime"
            return value
        except Exception:
            unavailable[key] = "query_unavailable"
            return None

    packages = {}
    for key, distribution in PACKAGE_DISTRIBUTIONS.items():
        packages[key] = observe("packages." + key, lambda distribution=distribution: importlib.metadata.version(distribution))
    torch = sys.modules.get("torch")
    cuda = {"build": None, "cudnn": None, "gpu_product_names": None, "nvidia_driver": None}
    operations = {"amp_cpu_enabled": None, "amp_cuda_enabled": None, "tf32_matmul": None,
                  "tf32_cudnn": None, "float32_matmul_precision": None,
                  "deterministic_algorithms": None, "cudnn_benchmark": None, "cudnn_deterministic": None}
    threads = {"intra_op": None, "inter_op": None,
               "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}}
    if torch is None:
        for namespace, values in (("cuda", cuda), ("operations", operations), ("threads", threads)):
            for key in values:
                if values[key] is None:
                    unavailable[f"{namespace}.{key}"] = "torch_not_imported"
    else:
        packages["torch"] = str(torch.__version__)
        cuda["build"] = observe("cuda.build", lambda: torch.version.cuda)
        cuda["cudnn"] = observe("cuda.cudnn", lambda: torch.backends.cudnn.version())

        def autocast(kind):
            try:
                return bool(torch.is_autocast_enabled(kind))
            except TypeError:
                return bool(torch.is_autocast_cpu_enabled() if kind == "cpu" else torch.is_autocast_enabled())

        accessors = {
            "amp_cpu_enabled": lambda: autocast("cpu"),
            "amp_cuda_enabled": lambda: autocast("cuda"),
            "tf32_matmul": lambda: bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": lambda: bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled,
            "cudnn_benchmark": lambda: bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": lambda: bool(torch.backends.cudnn.deterministic),
        }
        for key, accessor in accessors.items():
            operations[key] = observe("operations." + key, accessor)
        threads["intra_op"] = observe("threads.intra_op", torch.get_num_threads)
        threads["inter_op"] = observe("threads.inter_op", torch.get_num_interop_threads)
        if device != "cuda":
            unavailable["cuda.gpu_product_names"] = "cpu_execution_no_gpu_query"
            unavailable["cuda.nvidia_driver"] = "cpu_execution_no_gpu_query"
        elif observe("cuda.available", torch.cuda.is_available):
            cuda["gpu_product_names"] = observe("cuda.gpu_product_names", lambda: [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            def driver():
                result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=True, timeout=5)
                values = sorted(set(line.strip() for line in result.stdout.splitlines()))
                return values if values and all(re.fullmatch(r"[0-9.]+", item) for item in values) else None
            cuda["nvidia_driver"] = observe("cuda.nvidia_driver", driver)
        else:
            unavailable["cuda.gpu_product_names"] = "cuda_unavailable"
            unavailable["cuda.nvidia_driver"] = "cuda_unavailable"
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "profile": profile, "device": device,
        "runtime_policy": get_runtime_policy(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine()},
        "packages": packages, "cuda": cuda, "operations": operations, "threads": threads,
        "seed_policy": {"selection": "explicit_run_seed", "range": "0 <= seed < 2**32",
                        "rng_initialization": "pinned_dataset_adapter", "one_process_per_seed": True},
        "comparison_policy": {
            "accuracy": "source_config_data_evaluation_and_runtime_compatibility_required",
            "timing": "same_hardware_threads_packages_and_runtime_required",
            "bitwise_or_cross_gpu_equality": "not_guaranteed",
        },
        "unavailable": unavailable,
    }
    return public_payload(payload)


# A descriptive alias for callers that prefer collection terminology.
collect_runtime_metadata = capture_runtime_metadata
