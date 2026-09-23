"""Names and immutable contracts shared by the ablation tools."""

from __future__ import annotations

from dataclasses import dataclass
import re

EXPERIMENT_ID = "cafe_plus_fno_ablation_v1"
DEFAULT_SEEDS = (0, 42, 73, 108, 202)
FORMAL_EPOCHS = 500


def seed_value(value: str | int) -> int:
    """Accept explicit NumPy/PyTorch-compatible seeds, independently of defaults."""
    if type(value) is int:
        seed = value
    elif type(value) is str and re.fullmatch(r"[+-]?[0-9]+", value):
        seed = int(value)
    else:
        raise ValueError("SEED_TYPE_INVALID")
    if not 0 <= seed < 2**32:
        raise ValueError("SEED_OUT_OF_RANGE")
    return seed


def validate_seeds(seeds: list[int]) -> None:
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("SEED_LIST_EMPTY_OR_INVALID")
    for seed in seeds:
        if type(seed) is not int:
            raise ValueError("SEED_TYPE_INVALID")
        seed_value(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("SEED_DUPLICATE")

MODEL_VARIANTS = {
    "dense": "cafe_plus_fno",
    "cp": "cp_cafe_plus_fno",
    "tt": "tt_cafe_plus_fno",
    "tucker": "tucker_cafe_plus_fno",
}


INPUT32_CONDITION = "without_linear_branches_input32"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    use_fourier: bool
    use_chebyshev: bool
    learnable_sigma: bool
    use_linear_branches: bool
    change_scope: str
    use_hadamard_product: bool = True


ABLATION_CONDITIONS = {
    "full": Condition(
        "full",
        "Full",
        True,
        True,
        False,
        True,
        "No architecture change: Fourier + Chebyshev, fixed sigma, and the "
        "original Linear branches with Hadamard multiplication.",
    ),
    "learnable_sigma": Condition(
        "learnable_sigma",
        "Learnable sigma",
        True,
        True,
        True,
        True,
        "Only log_sigma changes from a persistent buffer to a trainable scalar "
        "Parameter in every CAFE encoder; the sampled Gaussian G basis stays a "
        "fixed persistent buffer.",
    ),
    "fourier_only": Condition(
        "fourier_only",
        "Fourier only",
        True,
        False,
        False,
        True,
        "Chebyshev coordinates are excluded from the encoder input. Fourier "
        "basis count, G, phase scale, sigma initialization, Linear branches, "
        "and Hadamard multiplication are retained.",
    ),
    "chebyshev_only": Condition(
        "chebyshev_only",
        "Chebyshev only",
        False,
        True,
        False,
        True,
        "Random Fourier features are excluded. Chebyshev order, coordinate "
        "normalization, Linear branches, and Hadamard multiplication remain.",
    ),
    "without_linear_branches": Condition(
        "without_linear_branches",
        "w/o Linear branches and Hadamard product",
        True,
        True,
        False,
        False,
        "The branch Linear modules and their elementwise product are removed. "
        "The concatenated CAFE embedding goes directly into the kernel MLP, "
        "resizing its input projection if needed. Kernel reconstruction is "
        "retained; the embedding is not multiplied by zeros.",
        use_hadamard_product=False,
    ),
    INPUT32_CONDITION: Condition(
        INPUT32_CONDITION,
        "w/o Linear branches and Hadamard product (direct input 32)",
        True,
        True,
        False,
        False,
        "Dense only. The original branch-free condition is retained unchanged. "
        "Use the first eight columns of the native encoder's fixed Gaussian G "
        "with their cosine/sine pairs (16 features), plus T0 through T15 in "
        "1D or T0 through T7 per axis in 2D (16 Chebyshev features). The 32 "
        "features enter the kernel MLP directly; no Linear branch or Hadamard "
        "combination remains. Hidden and output widths are unchanged. This is "
        "a fixed-32 input comparison, not an equal-total-parameter claim.",
        use_hadamard_product=False,
    ),
}


def validate_condition_variant(condition: str, model_variant: str) -> None:
    """Reject unsupported tuples before constructing models or writing runs."""
    if condition not in ABLATION_CONDITIONS or model_variant not in MODEL_VARIANTS:
        raise ValueError("ABLATION_IDENTITY_INVALID")
    if condition == INPUT32_CONDITION and model_variant != "dense":
        raise ValueError("INPUT32_DENSE_ONLY")


def conditions_for_variant(model_variant: str) -> tuple[str, ...]:
    """Return runnable conditions; old variants retain their five conditions."""
    if model_variant not in MODEL_VARIANTS:
        raise ValueError("MODEL_VARIANT_INVALID")
    return tuple(key for key in ABLATION_CONDITIONS
                 if model_variant == "dense" or key != INPUT32_CONDITION)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    config_module: str
    train_module: str
    spatial_dim: int
    verify_function: str
    load_function: str
    contract_key: str
    trainer_kind: str


DATASETS = {
    "airfoil": DatasetSpec(
        "airfoil", "experiments.configs.airfoil", "experiments.train_airfoil",
        2, "verify_airfoil_dataset", "load_airfoil", "airfoil221x51", "loop",
    ),
    "burgers1d": DatasetSpec(
        "burgers1d", "experiments.configs.burgers1d", "experiments.train_burgers",
        1, "verify_burgers_dataset", "load_burgers", "burgers1024", "loop",
    ),
    "cfd1d": DatasetSpec(
        "cfd1d", "experiments.configs.cfd1d", "experiments.train_cfd1d",
        1, "verify_cfd1d_dataset", "load_cfd1d", "cfd1d1024", "loop",
    ),
    "cfd2d": DatasetSpec(
        "cfd2d", "experiments.configs.cfd2d", "experiments.train_cfd2d",
        2, "verify_cfd2d_dataset", "load_cfd2d", "cfd2d128", "loop",
    ),
    "darcy": DatasetSpec(
        "darcy", "experiments.configs.darcy", "experiments.train_darcy",
        2, "verify_dataset", "load_darcy", "darcy128", "neuralop_trainer",
    ),
    "ns2d": DatasetSpec(
        "ns2d", "experiments.configs.ns2d", "experiments.train_ns2d",
        2, "verify_ns_dataset", "load_ns", "ns128", "neuralop_trainer",
    ),
    "reacdiff1d": DatasetSpec(
        "reacdiff1d", "experiments.configs.reacdiff1d", "experiments.train_reacdiff",
        1, "verify_reacdiff_dataset", "load_reacdiff", "reacdiff1024", "loop",
    ),
}
