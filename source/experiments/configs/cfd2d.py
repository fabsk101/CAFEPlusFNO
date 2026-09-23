"""Canonical released-runtime SirenFNO CFD-2D experiment configuration.

The pinned author helper selects only the ``Vx`` dataset from the public
PDEBench HDF5 artifact. The verified runtime is therefore a scalar 5-to-5
autoregressive task; the other top-level physical fields are not concatenated.
"""

from typing import Final


DATASET_ID: Final = "cfd2d128"
EXPERIMENT_ID: Final = "cfd2d128_released_sirenfno_vx"
DATASET_NAME: Final = "PDEBench 2D CFD (released SirenFNO Vx field)"
DATASET_SOURCE: Final = (
    "https://darus.uni-stuttgart.de/api/access/datafile/164687"
)
DATASET_FILENAME: Final = (
    "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
)
SELECTED_FIELD: Final = "Vx"
SELECTED_HDF5_KEY: Final = "/Vx"
CHANNEL_SEMANTICS: Final = ("Vx velocity component",)
DATASET_FIELD_POLICY: Final = "released_sirenfno_single_field"
AUTHOR_SOURCE_NOTE: Final = (
    "The pinned configuration initializes physical_channels=3, then replaces "
    "it with the loaded tensor's final dimension. The pinned HDF5 selector "
    "chooses Vx, yielding one runtime channel."
)
RESOLUTION: Final = (128, 128)
EXPECTED_TIME_STEPS: Final = 21
REDUCE_X: Final = 1
REDUCE_T: Final = 1
PHYSICAL_CHANNELS: Final = 1
INPUT_STEPS: Final = 5
ROLLOUT: Final = 5
INPUT_DIM: Final = INPUT_STEPS * PHYSICAL_CHANNELS
OUTPUT_DIM: Final = PHYSICAL_CHANNELS
FIELD_DIM: Final = PHYSICAL_CHANNELS
N_TRAIN: Final = 1800
N_VAL: Final = 0
N_TEST: Final = 200

BATCH_SIZE: Final = 32
NUM_WORKERS: Final = 0
PIN_MEMORY: Final = True
TRAIN_DROP_LAST: Final = True
EVAL_DROP_LAST: Final = False

EPOCHS: Final = 500
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4
SCHEDULER_T_MAX: Final = 500
EVAL_INTERVAL: Final = 1
USE_AMP: Final = False
PUSHFORWARD_DETACH: Final = False
MAIN_REPORTING_METRIC: Final = (
    "final_test_corrected_trajectory_relative_l2"
)
SECONDARY_REPORTING_METRICS: Final = (
    "final_test_corrected_step_relative_l2",
)
CANONICAL_SEED: Final = 42
SEED_POLICY: Final = (
    "repository_multi_seed_harness_single_global_seed_before_loader_and_model"
)
SELECTION_POLICY: Final = "fixed_final_epoch_no_test_selection"
SPLIT_POLICY: Final = "first_1800_train_next_200_test_no_random_split"
NORMALIZATION_POLICY: Final = "none"

CFD2D_SOURCE_BLOBS: Final = {
    "SirenFNO2D.py": "b0a987e42cd00547dfa64d0e4dfa619ee7106730",
    "baseline/AMFNO.py": "70b2affe7237e0123ef0baa0bd970b62ec87a020",
    "baseline/UFNO.py": "594dfbc7147ffe737930eaa12e12330c085448d3",
    "neuralop/utils.py": "f4fa635f4726b98daefcca71d580a2beb6fd4594",
    "train_CFD2D.py": "cc36bdbef018f0f5e13587dbf3a4d04328a15d27",
    "utils.py": "240a272db4d1b3f96b7fa4023a581c3876613d9b",
}

MODEL_CHOICES: Final = (
    "fno", "ufno", "tfno_cp", "amfno",
    "sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno",
    "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
    "tucker_cafe_plus_fno",
)

FNO_CONFIG: Final = {
    "n_modes": (32, 32),
    "hidden_channels": 32,
    "in_channels": INPUT_DIM,
    "out_channels": OUTPUT_DIM,
    "n_layers": 4,
    "lifting_channels": 64,
    "projection_channels": 64,
}
TFNO_CP_CONFIG: Final = {
    **FNO_CONFIG,
    "implementation": "factorized",
    "factorization": "cp",
    "rank": 0.05,
}
AMFNO_CONFIG: Final = {
    "width": 32,
    "n1": 10,
    "n2": 10,
    "padding": 0,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "mlp_dropout": 0,
}
UFNO_CONFIG: Final = {
    "n_modes": (12, 12),
    "hidden_channels": 32,
    "in_channels": INPUT_DIM,
    "out_channels": OUTPUT_DIM,
    "n_layers": 4,
    "positional_embedding": "grid",
    "use_channel_mlp": True,
    "use_unet_from": 2,
    "unet_dropout": 0.0,
    "domain_padding": None,
    "fno_block_precision": "full",
}
SIRENFNO_COMMON_CONFIG: Final = {
    "width": 32,
    "padding": 0,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "mlp_dropout": 0.0,
    "add_grid": True,
    "siren_dim_in": 16,
    "hidden_dim": 32,
    "omega": 30.0,
    "n_hidden": 1,
    "ff_sigma": 256,
    "learnable_ff": True,
}
SIRENFNO_FACTORIZATIONS: Final = {
    "sirenfno": {"factorization": "dense"},
    "cpsirenfno": {"factorization": "cp", "rank": 8},
    "ttsirenfno": {"factorization": "tt", "rank": 8},
    "tuckersirenfno": {"factorization": "tucker", "rank": 8},
}

CAFEPLUSFNO_COMMON_CONFIG: Final = {
    "width": 32,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "padding": 0,
    "mlp_dropout": 0.0,
    "add_grid": True,
    "num_layers": 4,
    "ffn_expansion": 4,
    "rff_basis": 32,
    "cheb_basis": 32,
    "cafe_branches": 2,
    "cafe_branch_dim": None,
    "kernel_hidden_dim": 32,
    "kernel_mlp_type": "joint",
    "kernel_activation": "gelu",
    "kernel_output_init_std": 1e-3,
    "sigma_init": 1.0,
    "learnable_sigma": False,
    "enforce_hermitian": True,
    "input_layout": "auto",
    "output_layout": "match_input",
}
CAFEPLUSFNO_FACTOR_COMMON_CONFIG: Final = {
    "factor_rff_basis": 16,
    "factor_cheb_basis": 8,
    "factor_branch_dim": 12,
    "factor_output_init_mode": "xavier",
    "factor_kernel_target_rms": 1e-3,
}
CAFEPLUSFNO_FACTORIZATIONS: Final = {
    "cafe_plus_fno": {
        "factorization": "dense",
        "rank": 16,
        "cafe_branch_dim": 16,
        "kernel_hidden_dim": 32,
    },
    "cp_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "cp", "rank": 8, "factor_hidden_dim": 16,
    },
    "tt_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "tt", "rank": 8, "factor_hidden_dim": 24,
    },
    "tucker_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "tucker", "rank": 8, "factor_hidden_dim": 16,
    },
}

PUBLISHED_PARAMETER_COUNTS_APPROX: Final = {
    "fno": 4_469_900,
    "ufno": 991_000,
    "tfno_cp": 237_800,
    "amfno": 385_600,
    "sirenfno": 304_800,
    "cpsirenfno": 64_000,
    "ttsirenfno": 92_700,
    "tuckersirenfno": 96_800,
}
EXPECTED_PINNED_PARAMETER_COUNTS: Final = {
    "fno": 4_469_857,
    "ufno": 991_009,
    "tfno_cp": 237_761,
    "amfno": 385_601,
    "sirenfno": 304_769,
    "cpsirenfno": 64_001,
    "ttsirenfno": 92_673,
    "tuckersirenfno": 96_769,
}
EXPECTED_CAFE_PARAMETER_COUNTS: Final = {
    "cafe_plus_fno": 327_169,
    "cp_cafe_plus_fno": 80_645,
    "tt_cafe_plus_fno": 108_421,
    "tucker_cafe_plus_fno": 113_413,
}


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent copy of one audited CFD-2D constructor."""

    if model_name == "fno":
        return dict(FNO_CONFIG)
    if model_name == "tfno_cp":
        return dict(TFNO_CP_CONFIG)
    if model_name == "amfno":
        return dict(AMFNO_CONFIG)
    if model_name == "ufno":
        return dict(UFNO_CONFIG)
    if model_name in SIRENFNO_FACTORIZATIONS:
        return {**SIRENFNO_COMMON_CONFIG, **SIRENFNO_FACTORIZATIONS[model_name]}
    if model_name in CAFEPLUSFNO_FACTORIZATIONS:
        return {**CAFEPLUSFNO_COMMON_CONFIG, **CAFEPLUSFNO_FACTORIZATIONS[model_name]}
    raise KeyError(f"Unsupported CFD-2D model: {model_name}")


def model_configuration_provenance(model_name: str) -> dict[str, object]:
    """Describe the released-runtime architecture policy."""

    return {
        "policy": "pinned_released_cfd2d_vx_constructor",
        "paper_table_3_parameter_parity": model_name in EXPECTED_PINNED_PARAMETER_COUNTS,
    }
