"""Canonical Darcy Flow 128x128 paper-experiment configuration.

Values in this module are transcribed from the SirenFNO-author
``train_Darcy128.py`` reference and retain its constructor keyword spelling.
"""

from typing import Final

DATASET_NAME: Final = "Darcy Flow"
DATASET_SOURCE: Final = "NeuralOperator"
RESOLUTION: Final = 128
N_TRAIN: Final = 1000
N_TEST: Final = 200
BATCH_SIZE: Final = 32
TEST_BATCH_SIZE: Final = 32

EPOCHS: Final = 500
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4
SCHEDULER_T_MAX: Final = 500
EVAL_INTERVAL: Final = 1
CANONICAL_SEED: Final = 42
SELECTION_POLICY: Final = "fixed_final_epoch_no_test_selection"

MODEL_CHOICES: Final = (
    "fno",
    "ufno",
    "tfno_cp",
    "amfno",
    "sirenfno",
    "cpsirenfno",
    "ttsirenfno",
    "tuckersirenfno",
    "cafe_plus_fno",
    "cp_cafe_plus_fno",
    "tt_cafe_plus_fno",
    "tucker_cafe_plus_fno",
)

FNO_CONFIG: Final = {
    "n_modes": (16, 16),
    "hidden_channels": 32,
    "in_channels": 1,
    "out_channels": 1,
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
    "input_dim": 1,
    "output_dim": 1,
    "mlp_dropout": 0,
}

UFNO_CONFIG: Final = {
    "n_modes": (12, 12),
    "in_channels": 1,
    "out_channels": 1,
    "hidden_channels": 64,
    "n_layers": 6,
    "positional_embedding": "grid",
    "use_channel_mlp": True,
    "use_unet_from": 3,
    "unet_dropout": 0,
    "domain_padding": None,
    "fno_block_precision": "full",
}

SIRENFNO_COMMON_CONFIG: Final = {
    "width": 32,
    "input_dim": 1,
    "output_dim": 1,
    "add_grid": True,
    "padding": 0,
    "mlp_dropout": 0.0,
    "hidden_dim": 32,
    "omega": 30.0,
    "n_hidden": 1,
    "siren_dim_in": 32,
    "ff_sigma": 512.0,
    "learnable_ff": True,
}

SIRENFNO_FACTORIZATIONS: Final = {
    "sirenfno": {"factorization": "dense"},
    "cpsirenfno": {"factorization": "cp", "rank": 8},
    "ttsirenfno": {"factorization": "tt", "rank": 8},
    "tuckersirenfno": {"factorization": "tucker", "rank": 10},
}

CAFEPLUSFNO_COMMON_CONFIG: Final = {
    "width": 32,
    "input_dim": 1,
    "output_dim": 1,
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
        "cafe_branch_dim": 32,
        "kernel_hidden_dim": 32,
    },
    "cp_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "cp",
        "rank": 8,
        "factor_hidden_dim": 16,
    },
    "tt_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "tt",
        "rank": 8,
        "factor_hidden_dim": 24,
    },
    "tucker_cafe_plus_fno": {
        **CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
        "factorization": "tucker",
        "rank": 10,
        "factor_hidden_dim": 16,
    },
}


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent copy of the canonical constructor arguments."""

    if model_name == "fno":
        return dict(FNO_CONFIG)
    if model_name == "tfno_cp":
        return dict(TFNO_CP_CONFIG)
    if model_name == "amfno":
        return dict(AMFNO_CONFIG)
    if model_name == "ufno":
        return dict(UFNO_CONFIG)
    if model_name in SIRENFNO_FACTORIZATIONS:
        return {
            **SIRENFNO_COMMON_CONFIG,
            **SIRENFNO_FACTORIZATIONS[model_name],
        }
    if model_name in CAFEPLUSFNO_FACTORIZATIONS:
        return {
            **CAFEPLUSFNO_COMMON_CONFIG,
            **CAFEPLUSFNO_FACTORIZATIONS[model_name],
        }
    raise KeyError(f"Unsupported Darcy model: {model_name}")
