"""Precommitted unified Airfoil experiment configuration.

Airfoil was not evaluated by SirenFNO. The Siren variants below therefore use
an explicit researcher-precommitted equal-capacity configuration, not an
author-provided Airfoil configuration. No validation or test result was used
to choose the SIREN capacity or factorization ranks.
"""

from typing import Final


DATASET_ID: Final = "airfoil221x51"
EXPERIMENT_ID: Final = "airfoil221x51_unified_structured_rerun"
DATASET_NAME: Final = "Geo-FNO NACA Airfoil Euler structured benchmark"
DATASET_SOURCE: Final = (
    "https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8"
)
DATASET_FILENAMES: Final = (
    "NACA_Cylinder_X.npy",
    "NACA_Cylinder_Y.npy",
    "NACA_Cylinder_Q.npy",
)
RESOLUTION: Final = (221, 51)
RAW_SAMPLE_COUNT: Final = 2490
RAW_SOLUTION_CHANNELS: Final = 5
TARGET_FIELD_INDEX: Final = 4
INPUT_DIM: Final = 2
OUTPUT_DIM: Final = 1
N_TRAIN: Final = 1000
N_VAL: Final = 0
N_TEST: Final = 200

BATCH_SIZE: Final = 8
NUM_WORKERS: Final = 0
PIN_MEMORY: Final = False
TRAIN_DROP_LAST: Final = False
EVAL_DROP_LAST: Final = False

EPOCHS: Final = 500
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4
SCHEDULER_T_MAX_POLICY: Final = "epochs_times_train_loader_length"
SCHEDULER_STEP_POLICY: Final = (
    "after_every_optimizer_step_plus_one_additional_step_at_epoch_end"
)
EXPECTED_TRAIN_BATCHES_PER_EPOCH: Final = 125
EXPECTED_SCHEDULER_T_MAX: Final = 62_500
EVAL_INTERVAL: Final = 1
USE_AMP: Final = False
MAIN_REPORTING_METRIC: Final = "final_test_relative_l2"
CANONICAL_SEED: Final = 42
PAPER_SEEDS: Final = (0, 42, 73, 108, 202)
SEED_POLICY: Final = (
    "repository_multi_seed_harness_single_global_seed_before_loader_and_model"
)
SELECTION_POLICY: Final = "fixed_final_epoch_no_test_selection"
SPLIT_POLICY: Final = "first_1000_train_next_200_test_no_random_split"
NORMALIZATION_POLICY: Final = "none"
DATASET_LOADING_MODE: Final = "verified_local_offline"

MODEL_CHOICES: Final = (
    "fno", "tfno_cp", "amfno",
    "sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno",
    "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
    "tucker_cafe_plus_fno",
)

AIRFOIL_EXCLUDED_BASELINES: Final = {
    "ufno": {
        "reason": (
            "The exact U-FNO implementation used for the published AM-FNO "
            "Airfoil result cannot be reproducibly reconstructed from the "
            "released artifacts because the referenced supplemental FNOs.py "
            "is absent. The pinned project U-FNO also requires an Airfoil-"
            "specific compatibility modification for the 221x51 shape under "
            "the common padding policy."
        ),
        "performance_based_exclusion": False,
        "excluded_before_paper_training": True,
        "published_result_imported": False,
    }
}

# Researcher-precommitted translation of the published ``modes=12`` Airfoil
# baseline setting to the pinned NeuralOperator n_modes API. The absent
# supplemental FNOs.py prevents a stronger exact-implementation claim.
EFFECTIVE_RETAINED_MODES: Final = (12, 12)
NEURALOP_N_MODES: Final = (24, 22)
PADDING_GRID_CELLS: Final = 8
NEURALOP_DOMAIN_PADDING: Final = [
    PADDING_GRID_CELLS / RESOLUTION[0],
    PADDING_GRID_CELLS / RESOLUTION[1],
]

FNO_CONFIG: Final = {
    "n_modes": NEURALOP_N_MODES,
    "hidden_channels": 32,
    "in_channels": INPUT_DIM,
    "out_channels": OUTPUT_DIM,
    "n_layers": 4,
    "lifting_channel_ratio": 2,
    "projection_channel_ratio": 2,
    "positional_embedding": "grid",
    "domain_padding": NEURALOP_DOMAIN_PADDING,
    "domain_padding_mode": "one-sided",
}
TFNO_CP_CONFIG: Final = {
    **FNO_CONFIG,
    "implementation": "factorized",
    "factorization": "cp",
    "rank": 0.05,
}
AMFNO_CONFIG: Final = {
    "width": 32,
    "n1": 32,
    "n2": 32,
    "padding": PADDING_GRID_CELLS,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "mlp_dropout": 0.0,
    "add_grid": True,
}

SIRENFNO_AIRFOIL_COMMON_CONFIG: Final = {
    "width": 32,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "padding": PADDING_GRID_CELLS,
    "mlp_dropout": 0.0,
    "add_grid": True,
    "hidden_dim": 32,
    "omega": 30.0,
    "n_hidden": 1,
    "siren_dim_in": 32,
    "ff_sigma": 512.0,
    "learnable_ff": True,
}
SIRENFNO_COMMON_CONFIG: Final = SIRENFNO_AIRFOIL_COMMON_CONFIG
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
    "padding": PADDING_GRID_CELLS,
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

AIRFOIL_RUNTIME_SOURCE_BLOBS: Final = {
    "SirenFNO2D.py": "b0a987e42cd00547dfa64d0e4dfa619ee7106730",
    "baseline/AMFNO.py": "70b2affe7237e0123ef0baa0bd970b62ec87a020",
    "neuralop/losses/data_losses.py": "a2c05e59f94b5c7720ead7c5ec54793251280706",
    "neuralop/models/fno.py": "64628c081b80bdc6b3b79b3a90440e4f13fa60fe",
    "neuralop/utils.py": "f4fa635f4726b98daefcca71d580a2beb6fd4594",
}

EXPECTED_PINNED_PARAMETER_COUNTS: Final = {
    "fno": 2_372_513,
    "tfno_cp": 131_993,
    "amfno": 1_136_673,
    "sirenfno": 308_897,
    "cpsirenfno": 80_545,
    "ttsirenfno": 109_217,
    "tuckersirenfno": 113_313,
}
EXPECTED_CAFE_PARAMETER_COUNTS: Final = {
    "cafe_plus_fno": 345_633,
    "cp_cafe_plus_fno": 80_549,
    "tt_cafe_plus_fno": 108_325,
    "tucker_cafe_plus_fno": 113_317,
}


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent, explicit Airfoil constructor configuration."""

    if model_name == "fno":
        return {**FNO_CONFIG, "domain_padding": list(NEURALOP_DOMAIN_PADDING)}
    if model_name == "tfno_cp":
        return {**TFNO_CP_CONFIG, "domain_padding": list(NEURALOP_DOMAIN_PADDING)}
    if model_name == "amfno":
        return dict(AMFNO_CONFIG)
    if model_name in SIRENFNO_FACTORIZATIONS:
        return {
            **SIRENFNO_AIRFOIL_COMMON_CONFIG,
            **SIRENFNO_FACTORIZATIONS[model_name],
        }
    if model_name in CAFEPLUSFNO_FACTORIZATIONS:
        return {
            **CAFEPLUSFNO_COMMON_CONFIG,
            **CAFEPLUSFNO_FACTORIZATIONS[model_name],
        }
    raise KeyError(f"Unsupported Airfoil model: {model_name}")


def model_configuration_provenance(model_name: str) -> dict[str, object]:
    """Return explicit provenance without unsupported reproduction claims."""

    if model_name == "amfno":
        policy = "amfno_airfoil_author_protocol_on_pinned_sirenfno_implementation"
    elif model_name in {"fno", "tfno_cp"}:
        policy = "researcher_precommitted_unified_airfoil_baseline"
    elif model_name in SIRENFNO_FACTORIZATIONS:
        policy = "researcher_precommitted_equal_capacity_airfoil_siren"
    else:
        policy = "researcher_precommitted_cafe_airfoil"
    return {
        "policy": policy,
        "sirenfno_author_airfoil_configuration": False,
        "configuration_selected_before_airfoil_results": True,
        "validation_or_test_used_for_configuration": False,
        "amfno_table_2_numerical_reproduction_claimed": False,
        "airfoil_protocol_source": "audited_amfno_supplemental_airfoil_source",
        "runtime_model_source": "pinned_sirenfno_revision",
        "f_fno_architecture_parity_claimed": False,
        "effective_retained_modes": EFFECTIVE_RETAINED_MODES,
        "padding_grid_cells": PADDING_GRID_CELLS,
    }
