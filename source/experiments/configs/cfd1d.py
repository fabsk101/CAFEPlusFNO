"""Canonical released-runtime SirenFNO CFD-1D experiment configuration.

The pinned author runtime selects only the ``Vx`` dataset from the public
PDEBench HDF5 artifact. Consequently this experiment is a scalar 10-to-1
autoregressive task; density and pressure are deliberately not concatenated.
"""

from typing import Final


DATASET_ID: Final = "cfd1d1024"
EXPERIMENT_ID: Final = "cfd1d1024_released_sirenfno_vx"
DATASET_NAME: Final = "PDEBench 1D CFD (released SirenFNO Vx field)"
DATASET_SOURCE: Final = (
    "https://darus.uni-stuttgart.de/api/access/datafile/164672"
)
DATASET_FILENAME: Final = (
    "1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5"
)
SELECTED_FIELD: Final = "Vx"
DATASET_FIELD_POLICY: Final = "released_sirenfno_single_field"
AUTHOR_SOURCE_NOTE: Final = (
    "Released SirenFNO runtime selects one physical field; density and "
    "pressure are not concatenated."
)
TEMPORAL_PROVENANCE_NOTE: Final = (
    "The pinned source comment refers to 201 snapshots, while the verified "
    "public Vx artifact contains 101 field snapshots. The released 10-to-10 "
    "window is applied to the 101 snapshots without interpolation."
)
RESOLUTION: Final = 1024
EXPECTED_TIME_STEPS: Final = 101
AUTHOR_COMMENTED_TIME_STEPS: Final = 201
REDUCE_X: Final = 1
REDUCE_T: Final = 1
PHYSICAL_CHANNELS: Final = 1
INPUT_STEPS: Final = 10
ROLLOUT: Final = 10
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

CFD1D_SOURCE_BLOBS: Final = {
    "SirenFNO1D.py": "e05c7fce2c72d206345acc9754615b0eb393ad3a",
    "neuralop/utils.py": "f4fa635f4726b98daefcca71d580a2beb6fd4594",
    "train_CFD.py": "5656a35914271031ec2aee600d658daedc3f7be5",
    "utils.py": "240a272db4d1b3f96b7fa4023a581c3876613d9b",
}

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
    "n_modes_height": 1024,
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
    "width": 64,
    "n1": 10,
    "padding": 0,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "mlp_dropout": 0,
}

UFNO_CONFIG: Final = {
    "n_modes_height": 32,
    "hidden_channels": 64,
    "in_channels": INPUT_DIM,
    "out_channels": OUTPUT_DIM,
    "n_layers": 6,
    "lifting_channels": 128,
    "projection_channels": 128,
    "positional_embedding": "grid",
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
    "omega": 15.0,
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
        "rank": 8,
        "factor_hidden_dim": 16,
    },
}

PUBLISHED_PARAMETER_COUNTS_APPROX: Final = {
    "fno": 4_216_200,
    "ufno": 1_892_200,
    "tfno_cp": 226_400,
    "amfno": 823_100,
    "sirenfno": 304_800,
    "cpsirenfno": 57_700,
    "ttsirenfno": 72_000,
    "tuckersirenfno": 61_800,
}

EXPECTED_PINNED_PARAMETER_COUNTS: Final = {
    "fno": 4_216_161,
    "ufno": 1_892_161,
    "tfno_cp": 226_369,
    "amfno": 823_073,
    "sirenfno": 304_833,
    "cpsirenfno": 57_665,
    "ttsirenfno": 72_001,
    "tuckersirenfno": 61_761,
}


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent copy of one audited CFD-1D constructor."""

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
    raise KeyError(f"Unsupported CFD-1D model: {model_name}")


def model_configuration_provenance(model_name: str) -> dict[str, object]:
    """Describe the released-runtime architecture policy."""

    return {
        "policy": "pinned_released_cfd1d_vx_constructor",
        "paper_table_3_parameter_parity": (
            model_name in EXPECTED_PINNED_PARAMETER_COUNTS
        ),
    }
