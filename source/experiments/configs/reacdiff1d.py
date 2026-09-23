"""Canonical released-runtime SirenFNO Reaction-Diffusion experiment."""

from typing import Final


DATASET_ID: Final = "reacdiff1024"
EXPERIMENT_ID: Final = "reacdiff1024_released_sirenfno"
DATASET_NAME: Final = "PDEBench Reaction-Diffusion 1D"
DATASET_SOURCE: Final = (
    "https://darus.uni-stuttgart.de/api/access/datafile/133177"
)
DATASET_FILENAME: Final = "ReacDiff_Nu0.5_Rho1.0.hdf5"
RESOLUTION: Final = 1024
EXPECTED_TIME_STEPS: Final = 101
AUTHOR_COMMENTED_TIME_STEPS: Final = 201
TEMPORAL_PROVENANCE_NOTE: Final = (
    "The pinned source comment says 201 snapshots, while the verified public "
    "artifact contains 101. The released 10-to-10 window is applied directly "
    "to those 101 snapshots without interpolation or decimation."
)
REDUCE_X: Final = 1
REDUCE_T: Final = 1
PHYSICAL_CHANNELS: Final = 1
INPUT_STEPS: Final = 10
ROLLOUT: Final = 10
INPUT_DIM: Final = INPUT_STEPS * PHYSICAL_CHANNELS
OUTPUT_DIM: Final = PHYSICAL_CHANNELS
FIELD_DIM: Final = PHYSICAL_CHANNELS
N_TRAIN: Final = 1000
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
SPLIT_POLICY: Final = "first_1000_train_next_200_test_no_random_split"
NORMALIZATION_POLICY: Final = "none"

REACDIFF_SOURCE_BLOBS: Final = {
    "train_ReacDiff.py": "4042c2652a003d6a086506b33667d6dcdddda6ac",
    "utils.py": "240a272db4d1b3f96b7fa4023a581c3876613d9b",
    "SirenFNO1D.py": "e05c7fce2c72d206345acc9754615b0eb393ad3a",
    "baseline/AMFNO.py": "70b2affe7237e0123ef0baa0bd970b62ec87a020",
    "baseline/UFNO.py": "594dfbc7147ffe737930eaa12e12330c085448d3",
    "neuralop/models/fno.py": "64628c081b80bdc6b3b79b3a90440e4f13fa60fe",
    "neuralop/losses/data_losses.py": (
        "a2c05e59f94b5c7720ead7c5ec54793251280706"
    ),
    "neuralop/utils.py": "f4fa635f4726b98daefcca71d580a2beb6fd4594",
    "neuralop/layers/spectral_convolution.py": (
        "391ca34cc24fcbc70012c0d2070d58b16894bb6a"
    ),
    "neuralop/layers/fno_block.py": (
        "04f9f4813594ab67fbe32ed146ea47d0a765a415"
    ),
    "neuralop/layers/channel_mlp.py": (
        "efbb740923b1397eee0494f90fed6defe9d32679"
    ),
    "neuralop/layers/embeddings.py": (
        "5b187b521446c1d65c6103fac1b918123a99d3ce"
    ),
    "neuralop/layers/padding.py": (
        "448b0928a9cf2c1f0c945449fd5c79516c4e7dd3"
    ),
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

EXPECTED_CAFE_PARAMETER_COUNTS: Final = {
    "cafe_plus_fno": 323_201,
    "cp_cafe_plus_fno": 70_149,
    "tt_cafe_plus_fno": 85_381,
    "tucker_cafe_plus_fno": 74_245,
}


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent copy of one audited ReacDiff constructor."""

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
    raise KeyError(f"Unsupported ReacDiff model: {model_name}")


def model_configuration_provenance(model_name: str) -> dict[str, object]:
    """Describe released baseline and precommitted CAFE policies."""

    return {
        "policy": (
            "pinned_released_reacdiff_constructor"
            if model_name in EXPECTED_PINNED_PARAMETER_COUNTS
            else "precommitted_validated_cafe_1d_policy"
        ),
        "published_parameter_count_used_only_as_audit_anchor": (
            model_name in EXPECTED_PINNED_PARAMETER_COUNTS
        ),
    }
