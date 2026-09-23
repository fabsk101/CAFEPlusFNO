"""Canonical PDEBench Burgers-1D paper-experiment configuration.

The data, optimization, rollout, and released baseline settings are transcribed
from the pinned SirenFNO ``train_Burgers.py``.  In particular, AM-FNO uses the
released width 64 constructor. U-FNO retains the released constructor because
no unique published alternative is recoverable from public provenance.
"""

from typing import Final


DATASET_ID: Final = "burgers1d"
DATASET_NAME: Final = "PDEBench 1D Burgers"
DATASET_SOURCE: Final = (
    "https://darus.uni-stuttgart.de/api/access/datafile/268190"
)
DATASET_FILENAME: Final = "1D_Burgers_Sols_Nu0.001.hdf5"
RESOLUTION: Final = 1024
EXPECTED_TIME_STEPS: Final = 201
REDUCE_X: Final = 1
REDUCE_T: Final = 1
INPUT_STEPS: Final = 10
ROLLOUT: Final = 10
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
CANONICAL_SEED: Final = 42
SEED_POLICY: Final = (
    "repository_multi_seed_harness_single_global_seed_before_loader_and_model"
)
SELECTION_POLICY: Final = "fixed_final_epoch_no_test_selection"
SPLIT_POLICY: Final = "first_1000_train_next_200_test_no_random_split"
NORMALIZATION_POLICY: Final = (
    "global_mean_std_from_torch_randperm_subset_of_at_most_200_train_trajectories"
)
MAX_NORMALIZATION_TRAJECTORIES: Final = 200

BURGERS_SOURCE_BLOBS: Final = {
    "SirenFNO1D.py": "e05c7fce2c72d206345acc9754615b0eb393ad3a",
    "neuralop/utils.py": "f4fa635f4726b98daefcca71d580a2beb6fd4594",
    "train_Burgers.py": "099d8f7d886de62aa1911af4fb0cfbe8e867fdc6",
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
    "in_channels": INPUT_STEPS,
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

# The pinned public Burgers source hard-codes width=64. Existing width=32
# checkpoints predate this restored baseline and must not be relabelled.
AMFNO_CONFIG: Final = {
    "width": 64,
    "n1": 10,
    "padding": 0,
    "input_dim": INPUT_STEPS,
    "output_dim": 1,
    "mlp_dropout": 0,
}

# This deliberately retains the released pinned constructor.  It produces
# 1,883,073 parameters and is not claimed to reproduce Table 3's 4,258.5k.
UFNO_CONFIG: Final = {
    "n_modes_height": 32,
    "hidden_channels": 64,
    "in_channels": INPUT_STEPS,
    "out_channels": 1,
    "n_layers": 6,
    "lifting_channels": 64,
    "projection_channels": 64,
    "positional_embedding": "grid",
}

SIRENFNO_COMMON_CONFIG: Final = {
    "width": 32,
    "padding": 0,
    "input_dim": INPUT_STEPS,
    "output_dim": 1,
    "mlp_dropout": 0.0,
    "add_grid": True,
    "siren_dim_in": 32,
    "hidden_dim": 32,
    "omega": 30.0,
    "n_hidden": 1,
    "ff_sigma": 1024,
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
    "input_dim": INPUT_STEPS,
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
        "rank": 8,
        "factor_hidden_dim": 16,
    },
}

PUBLISHED_PARAMETER_COUNTS_APPROX: Final = {
    "fno": 4_216_200,
    "ufno": 4_258_500,
    "tfno_cp": 226_400,
    "amfno": 207_600,
    "sirenfno": 308_900,
    "cpsirenfno": 70_100,
    "ttsirenfno": 84_500,
    "tuckersirenfno": 74_200,
}

EXPECTED_PINNED_PARAMETER_COUNTS: Final = {
    "fno": 4_216_161,
    "ufno": 1_883_073,
    "tfno_cp": 226_369,
    "amfno": 823_073,
    "sirenfno": 308_993,
    "cpsirenfno": 70_145,
    "ttsirenfno": 84_481,
    "tuckersirenfno": 74_241,
}

AMFNO_PROVENANCE_DISCREPANCY: Final = (
    "The pinned public Burgers source uses AM-FNO width 64. This experiment "
    "restores that released constructor for source parity; the resulting "
    "823,073-parameter run is distinct from prior width 32 checkpoints and "
    "does not claim to reproduce the paper's approximate Table 3 count."
)

UFNO_PROVENANCE_DISCREPANCY: Final = (
    "The released Burgers U-FNO constructor at the pinned SirenFNO commit "
    "does not reproduce the Burgers U-FNO parameter count reported in Table "
    "3. Because the published configuration cannot be uniquely reconstructed "
    "from public artifacts, this experiment uses the released constructor "
    "without reverse-engineering an undocumented alternative."
)


def model_constructor_kwargs(model_name: str) -> dict[str, object]:
    """Return an independent copy of one audited Burgers constructor."""

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
    raise KeyError(f"Unsupported Burgers model: {model_name}")


def model_configuration_provenance(model_name: str) -> dict[str, object]:
    """Return public provenance for resolved constructor discrepancies."""

    if model_name == "amfno":
        return {
            "policy": "pinned_released_constructor",
            "released_constructor_width": 64,
            "resolved_constructor_width": 64,
            "published_parameter_count_approx": 207_600,
            "resolved_parameter_count": 823_073,
            "table_3_architecture_parity_claimed": False,
            "prior_width_32_checkpoints_compatible": False,
            "note": AMFNO_PROVENANCE_DISCREPANCY,
        }
    if model_name == "ufno":
        return {
            "policy": "pinned_released_constructor",
            "published_parameter_count_approx": 4_258_500,
            "released_parameter_count": 1_883_073,
            "table_3_architecture_parity_claimed": False,
            "note": UFNO_PROVENANCE_DISCREPANCY,
        }
    return {"policy": "pinned_released_constructor"}
