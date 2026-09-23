"""Run each paper model/seed in an independent Python process."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.configs import (  # noqa: E402
    airfoil,
    burgers1d,
    cfd1d,
    cfd2d,
    darcy,
    ns2d,
    reacdiff1d,
)
from experiments.configs.seeds import PAPER_SEEDS  # noqa: E402
from experiments.common.checkpoints import load_weights_only_checkpoint  # noqa: E402
from experiments.common.dataset_provenance import dataset_manifest  # noqa: E402
from experiments.common.sirenfno_backend import (  # noqa: E402
    source_repository_state,
    upstream_manifest,
)


COMPLETION_SCHEMA_VERSION = "paper_run_completion_v1"
EVALUATION_SCHEMA_VERSION = "paper_evaluation_record_v2"
CORRECTED_RELATIVE_L2_VERSION = "corrected_relative_l2_v1"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_COMPATIBILITY_PATHS = frozenset(
    {
        "README.md",
        "experiments/common/checkpoints.py",
        "scripts/run_paper_sweep.py",
        "scripts/summarize_results.py",
        "tests/test_results.py",
        "tests/test_sweep.py",
    }
)


class CompletionStatus(str, Enum):
    """Machine-readable outcome of one paper-run audit."""

    COMPLETE = "complete"
    COMPLETE_REEVALUATED = "complete_reevaluated"
    LEGACY_EVALUATION = "legacy_evaluation"
    INVALID_EVALUATION = "invalid_evaluation"
    STALE_TRAINING_SOURCE = "stale_training_source"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class RunCompletionAudit:
    """Detailed result shared by sweep, re-evaluation, and summarization code."""

    status: CompletionStatus
    reasons: tuple[str, ...]
    training_valid: bool
    evaluation_valid: bool

    @property
    def complete(self) -> bool:
        return self.status in {
            CompletionStatus.COMPLETE,
            CompletionStatus.COMPLETE_REEVALUATED,
        }

    def describe(self) -> str:
        return "; ".join(self.reasons) if self.reasons else self.status.value


@dataclass(frozen=True)
class ValidatedTrainingArtifacts:
    """Loaded artifacts returned only after the shared training audit passes."""

    summary: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    model: str
    seed: int
    experiment: str
    training_protocol_version: str
    training_protocol_identifier: str
    training_configuration: Mapping[str, Any]
    data_configuration: Mapping[str, Any]
    model_configuration: Mapping[str, Any]
    dataset_identity: Mapping[str, Any]
    checkpoint_sha256: str
    checkpoint_load_metadata: Mapping[str, Any]
    training_source_commit: str
    training_source_dirty: bool
    training_source_reuse: Mapping[str, Any]
    training_log_identity: Mapping[str, Any]
    validation_status: str
    evaluation_classification: str
    original_evaluation_protocol_version: str | None


class TrainingArtifactValidationError(RuntimeError):
    """Raised when a run cannot safely be used as a re-evaluation input."""

    def __init__(self, audit: RunCompletionAudit) -> None:
        self.audit = audit
        super().__init__(audit.describe())


def historical_source_compatibility_evidence(
    historical_commit: str,
    current_commit: str,
) -> dict[str, Any]:
    """Verify that an exact source transition is limited to compatibility code."""

    if (
        _COMMIT_PATTERN.fullmatch(historical_commit) is None
        or _COMMIT_PATTERN.fullmatch(current_commit) is None
        or historical_commit == current_commit
    ):
        raise ValueError("Historical source comparison requires two exact commits.")
    command_prefix = [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-renames",
        historical_commit,
        current_commit,
        "--",
    ]
    names_result = subprocess.run(
        [*command_prefix[:2], "--name-only", *command_prefix[2:]],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    changed_paths = tuple(
        line.strip().replace("\\", "/")
        for line in names_result.stdout.splitlines()
        if line.strip()
    )
    if not changed_paths or any(
        path not in _SOURCE_COMPATIBILITY_PATHS for path in changed_paths
    ):
        raise ValueError(
            "Source transition includes changes outside the compatibility scope."
        )
    diff_result = subprocess.run(
        [*command_prefix[:2], "--binary", *command_prefix[2:]],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    if not diff_result.stdout:
        raise ValueError("Historical source comparison produced an empty diff.")
    return {
        "verified_change_scope": "checkpoint_storage_loading_validation_only",
        "source_diff_sha256": hashlib.sha256(diff_result.stdout).hexdigest(),
        "changed_paths": list(changed_paths),
    }


def _training_source_reuse_metadata(
    *,
    training_source_commit: str,
    checkpoint_sha256: str,
    current_source_commit: str | None,
    allowed_historical_training_source_commit: str | None,
    allowed_historical_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if training_source_commit == current_source_commit:
        return {
            "policy": "current_source_commit",
            "aggregation_source_commit": training_source_commit,
        }
    if (
        training_source_commit != allowed_historical_training_source_commit
        or checkpoint_sha256 != allowed_historical_checkpoint_sha256
        or not isinstance(current_source_commit, str)
    ):
        raise ValueError("Historical source identity is not explicitly allowed.")
    return {
        "policy": "exact_historical_commit_and_checkpoint_sha256",
        "aggregation_source_commit": current_source_commit,
        "historical_training_source_commit": training_source_commit,
        "historical_checkpoint_sha256": checkpoint_sha256,
        **historical_source_compatibility_evidence(
            training_source_commit,
            current_source_commit,
        ),
    }


def _training_configuration(
    *,
    epochs: int,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    scheduler_t_max: int,
    eval_interval: int,
    training_loss: str,
    selection_policy: str,
    scheduler_step_policy: str = "exactly_once_after_each_epoch_train_loop",
    amp_enabled: bool = False,
    test_batch_size: int | None = None,
    train_loader_shuffle: bool = True,
    train_loader_drop_last: bool = False,
    evaluation_loader_shuffle: bool = False,
    evaluation_loader_drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    **extra_configuration: Any,
) -> dict[str, Any]:
    configuration = {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_batch_size": batch_size if test_batch_size is None else test_batch_size,
        "train_loader_shuffle": train_loader_shuffle,
        "train_loader_drop_last": train_loader_drop_last,
        "evaluation_loader_shuffle": evaluation_loader_shuffle,
        "evaluation_loader_drop_last": evaluation_loader_drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": scheduler_t_max,
        "scheduler_step_policy": scheduler_step_policy,
        "eval_interval": eval_interval,
        "amp_enabled": amp_enabled,
        "training_loss": training_loss,
        "checkpoint_selection_policy": selection_policy,
    }
    configuration.update(extra_configuration)
    return configuration


def _data_configuration(
    *,
    dataset_id: str,
    resolution: Any,
    n_train: int,
    n_test: int,
    split_policy: str,
    preprocessing_protocol: str,
    normalization_policy: str,
    input_horizon: int,
    prediction_horizon: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "resolution": resolution,
        "n_train": n_train,
        "n_test": n_test,
        "split_policy": split_policy,
        "preprocessing_protocol": preprocessing_protocol,
        "normalization_policy": normalization_policy,
        "input_horizon": input_horizon,
        "prediction_horizon": prediction_horizon,
    }


_STATIC_EVALUATION_METADATA = {
    "evaluation_horizon": 1,
    "evaluation_sample_count": {"train": 1000, "test": 200},
}


_CORRECTED_EVALUATION_METADATA = {
    "evaluation_epsilon": 1e-12,
    "evaluation_tensor_layout": "B,T,C,*spatial",
    "evaluation_space": "physical_source_values_no_normalization",
}


DATASETS = {
    "airfoil221x51": {
        "module": "experiments.train_airfoil",
        "dataset_id": airfoil.DATASET_ID,
        "experiment": airfoil.EXPERIMENT_ID,
        "models": airfoil.MODEL_CHOICES,
        "epochs": airfoil.EPOCHS,
        "training_protocol_version": "airfoil_training_v1",
        "evaluation_protocol_version": "airfoil_sample_relative_l2_v1",
        "legacy_evaluation_protocol_versions": (),
        "model_configuration_factory": airfoil.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=airfoil.EPOCHS,
            batch_size=airfoil.BATCH_SIZE,
            optimizer="torch.optim.AdamW",
            learning_rate=airfoil.LEARNING_RATE,
            weight_decay=airfoil.WEIGHT_DECAY,
            scheduler_t_max=airfoil.EXPECTED_SCHEDULER_T_MAX,
            scheduler_step_policy=airfoil.SCHEDULER_STEP_POLICY,
            eval_interval=airfoil.EVAL_INTERVAL,
            amp_enabled=airfoil.USE_AMP,
            train_loader_drop_last=airfoil.TRAIN_DROP_LAST,
            evaluation_loader_drop_last=airfoil.EVAL_DROP_LAST,
            num_workers=airfoil.NUM_WORKERS,
            pin_memory=airfoil.PIN_MEMORY,
            seed_policy=airfoil.SEED_POLICY,
            training_loss="relative_L2_sum_divided_by_sample_count",
            selection_policy=airfoil.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id=airfoil.DATASET_ID,
            resolution=airfoil.RESOLUTION,
            n_train=airfoil.N_TRAIN,
            n_test=airfoil.N_TEST,
            split_policy=airfoil.SPLIT_POLICY,
            preprocessing_protocol="verified_airfoil_structured_numpy_v1",
            normalization_policy=airfoil.NORMALIZATION_POLICY,
            input_horizon=1,
            prediction_horizon=1,
        ),
        "evaluation_metadata": dict(_STATIC_EVALUATION_METADATA),
        "metric_columns": {
            "final_train_relative_l2": "train_relative_l2",
            "final_test_relative_l2": "test_relative_l2",
        },
        "paper_metric_columns": {
            "final_test_relative_l2": "test_relative_l2"
        },
        "record_metric_names": ("relative_l2",),
        "reevaluation_space": "physical_source_values_no_normalization",
        "require_current_source_commit": True,
        "default_results": Path("results")
        / "airfoil221x51_sirenfno_81918ec",
    },
    "burgers1024": {
        "module": "experiments.train_burgers",
        "dataset_id": burgers1d.DATASET_ID,
        "experiment": "burgers1024_released_sirenfno",
        "models": burgers1d.MODEL_CHOICES,
        "epochs": burgers1d.EPOCHS,
        "training_protocol_version": "burgers1024_training_v2_amfno64_roster",
        "evaluation_protocol_version": "burgers_rollout_relative_l2_v1",
        "legacy_evaluation_protocol_versions": (),
        "model_configuration_factory": burgers1d.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=burgers1d.EPOCHS,
            batch_size=burgers1d.BATCH_SIZE,
            optimizer="torch.optim.AdamW",
            learning_rate=burgers1d.LEARNING_RATE,
            weight_decay=burgers1d.WEIGHT_DECAY,
            scheduler_t_max=burgers1d.SCHEDULER_T_MAX,
            eval_interval=burgers1d.EVAL_INTERVAL,
            amp_enabled=burgers1d.USE_AMP,
            train_loader_drop_last=burgers1d.TRAIN_DROP_LAST,
            evaluation_loader_drop_last=burgers1d.EVAL_DROP_LAST,
            num_workers=burgers1d.NUM_WORKERS,
            pin_memory=burgers1d.PIN_MEMORY,
            pushforward_detach=burgers1d.PUSHFORWARD_DETACH,
            seed_policy=burgers1d.SEED_POLICY,
            training_loss="rollout_loss_rel_l2",
            selection_policy=burgers1d.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id=burgers1d.DATASET_ID,
            resolution=burgers1d.RESOLUTION,
            n_train=burgers1d.N_TRAIN,
            n_test=burgers1d.N_TEST,
            split_policy=burgers1d.SPLIT_POLICY,
            preprocessing_protocol="pinned_sirenfno_burgers_rollout_v1",
            normalization_policy=burgers1d.NORMALIZATION_POLICY,
            input_horizon=burgers1d.INPUT_STEPS,
            prediction_horizon=burgers1d.ROLLOUT,
        ),
        "evaluation_metadata": {
            "evaluation_horizon": burgers1d.ROLLOUT,
            "evaluation_sample_count": {
                "train": burgers1d.N_TRAIN,
                "test": burgers1d.N_TEST,
            },
        },
        "metric_columns": {
            "final_train_step_relative_l2": "train_step_relative_l2",
            "final_train_full_relative_l2": "train_full_relative_l2",
            "final_test_step_relative_l2": "test_step_relative_l2",
            "final_test_full_relative_l2": "test_full_relative_l2",
        },
        "paper_metric_columns": {
            "final_test_step_relative_l2": "test_step_relative_l2",
            "final_test_full_relative_l2": "test_full_relative_l2",
        },
        "record_metric_names": ("step_relative_l2", "trajectory_relative_l2"),
        "reevaluation_space": "train_normalized_values",
        "require_current_source_commit": True,
        "default_results": Path("results")
        / "burgers1024_sirenfno_81918ec_protocol_v2",
    },
    "cfd1d1024": {
        "module": "experiments.train_cfd1d",
        "dataset_id": cfd1d.DATASET_ID,
        "experiment": cfd1d.EXPERIMENT_ID,
        "models": cfd1d.MODEL_CHOICES,
        "epochs": cfd1d.EPOCHS,
        "training_protocol_version": "cfd1d1024_training_v1",
        "evaluation_protocol_version": CORRECTED_RELATIVE_L2_VERSION,
        "legacy_evaluation_protocol_versions": (
            "legacy_sirenfno_batch_reduced_relative_l2_v0",
        ),
        "model_configuration_factory": cfd1d.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=cfd1d.EPOCHS,
            batch_size=cfd1d.BATCH_SIZE,
            optimizer="torch.optim.AdamW",
            learning_rate=cfd1d.LEARNING_RATE,
            weight_decay=cfd1d.WEIGHT_DECAY,
            scheduler_t_max=cfd1d.SCHEDULER_T_MAX,
            eval_interval=cfd1d.EVAL_INTERVAL,
            amp_enabled=cfd1d.USE_AMP,
            train_loader_drop_last=cfd1d.TRAIN_DROP_LAST,
            evaluation_loader_drop_last=cfd1d.EVAL_DROP_LAST,
            num_workers=cfd1d.NUM_WORKERS,
            pin_memory=cfd1d.PIN_MEMORY,
            pushforward_detach=cfd1d.PUSHFORWARD_DETACH,
            seed_policy=cfd1d.SEED_POLICY,
            training_loss="rollout_loss_lp_relative_l2",
            selection_policy=cfd1d.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id=cfd1d.DATASET_ID,
            resolution=cfd1d.RESOLUTION,
            n_train=cfd1d.N_TRAIN,
            n_test=cfd1d.N_TEST,
            split_policy=cfd1d.SPLIT_POLICY,
            preprocessing_protocol="pinned_sirenfno_cfd1d_vx_rollout_v1",
            normalization_policy=cfd1d.NORMALIZATION_POLICY,
            input_horizon=cfd1d.INPUT_STEPS,
            prediction_horizon=cfd1d.ROLLOUT,
        ),
        "evaluation_metadata": {
            **_CORRECTED_EVALUATION_METADATA,
            "evaluation_horizon": cfd1d.ROLLOUT,
            "evaluation_sample_count": {
                "train": cfd1d.N_TRAIN,
                "test": cfd1d.N_TEST,
            },
        },
        "metric_columns": {
            "final_train_corrected_step_relative_l2": (
                "train_corrected_step_relative_l2"
            ),
            "final_train_corrected_trajectory_relative_l2": (
                "train_corrected_trajectory_relative_l2"
            ),
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "paper_metric_columns": {
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "record_metric_names": (
            "corrected_step_relative_l2",
            "corrected_trajectory_relative_l2",
        ),
        "reevaluation_space": "physical_source_values_no_normalization",
        "require_current_source_commit": True,
        "default_results": Path("results")
        / "cfd1d1024_sirenfno_81918ec",
    },
    "cfd2d128": {
        "module": "experiments.train_cfd2d",
        "dataset_id": cfd2d.DATASET_ID,
        "experiment": cfd2d.EXPERIMENT_ID,
        "models": cfd2d.MODEL_CHOICES,
        "epochs": cfd2d.EPOCHS,
        "training_protocol_version": "cfd2d128_training_v1",
        "evaluation_protocol_version": CORRECTED_RELATIVE_L2_VERSION,
        "legacy_evaluation_protocol_versions": (
            "legacy_sirenfno_batch_reduced_relative_l2_v0",
        ),
        "model_configuration_factory": cfd2d.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=cfd2d.EPOCHS,
            batch_size=cfd2d.BATCH_SIZE,
            optimizer="torch.optim.AdamW",
            learning_rate=cfd2d.LEARNING_RATE,
            weight_decay=cfd2d.WEIGHT_DECAY,
            scheduler_t_max=cfd2d.SCHEDULER_T_MAX,
            eval_interval=cfd2d.EVAL_INTERVAL,
            amp_enabled=cfd2d.USE_AMP,
            train_loader_drop_last=cfd2d.TRAIN_DROP_LAST,
            evaluation_loader_drop_last=cfd2d.EVAL_DROP_LAST,
            num_workers=cfd2d.NUM_WORKERS,
            pin_memory=cfd2d.PIN_MEMORY,
            pushforward_detach=cfd2d.PUSHFORWARD_DETACH,
            seed_policy=cfd2d.SEED_POLICY,
            training_loss="rollout_loss_lp_relative_l2_require_lp_true",
            selection_policy=cfd2d.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id=cfd2d.DATASET_ID,
            resolution=cfd2d.RESOLUTION,
            n_train=cfd2d.N_TRAIN,
            n_test=cfd2d.N_TEST,
            split_policy=cfd2d.SPLIT_POLICY,
            preprocessing_protocol="pinned_sirenfno_cfd2d_vx_rollout_v1",
            normalization_policy=cfd2d.NORMALIZATION_POLICY,
            input_horizon=cfd2d.INPUT_STEPS,
            prediction_horizon=cfd2d.ROLLOUT,
        ),
        "evaluation_metadata": {
            **_CORRECTED_EVALUATION_METADATA,
            "evaluation_horizon": cfd2d.ROLLOUT,
            "evaluation_sample_count": {
                "train": cfd2d.N_TRAIN,
                "test": cfd2d.N_TEST,
            },
        },
        "metric_columns": {
            "final_train_corrected_step_relative_l2": (
                "train_corrected_step_relative_l2"
            ),
            "final_train_corrected_trajectory_relative_l2": (
                "train_corrected_trajectory_relative_l2"
            ),
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "paper_metric_columns": {
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "record_metric_names": (
            "corrected_step_relative_l2",
            "corrected_trajectory_relative_l2",
        ),
        "reevaluation_space": "physical_source_values_no_normalization",
        "require_current_source_commit": True,
        "default_results": Path("results")
        / "cfd2d128_sirenfno_81918ec",
    },
    "darcy128": {
        "module": "experiments.train_darcy",
        "dataset_id": "darcy128",
        "experiment": "darcy128_released_sirenfno",
        "models": darcy.MODEL_CHOICES,
        "epochs": darcy.EPOCHS,
        "training_protocol_version": "darcy128_training_v1",
        "evaluation_protocol_version": "darcy_decoded_relative_l2_h1_v1",
        "legacy_evaluation_protocol_versions": (),
        "model_configuration_factory": darcy.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=darcy.EPOCHS,
            batch_size=darcy.BATCH_SIZE,
            optimizer="NeuralOperator AdamW",
            learning_rate=darcy.LEARNING_RATE,
            weight_decay=darcy.WEIGHT_DECAY,
            scheduler_t_max=darcy.SCHEDULER_T_MAX,
            eval_interval=darcy.EVAL_INTERVAL,
            test_batch_size=darcy.TEST_BATCH_SIZE,
            train_loader_shuffle=False,
            pin_memory=True,
            training_loss="relative L2",
            selection_policy=darcy.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id="darcy128",
            resolution=darcy.RESOLUTION,
            n_train=darcy.N_TRAIN,
            n_test=darcy.N_TEST,
            split_policy="official_pre_split_train_and_test_pt_files",
            preprocessing_protocol="pinned_neuraloperator_load_darcy_pt_v1",
            normalization_policy="pinned_neuraloperator_data_processor",
            input_horizon=1,
            prediction_horizon=1,
        ),
        "evaluation_metadata": dict(_STATIC_EVALUATION_METADATA),
        "metric_columns": {
            "final_train_loss": "train_relative_l2",
            "final_test_relative_l2": "test_relative_l2",
            "final_test_h1": "test_h1",
        },
        "paper_metric_columns": {
            "final_test_relative_l2": "test_relative_l2",
            "final_test_h1": "test_h1",
        },
        "record_metric_names": ("relative_l2", "h1"),
        "reevaluation_space": "decoded_physical_values",
        "require_current_source_commit": True,
        "default_results": Path("results") / "darcy128_sirenfno_81918ec",
    },
    "ns128": {
        "module": "experiments.train_ns2d",
        "dataset_id": "ns128",
        "experiment": "ns128_released_sirenfno",
        "models": ns2d.MODEL_CHOICES,
        "epochs": ns2d.EPOCHS,
        "training_protocol_version": "ns128_training_v1",
        "evaluation_protocol_version": "ns_decoded_relative_l2_h1_v1",
        "legacy_evaluation_protocol_versions": (),
        "model_configuration_factory": ns2d.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=ns2d.EPOCHS,
            batch_size=ns2d.BATCH_SIZE,
            optimizer="NeuralOperator AdamW",
            learning_rate=ns2d.LEARNING_RATE,
            weight_decay=ns2d.WEIGHT_DECAY,
            scheduler_t_max=ns2d.SCHEDULER_T_MAX,
            eval_interval=ns2d.EVAL_INTERVAL,
            test_batch_size=ns2d.TEST_BATCH_SIZE,
            train_loader_shuffle=False,
            pin_memory=True,
            seed_policy=ns2d.SEED_POLICY,
            training_loss="relative L2",
            selection_policy=ns2d.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id="ns128",
            resolution=ns2d.RESOLUTION,
            n_train=ns2d.N_TRAIN,
            n_test=ns2d.N_TEST,
            split_policy="official_pre_split_train_and_test_pt_files",
            preprocessing_protocol="pinned_neuraloperator_navier_stokes_v1",
            normalization_policy="pinned_neuraloperator_data_processor",
            input_horizon=1,
            prediction_horizon=1,
        ),
        "evaluation_metadata": dict(_STATIC_EVALUATION_METADATA),
        "metric_columns": {
            "final_train_loss": "train_relative_l2",
            "final_test_relative_l2": "test_relative_l2",
            "final_test_h1": "test_h1",
        },
        "paper_metric_columns": {
            "final_test_relative_l2": "test_relative_l2",
            "final_test_h1": "test_h1",
        },
        "record_metric_names": ("relative_l2", "h1"),
        "reevaluation_space": "decoded_physical_values",
        "require_current_source_commit": True,
        "default_results": Path("results") / "ns128_sirenfno_81918ec",
    },
    "reacdiff1024": {
        "module": "experiments.train_reacdiff",
        "dataset_id": reacdiff1d.DATASET_ID,
        "experiment": reacdiff1d.EXPERIMENT_ID,
        "models": reacdiff1d.MODEL_CHOICES,
        "epochs": reacdiff1d.EPOCHS,
        "training_protocol_version": "reacdiff1024_training_v1",
        "evaluation_protocol_version": CORRECTED_RELATIVE_L2_VERSION,
        "legacy_evaluation_protocol_versions": (
            "legacy_sirenfno_batch_reduced_relative_l2_v0",
        ),
        "model_configuration_factory": reacdiff1d.model_constructor_kwargs,
        "training_configuration": _training_configuration(
            epochs=reacdiff1d.EPOCHS,
            batch_size=reacdiff1d.BATCH_SIZE,
            optimizer="torch.optim.AdamW",
            learning_rate=reacdiff1d.LEARNING_RATE,
            weight_decay=reacdiff1d.WEIGHT_DECAY,
            scheduler_t_max=reacdiff1d.SCHEDULER_T_MAX,
            eval_interval=reacdiff1d.EVAL_INTERVAL,
            amp_enabled=reacdiff1d.USE_AMP,
            train_loader_drop_last=reacdiff1d.TRAIN_DROP_LAST,
            evaluation_loader_drop_last=reacdiff1d.EVAL_DROP_LAST,
            num_workers=reacdiff1d.NUM_WORKERS,
            pin_memory=reacdiff1d.PIN_MEMORY,
            pushforward_detach=reacdiff1d.PUSHFORWARD_DETACH,
            seed_policy=reacdiff1d.SEED_POLICY,
            training_loss="LpLoss(d=1,p=2,reduction=mean) rollout_loss_lp",
            selection_policy=reacdiff1d.SELECTION_POLICY,
        ),
        "data_configuration": _data_configuration(
            dataset_id=reacdiff1d.DATASET_ID,
            resolution=reacdiff1d.RESOLUTION,
            n_train=reacdiff1d.N_TRAIN,
            n_test=reacdiff1d.N_TEST,
            split_policy=reacdiff1d.SPLIT_POLICY,
            preprocessing_protocol="pinned_sirenfno_reacdiff_rollout_v1",
            normalization_policy=reacdiff1d.NORMALIZATION_POLICY,
            input_horizon=reacdiff1d.INPUT_STEPS,
            prediction_horizon=reacdiff1d.ROLLOUT,
        ),
        "evaluation_metadata": {
            **_CORRECTED_EVALUATION_METADATA,
            "evaluation_horizon": reacdiff1d.ROLLOUT,
            "evaluation_sample_count": {
                "train": reacdiff1d.N_TRAIN,
                "test": reacdiff1d.N_TEST,
            },
        },
        "metric_columns": {
            "final_train_corrected_step_relative_l2": (
                "train_corrected_step_relative_l2"
            ),
            "final_train_corrected_trajectory_relative_l2": (
                "train_corrected_trajectory_relative_l2"
            ),
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "paper_metric_columns": {
            "final_test_corrected_step_relative_l2": (
                "test_corrected_step_relative_l2"
            ),
            "final_test_corrected_trajectory_relative_l2": (
                "test_corrected_trajectory_relative_l2"
            ),
        },
        "record_metric_names": (
            "corrected_step_relative_l2",
            "corrected_trajectory_relative_l2",
        ),
        "reevaluation_space": "physical_source_values_no_normalization",
        "require_current_source_commit": True,
        "default_results": Path("results")
        / "reacdiff1024_sirenfno_81918ec",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-legacy-torch-version",
        action="store_true",
        help="read a trusted legacy TorchVersion checkpoint in weights-only mode",
    )
    parser.add_argument(
        "--allow-legacy-neuraloperator-metadata",
        action="store_true",
        help=(
            "read a trusted legacy checkpoint using only the verified pinned "
            "SpectralConv and PyTorch GELU weights-only allowlist"
        ),
    )
    parser.add_argument(
        "--reuse-training-source-commit",
        help=(
            "exact 40-hex historical training commit allowed for one existing "
            "checkpoint; requires --reuse-checkpoint-sha256"
        ),
    )
    parser.add_argument(
        "--reuse-checkpoint-sha256",
        help=(
            "exact 64-hex checkpoint digest allowed with the historical "
            "training commit"
        ),
    )
    return parser.parse_args()


def _dataset_identity_matches(
    identity: object,
    *,
    dataset_id: str,
    canonical: Mapping[str, Any],
) -> bool:
    if not isinstance(identity, dict):
        return False
    if identity.get("dataset_id") != dataset_id:
        return False
    for key in ("source", "resolution", "n_train", "n_test"):
        if not _metadata_matches(identity.get(key), canonical.get(key)):
            return False

    recorded_files = identity.get("files")
    canonical_files = canonical.get("files")
    if not isinstance(recorded_files, dict) or not isinstance(canonical_files, dict):
        return False
    if set(recorded_files) != set(canonical_files):
        return False
    for filename, expected in canonical_files.items():
        recorded = recorded_files.get(filename)
        if not isinstance(recorded, dict) or not isinstance(expected, dict):
            return False
        if recorded.get("sha256") != expected.get("sha256"):
            return False
        if recorded.get("size_bytes") != expected.get("size_bytes"):
            return False
    return True


def _normalise_metadata(value: Any) -> Any:
    """Return a stable JSON-compatible representation for exact comparisons."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalise_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_metadata(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _metadata_matches(actual: Any, expected: Any) -> bool:
    try:
        actual_json = json.dumps(
            _normalise_metadata(actual),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_json = json.dumps(
            _normalise_metadata(expected),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return False
    return actual_json == expected_json


def canonical_metadata_sha256(value: Any) -> str:
    """Return the SHA256 of the same canonical metadata form used by audits."""

    rendered = json.dumps(
        _normalise_metadata(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def training_protocol_identifier(spec: Mapping[str, Any]) -> str:
    """Identify the complete training/data contract, not only its display name."""

    payload = {
        "training_protocol_version": spec["training_protocol_version"],
        "training_configuration": spec["training_configuration"],
        "data_configuration": spec["data_configuration"],
    }
    return "paper_contract_sha256:" + canonical_metadata_sha256(payload)


def run_contract_metadata(
    dataset: str,
    source_state: Mapping[str, Any],
    *,
    evaluation_source_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common summary/checkpoint metadata for one paper run.

    Training entrypoints should merge this result into both the summary and
    checkpoint.  The checkpoint is then saved, :func:`sha256_file` is called,
    and only the summary receives ``checkpoint_sha256``; placing that digest
    inside the checkpoint would create an impossible self-hash cycle.
    """

    try:
        spec = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown paper dataset: {dataset}") from exc
    evaluation_state = (
        source_state
        if evaluation_source_state is None
        else evaluation_source_state
    )
    training_commit = source_state.get("source_repository_commit")
    evaluation_commit = evaluation_state.get("source_repository_commit")
    if not isinstance(training_commit, str) or not _COMMIT_PATTERN.fullmatch(
        training_commit
    ):
        raise ValueError("Training source commit must be a lowercase 40-hex digest.")
    if not isinstance(evaluation_commit, str) or not _COMMIT_PATTERN.fullmatch(
        evaluation_commit
    ):
        raise ValueError("Evaluation source commit must be a lowercase 40-hex digest.")
    training_dirty = source_state.get("source_repository_dirty")
    evaluation_dirty = evaluation_state.get("source_repository_dirty")
    if not isinstance(training_dirty, bool) or not isinstance(evaluation_dirty, bool):
        raise ValueError("Source dirty flags must be explicit booleans.")
    expected_upstream = upstream_manifest()["SirenFNO"]["commit"]
    return {
        "completion_schema_version": COMPLETION_SCHEMA_VERSION,
        "experiment": spec.get("experiment"),
        "dataset_id": spec["dataset_id"],
        "training_protocol_version": spec["training_protocol_version"],
        "training_protocol_identifier": training_protocol_identifier(spec),
        "training_configuration": _normalise_metadata(
            spec["training_configuration"]
        ),
        "data_configuration": _normalise_metadata(spec["data_configuration"]),
        "checkpoint_selection_policy": spec["training_configuration"][
            "checkpoint_selection_policy"
        ],
        "training_source_commit": training_commit,
        "training_source_dirty": training_dirty,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_protocol_version": spec["evaluation_protocol_version"],
        "evaluation_source_commit": evaluation_commit,
        "evaluation_source_dirty": evaluation_dirty,
        **_normalise_metadata(spec["evaluation_metadata"]),
        # Keep the old unambiguous training-source aliases as an independent
        # cross-check against metadata that merely claims to be current.
        "source_repository_commit": training_commit,
        "source_repository_dirty": training_dirty,
        "sirenfno_upstream_commit": expected_upstream,
    }


def run_environment_provenance(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the subset that must also be duplicated inside checkpoint env."""

    return {
        key: contract[key]
        for key in (
            "training_source_commit",
            "training_source_dirty",
            "source_repository_commit",
            "source_repository_dirty",
            "sirenfno_upstream_commit",
        )
    }


def _artifact_identity_matches(
    artifact: object,
    *,
    spec: Mapping[str, Any],
    model: str,
    seed: int,
    canonical_dataset: Mapping[str, Any],
) -> bool:
    if not isinstance(artifact, dict):
        return False
    if artifact.get("dataset_id") != str(spec["dataset_id"]):
        return False
    recorded_seed = artifact.get("seed")
    if artifact.get("model") != model:
        return False
    if (
        isinstance(recorded_seed, bool)
        or not isinstance(recorded_seed, int)
        or recorded_seed != seed
    ):
        return False
    experiment_seed = artifact.get("experiment_seed")
    if (
        isinstance(experiment_seed, bool)
        or not isinstance(experiment_seed, int)
        or experiment_seed != seed
    ):
        return False
    expected_experiment = spec.get("experiment")
    if expected_experiment is not None and artifact.get("experiment") != expected_experiment:
        return False
    return _dataset_identity_matches(
        artifact.get("dataset_identity"),
        dataset_id=str(spec["dataset_id"]),
        canonical=canonical_dataset,
    )


def sha256_file(path: Path) -> str:
    """Hash an artifact without introducing a checkpoint self-hash cycle."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifact is not an object")
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _contains_nonfinite(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return not bool(torch.isfinite(value).all().item())
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _read_training_log(
    path: Path,
    *,
    epochs: int,
) -> tuple[tuple[str, ...], list[dict[str, str]], tuple[str, ...]]:
    reasons: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames
        rows = list(reader)
    if raw_fieldnames is None:
        return (), rows, ("csv.missing_header",)
    fieldnames = tuple(raw_fieldnames)
    if len(fieldnames) != len(set(fieldnames)):
        reasons.append("csv.duplicate_header")
    if any(None in row for row in rows):
        reasons.append("csv.extra_unheaded_value")
    if len(rows) != epochs:
        reasons.append("csv.incomplete_epoch_history")

    actual_epochs: list[int] = []
    for row in rows:
        raw_epoch = row.get("epoch")
        try:
            if raw_epoch is None or str(int(raw_epoch)) != raw_epoch.strip():
                raise ValueError
            actual_epochs.append(int(raw_epoch))
        except (TypeError, ValueError):
            reasons.append("csv.invalid_epoch")
            break
    if actual_epochs and actual_epochs != list(range(1, epochs + 1)):
        reasons.append("csv.non_contiguous_or_duplicate_epochs")
    return fieldnames, rows, tuple(dict.fromkeys(reasons))


def _validate_csv_numeric_columns(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for column in columns:
        if column not in fieldnames:
            reasons.append(f"csv.missing_column:{column}")
            continue
        for row in rows:
            raw_value = row.get(column)
            try:
                value = float(raw_value) if raw_value is not None else math.nan
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                reasons.append(f"csv.nonfinite_or_invalid:{column}")
                break
    return tuple(reasons)


def _legacy_metric_columns(spec: Mapping[str, Any]) -> dict[str, str]:
    """Return the pre-correction rollout column names for legacy logs."""

    legacy: dict[str, str] = {}
    for metric_name, csv_column in spec["metric_columns"].items():
        legacy_name = str(metric_name).replace(
            "_corrected_trajectory_relative_l2", "_full_relative_l2"
        ).replace("_corrected_step_relative_l2", "_step_relative_l2")
        legacy_column = str(csv_column).replace(
            "_corrected_trajectory_relative_l2", "_full_relative_l2"
        ).replace("_corrected_step_relative_l2", "_step_relative_l2")
        legacy[legacy_name] = legacy_column
    return legacy


def _training_log_metric_contract(
    *,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    spec: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> tuple[str, Mapping[str, str], tuple[str, ...]]:
    """Classify native evaluation metadata and select its logged metric names.

    Evaluation provenance is deliberately not part of checkpoint validity.  A
    known or otherwise consistently labelled old evaluation remains eligible
    for re-evaluation, while disagreement between the two training artifacts
    is not accepted.
    """

    expected = str(spec["evaluation_protocol_version"])
    summary_protocol = summary.get("evaluation_protocol_version")
    checkpoint_protocol = checkpoint.get("evaluation_protocol_version")
    if summary_protocol != checkpoint_protocol:
        return (
            "invalid",
            spec["metric_columns"],
            ("evaluation.protocol_disagrees_between_training_artifacts",),
        )
    if summary_protocol is not None and (
        not isinstance(summary_protocol, str) or not summary_protocol
    ):
        return (
            "invalid",
            spec["metric_columns"],
            ("evaluation.invalid_training_artifact_protocol_label",),
        )
    if summary_protocol == expected:
        return "current", spec["metric_columns"], ()

    candidates = (_legacy_metric_columns(spec), spec["metric_columns"])
    for candidate in candidates:
        if (
            set(candidate.values()).issubset(fieldnames)
            and all(key in summary for key in candidate)
            and all(key in checkpoint for key in candidate)
        ):
            return "legacy", candidate, ()
    return "legacy", candidates[0], ()


def _validate_training_log_metrics(
    *,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    spec: Mapping[str, Any],
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> tuple[str, Mapping[str, str], tuple[str, ...]]:
    classification, metric_columns, protocol_reasons = _training_log_metric_contract(
        summary=summary,
        checkpoint=checkpoint,
        spec=spec,
        fieldnames=fieldnames,
    )
    reasons = list(protocol_reasons)
    reasons.extend(
        _validate_csv_numeric_columns(
            fieldnames,
            rows,
            tuple(metric_columns.values()),
        )
    )
    final_row = rows[-1] if rows else {}
    for metric_name, csv_column in metric_columns.items():
        summary_value = summary.get(metric_name)
        checkpoint_value = checkpoint.get(metric_name)
        if not _is_finite_number(summary_value):
            reasons.append(f"summary.invalid_metric:{metric_name}")
            continue
        if not _is_finite_number(checkpoint_value):
            reasons.append(f"checkpoint.invalid_metric:{metric_name}")
            continue
        if not math.isclose(
            float(summary_value),
            float(checkpoint_value),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            reasons.append(f"metric.summary_checkpoint_mismatch:{metric_name}")
        try:
            csv_value = float(final_row[csv_column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(csv_value) and not math.isclose(
            float(summary_value), csv_value, rel_tol=1e-9, abs_tol=1e-12
        ):
            reasons.append(f"metric.summary_csv_mismatch:{metric_name}")
    return classification, metric_columns, tuple(dict.fromkeys(reasons))


def _training_log_identity(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    metric_columns: Mapping[str, str],
) -> dict[str, Any]:
    epochs = [int(row["epoch"]) for row in rows]
    return {
        "sha256": sha256_file(path),
        "columns": list(fieldnames),
        "row_count": len(rows),
        "first_epoch": epochs[0],
        "final_epoch": epochs[-1],
        "epoch_sequence_sha256": canonical_metadata_sha256(epochs),
        "metric_columns": dict(metric_columns),
    }


def _expected_model_configuration(
    spec: Mapping[str, Any], model: str
) -> Mapping[str, Any]:
    factory = spec.get("model_configuration_factory")
    if not callable(factory):
        raise TypeError("model_configuration_factory is not callable")
    configuration = factory(model)
    if not isinstance(configuration, Mapping):
        raise TypeError("model configuration is not a mapping")
    return configuration


def _build_validation_model(spec: Mapping[str, Any], model: str) -> Any:
    builder: Callable[[str, torch.device], Any] | None = spec.get("model_builder")
    if builder is None:
        module = importlib.import_module(str(spec["module"]))
        builder = getattr(module, "build_model")
    if not callable(builder):
        raise TypeError("model builder is not callable")
    # Completion checks must not advance the caller's CPU training RNG.
    with torch.random.fork_rng(devices=[]):
        return builder(model, torch.device("cpu"))


def _validation_parameter_count(
    spec: Mapping[str, Any], validation_model: torch.nn.Module
) -> int:
    count_function = spec.get("parameter_count_function")
    if count_function is None and spec.get("model_builder") is None:
        module = importlib.import_module(str(spec["module"]))
        count_function = getattr(module, "count_model_params", None)
    if callable(count_function):
        return int(count_function(validation_model))
    return sum(parameter.numel() for parameter in validation_model.parameters())


def _append_if_mismatch(
    reasons: list[str],
    *,
    actual: Any,
    expected: Any,
    code: str,
) -> None:
    if not _metadata_matches(actual, expected):
        reasons.append(code)


def _validate_top_level_training_contract(
    *,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training_configuration: Mapping[str, Any],
) -> tuple[str, ...]:
    """Cross-check legacy display fields against the canonical nested contract."""

    reasons: list[str] = []
    required_summary_fields = (
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "scheduler_t_max",
        "eval_interval",
        "training_loss",
        "checkpoint_selection_policy",
    )
    for key in required_summary_fields:
        _append_if_mismatch(
            reasons,
            actual=summary.get(key),
            expected=training_configuration.get(key),
            code=f"summary.top_level_training_field_mismatch:{key}",
        )
    for label, artifact in (("summary", summary), ("checkpoint", checkpoint)):
        for key, expected in training_configuration.items():
            if key in artifact:
                _append_if_mismatch(
                    reasons,
                    actual=artifact[key],
                    expected=expected,
                    code=f"{label}.top_level_training_field_mismatch:{key}",
                )
    return tuple(dict.fromkeys(reasons))


def _validate_native_evaluation(
    *,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    spec: Mapping[str, Any],
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    current_evaluation_source_commit: str | None,
) -> tuple[CompletionStatus, tuple[str, ...]]:
    expected_protocol = str(spec["evaluation_protocol_version"])
    summary_protocol = summary.get("evaluation_protocol_version")
    checkpoint_protocol = checkpoint.get("evaluation_protocol_version")
    legacy_versions = set(spec.get("legacy_evaluation_protocol_versions", ()))
    if summary_protocol != expected_protocol or checkpoint_protocol != expected_protocol:
        if summary_protocol == checkpoint_protocol and (
            summary_protocol is None or summary_protocol in legacy_versions
        ):
            return (
                CompletionStatus.LEGACY_EVALUATION,
                ("evaluation.legacy_protocol",),
            )
        return (
            CompletionStatus.INVALID_EVALUATION,
            ("evaluation.protocol_version_mismatch",),
        )

    reasons: list[str] = []
    for label, artifact in (("summary", summary), ("checkpoint", checkpoint)):
        if artifact.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
            reasons.append(f"{label}.evaluation_schema_version_mismatch")
        evaluation_source = artifact.get("evaluation_source_commit")
        if not isinstance(evaluation_source, str) or not _COMMIT_PATTERN.fullmatch(
            evaluation_source
        ):
            reasons.append(f"{label}.invalid_evaluation_source_commit")
        if (
            current_evaluation_source_commit is not None
            and evaluation_source != current_evaluation_source_commit
        ):
            reasons.append(f"{label}.evaluation_source_commit_mismatch")
        if artifact.get("evaluation_source_dirty") is not False:
            reasons.append(f"{label}.dirty_evaluation_source")
        for key, expected in spec["evaluation_metadata"].items():
            _append_if_mismatch(
                reasons,
                actual=artifact.get(key),
                expected=expected,
                code=f"{label}.{key}_mismatch",
            )
    if summary.get("evaluation_source_commit") != checkpoint.get(
        "evaluation_source_commit"
    ):
        reasons.append("evaluation.source_disagrees_between_artifacts")

    metric_columns: Mapping[str, str] = spec["metric_columns"]
    reasons.extend(
        _validate_csv_numeric_columns(fieldnames, rows, tuple(metric_columns.values()))
    )
    final_row = rows[-1] if rows else {}
    for metric_name, csv_column in metric_columns.items():
        summary_value = summary.get(metric_name)
        checkpoint_value = checkpoint.get(metric_name)
        if not _is_finite_number(summary_value):
            reasons.append(f"summary.invalid_metric:{metric_name}")
            continue
        if not _is_finite_number(checkpoint_value):
            reasons.append(f"checkpoint.invalid_metric:{metric_name}")
            continue
        if not math.isclose(
            float(summary_value), float(checkpoint_value), rel_tol=1e-12, abs_tol=1e-15
        ):
            reasons.append(f"metric.summary_checkpoint_mismatch:{metric_name}")
        try:
            csv_value = float(final_row[csv_column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(csv_value) and not math.isclose(
            float(summary_value), csv_value, rel_tol=1e-9, abs_tol=1e-12
        ):
            reasons.append(f"metric.summary_csv_mismatch:{metric_name}")
    if reasons:
        return CompletionStatus.INVALID_EVALUATION, tuple(dict.fromkeys(reasons))
    return CompletionStatus.COMPLETE, ()


def _validate_reevaluation_record(
    *,
    record: Mapping[str, Any],
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_load_metadata: Mapping[str, Any],
    training_source_reuse: Mapping[str, Any],
    training_log_identity: Mapping[str, Any],
    training_evaluation_classification: str,
    spec: Mapping[str, Any],
    model: str,
    seed: int,
    current_evaluation_source_commit: str | None,
) -> tuple[CompletionStatus, tuple[str, ...]]:
    reasons: list[str] = []
    expected_values = {
        "record_type": "checkpoint_evaluation",
        "schema_version": 2,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_protocol_version": spec["evaluation_protocol_version"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load_metadata": checkpoint_load_metadata,
        "dataset_id": spec["dataset_id"],
        "experiment": spec.get("experiment"),
        "model": model,
        "seed": seed,
        "training_protocol_version": spec["training_protocol_version"],
        "training_protocol_identifier": training_protocol_identifier(spec),
        "training_source_commit": checkpoint.get("training_source_commit"),
        "training_source_dirty": checkpoint.get("training_source_dirty"),
        "training_artifact_validation_status": (
            "valid_training_legacy_evaluation"
            if training_evaluation_classification == "legacy"
            else "valid_training_current_evaluation"
        ),
        "training_artifact_evaluation_classification": (
            training_evaluation_classification
        ),
        "original_training_evaluation_protocol_version": summary.get(
            "evaluation_protocol_version"
        ),
        "training_log_identity": training_log_identity,
        "diagnostic_only": False,
        "paper_eligible": True,
        "diagnostic_reasons": [],
        "training_protocol": spec["training_configuration"],
        "training_protocol_sha256": canonical_metadata_sha256(
            spec["training_configuration"]
        ),
        "training_configuration": spec["training_configuration"],
        "data_configuration": spec["data_configuration"],
        "model_configuration": checkpoint.get("model_configuration"),
        "model_configuration_sha256": canonical_metadata_sha256(
            checkpoint.get("model_configuration")
        ),
        "dataset_identity": checkpoint.get("dataset_identity"),
        "dataset_identity_sha256": canonical_metadata_sha256(
            checkpoint.get("dataset_identity")
        ),
        "fixed_final_epoch": spec["epochs"],
        "checkpoint_selection_policy": spec["training_configuration"][
            "checkpoint_selection_policy"
        ],
        "factorization": checkpoint.get("factorization"),
        "rank": checkpoint.get("rank"),
        "parameter_count": checkpoint.get("parameter_count"),
    }
    if (
        "training_source_reuse" in record
        or checkpoint.get("training_source_commit")
        != current_evaluation_source_commit
    ):
        expected_values["training_source_reuse"] = training_source_reuse
    for key, expected in expected_values.items():
        _append_if_mismatch(
            reasons,
            actual=record.get(key),
            expected=expected,
            code=f"reevaluation.{key}_mismatch",
        )
    evaluation_source = record.get("evaluation_source_commit")
    if not isinstance(evaluation_source, str) or not _COMMIT_PATTERN.fullmatch(
        evaluation_source
    ):
        reasons.append("reevaluation.invalid_evaluation_source_commit")
    if (
        current_evaluation_source_commit is not None
        and evaluation_source != current_evaluation_source_commit
    ):
        reasons.append("reevaluation.evaluation_source_commit_mismatch")
    if type(record.get("evaluation_source_dirty")) is not bool:
        reasons.append("reevaluation.invalid_evaluation_source_dirty")
    elif record.get("evaluation_source_dirty") is not False:
        reasons.append("reevaluation.dirty_evaluation_source")
    if record.get("evaluation_space") != spec.get("reevaluation_space"):
        reasons.append("reevaluation.evaluation_space_mismatch")

    expected_evaluation_metadata: Mapping[str, Any] = spec["evaluation_metadata"]
    for key, expected in expected_evaluation_metadata.items():
        if key == "evaluation_sample_count":
            actual_counts = record.get(key)
            if not isinstance(actual_counts, Mapping) or actual_counts.get(
                "test"
            ) != expected["test"]:
                reasons.append("reevaluation.evaluation_sample_count_mismatch")
        else:
            _append_if_mismatch(
                reasons,
                actual=record.get(key),
                expected=expected,
                code=f"reevaluation.{key}_mismatch",
            )

    nested_metrics = record.get("metrics")
    if not isinstance(nested_metrics, Mapping):
        reasons.append("reevaluation.metrics_mapping_missing")
        nested_metrics = {}
    expected_record_metrics = set(spec["record_metric_names"])
    if set(nested_metrics) != expected_record_metrics:
        reasons.append("reevaluation.metric_set_mismatch")
    for metric_name in expected_record_metrics:
        if not _is_finite_number(nested_metrics.get(metric_name)):
            reasons.append(f"reevaluation.invalid_nested_metric:{metric_name}")
    for metric_name in spec["paper_metric_columns"]:
        if not _is_finite_number(record.get(metric_name)):
            reasons.append(f"reevaluation.invalid_metric:{metric_name}")
    if _contains_nonfinite(record):
        reasons.append("reevaluation.nonfinite_value")
    if reasons:
        return CompletionStatus.INVALID_EVALUATION, tuple(dict.fromkeys(reasons))
    return CompletionStatus.COMPLETE_REEVALUATED, ()


def resolve_current_source_commit(spec: Mapping[str, Any]) -> str | None:
    """Resolve the clean superproject revision for every paper dataset."""

    if spec.get("require_current_source_commit") is not True:
        return None
    state = source_repository_state(allow_dirty=False)
    commit = state.get("source_repository_commit")
    if state.get("source_repository_dirty") is not False:
        raise RuntimeError("The guarded paper source must be clean.")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("The guarded paper source commit is unavailable.")
    return commit


# Backward-compatible name for callers from the earlier sweep implementation.
_resolve_required_source_commit = resolve_current_source_commit


def _audit_run_artifacts(
    run_dir: Path,
    *,
    spec: Mapping[str, Any],
    model: str,
    seed: int,
    current_source_commit: str | None = None,
    evaluation_record_path: Path | None = None,
    allowed_historical_training_source_commit: str | None = None,
    allowed_historical_checkpoint_sha256: str | None = None,
    allow_dirty_training_source: bool = False,
    allow_legacy_torch_version: bool = False,
    allow_legacy_neuraloperator_metadata: bool = False,
    training_only: bool = False,
) -> RunCompletionAudit:
    """Audit one run without deleting, overwriting, or silently promoting it.

    ``evaluation_record_path`` supports a versioned, read-only re-evaluation
    record. A historical training source is accepted only when both its exact
    commit and this checkpoint's exact digest are explicitly supplied.
    """

    training_reasons: list[str] = []
    try:
        epochs = int(spec["epochs"])
        dataset_id = str(spec["dataset_id"])
        if model not in spec["models"]:
            training_reasons.append("request.unknown_model")
        if run_dir.parent.name != model or run_dir.name != f"seed_{seed}":
            training_reasons.append("run_directory.identity_mismatch")

        manifests = dataset_manifest()
        canonical_dataset = manifests.get(dataset_id)
        if not isinstance(canonical_dataset, dict):
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("dataset.manifest_missing",),
                False,
                False,
            )
        expected_sirenfno_commit = upstream_manifest()["SirenFNO"]["commit"]

        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "final_checkpoint.pt"
        log_path = run_dir / "training_log.csv"
        for name, path in (
            ("summary", summary_path),
            ("checkpoint", checkpoint_path),
            ("training_log", log_path),
        ):
            if not path.is_file():
                training_reasons.append(f"artifact.missing:{name}")
        if training_reasons:
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                tuple(dict.fromkeys(training_reasons)),
                False,
                False,
            )

        try:
            summary = _load_json_mapping(summary_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("summary.invalid_json",),
                False,
                False,
            )
        if _contains_nonfinite(summary):
            training_reasons.append("summary.nonfinite_value")
        try:
            checkpoint_load_result = load_weights_only_checkpoint(
                checkpoint_path,
                map_location="cpu",
                allow_legacy_torch_version=allow_legacy_torch_version,
                allow_legacy_neuraloperator_metadata=(
                    allow_legacy_neuraloperator_metadata
                ),
            )
            checkpoint = checkpoint_load_result.checkpoint
        except Exception:
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("checkpoint.unreadable",),
                False,
                False,
            )
        if not isinstance(checkpoint, Mapping):
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("checkpoint.not_mapping",),
                False,
                False,
            )
        if _contains_nonfinite(checkpoint):
            training_reasons.append("checkpoint.nonfinite_tensor_or_value")

        try:
            fieldnames, rows, csv_reasons = _read_training_log(
                log_path, epochs=epochs
            )
        except (OSError, UnicodeError, csv.Error):
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("csv.unreadable",),
                False,
                False,
            )
        training_reasons.extend(csv_reasons)
        training_reasons.extend(
            _validate_csv_numeric_columns(
                fieldnames,
                rows,
                (
                    "learning_rate",
                    "train_time_seconds",
                    "epoch_time_seconds",
                ),
            )
        )
        evaluation_classification, training_metric_columns, metric_reasons = (
            _validate_training_log_metrics(
                summary=summary,
                checkpoint=checkpoint,
                spec=spec,
                fieldnames=fieldnames,
                rows=rows,
            )
        )
        training_reasons.extend(metric_reasons)

        if not _artifact_identity_matches(
            summary,
            spec=spec,
            model=model,
            seed=seed,
            canonical_dataset=canonical_dataset,
        ):
            training_reasons.append("summary.run_identity_mismatch")
        if not _artifact_identity_matches(
            checkpoint,
            spec=spec,
            model=model,
            seed=seed,
            canonical_dataset=canonical_dataset,
        ):
            training_reasons.append("checkpoint.run_identity_mismatch")

        required_core = {
            "completion_schema_version": COMPLETION_SCHEMA_VERSION,
            "training_protocol_version": spec["training_protocol_version"],
            "training_protocol_identifier": training_protocol_identifier(spec),
            "training_configuration": spec["training_configuration"],
            "data_configuration": spec["data_configuration"],
            "checkpoint_selection_policy": spec["training_configuration"][
                "checkpoint_selection_policy"
            ],
        }
        for label, artifact in (("summary", summary), ("checkpoint", checkpoint)):
            for key, expected in required_core.items():
                _append_if_mismatch(
                    training_reasons,
                    actual=artifact.get(key),
                    expected=expected,
                    code=f"{label}.{key}_mismatch",
                )
        training_reasons.extend(
            _validate_top_level_training_contract(
                summary=summary,
                checkpoint=checkpoint,
                training_configuration=spec["training_configuration"],
            )
        )
        if summary.get("final_epoch") != epochs:
            training_reasons.append("summary.final_epoch_mismatch")
        if checkpoint.get("epoch") != epochs:
            training_reasons.append("checkpoint.epoch_mismatch")
        if summary.get("final_checkpoint") != "final_checkpoint.pt":
            training_reasons.append("summary.final_checkpoint_reference_mismatch")
        if summary.get("training_log") != "training_log.csv":
            training_reasons.append("summary.training_log_reference_mismatch")

        for key in (
            "model_class",
            "factorization",
            "rank",
            "parameter_count",
            "training_source_commit",
            "training_source_dirty",
        ):
            if not _metadata_matches(summary.get(key), checkpoint.get(key)):
                training_reasons.append(f"artifact.core_disagreement:{key}")

        environment = checkpoint.get("environment")
        if not isinstance(environment, Mapping):
            training_reasons.append("checkpoint.environment_missing")
            environment = {}

        training_source = checkpoint.get("training_source_commit")
        training_dirty = checkpoint.get("training_source_dirty")
        for label, artifact in (
            ("summary", summary),
            ("checkpoint", checkpoint),
            ("environment", environment),
        ):
            for source_key in (
                "training_source_commit",
                "source_repository_commit",
            ):
                recorded_source = artifact.get(source_key)
                if not isinstance(recorded_source, str) or not _COMMIT_PATTERN.fullmatch(
                    recorded_source
                ):
                    training_reasons.append(f"{label}.invalid_{source_key}")
                if recorded_source != training_source:
                    training_reasons.append(f"{label}.{source_key}_disagreement")
            for dirty_key in (
                "training_source_dirty",
                "source_repository_dirty",
            ):
                recorded_dirty = artifact.get(dirty_key)
                if type(recorded_dirty) is not bool:
                    training_reasons.append(f"{label}.invalid_{dirty_key}")
                elif recorded_dirty != training_dirty:
                    training_reasons.append(f"{label}.{dirty_key}_disagreement")
                elif recorded_dirty and not allow_dirty_training_source:
                    training_reasons.append(f"{label}.dirty_training_source")
            if artifact.get("sirenfno_upstream_commit") != expected_sirenfno_commit:
                training_reasons.append(f"{label}.sirenfno_commit_mismatch")

        checkpoint_digest = sha256_file(checkpoint_path)
        recorded_digest = summary.get("checkpoint_sha256")
        if not isinstance(recorded_digest, str) or not _SHA256_PATTERN.fullmatch(
            recorded_digest
        ):
            training_reasons.append("summary.invalid_checkpoint_sha256")
        elif recorded_digest != checkpoint_digest:
            training_reasons.append("summary.checkpoint_sha256_mismatch")
        if "checkpoint_sha256" in checkpoint:
            training_reasons.append("checkpoint.circular_self_hash_field")

        allowance_supplied = (
            allowed_historical_training_source_commit is not None
            or allowed_historical_checkpoint_sha256 is not None
        )
        allowance_valid = (
            isinstance(allowed_historical_training_source_commit, str)
            and _COMMIT_PATTERN.fullmatch(
                allowed_historical_training_source_commit
            )
            is not None
            and isinstance(allowed_historical_checkpoint_sha256, str)
            and _SHA256_PATTERN.fullmatch(allowed_historical_checkpoint_sha256)
            is not None
        )
        if allowance_supplied and not allowance_valid:
            training_reasons.append("request.invalid_exact_historical_allowance")
        exact_historical_identity_matches = bool(
            allowance_valid
            and training_source == allowed_historical_training_source_commit
            and checkpoint_digest == allowed_historical_checkpoint_sha256
        )
        exact_historical_source_allowed = False
        if (
            exact_historical_identity_matches
            and training_source != current_source_commit
        ):
            try:
                _training_source_reuse_metadata(
                    training_source_commit=str(training_source),
                    checkpoint_sha256=checkpoint_digest,
                    current_source_commit=current_source_commit,
                    allowed_historical_training_source_commit=(
                        allowed_historical_training_source_commit
                    ),
                    allowed_historical_checkpoint_sha256=(
                        allowed_historical_checkpoint_sha256
                    ),
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                training_reasons.append(
                    "training.source_diff_not_compatibility_only"
                )
            else:
                exact_historical_source_allowed = True
        stale_training_source = False
        require_current_commit = spec.get("require_current_source_commit") is True
        if (
            require_current_commit
            and training_source != current_source_commit
            and not exact_historical_source_allowed
        ):
            stale_training_source = True

        # Resolve the canonical constructor before loading weights. CAFE models
        # intentionally serialize a fully resolved get_config() payload that is
        # richer than the constructor kwargs, so the built configuration is the
        # authoritative exact comparison for every variant.
        _expected_model_configuration(spec, model)

        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            training_reasons.append("checkpoint.model_state_missing")
        elif not model_state:
            training_reasons.append("checkpoint.model_state_empty")
        elif not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in model_state.items()
        ):
            training_reasons.append("checkpoint.model_state_invalid_entry")

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            training_reasons.append("checkpoint.optimizer_state_missing")
        else:
            if not isinstance(optimizer_state.get("param_groups"), list) or not (
                optimizer_state.get("param_groups")
            ):
                training_reasons.append("checkpoint.optimizer_param_groups_missing")
            if not isinstance(optimizer_state.get("state"), Mapping) or not (
                optimizer_state.get("state")
            ):
                training_reasons.append("checkpoint.optimizer_state_empty")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if not isinstance(scheduler_state, Mapping) or not scheduler_state:
            training_reasons.append("checkpoint.scheduler_state_missing")

        if isinstance(model_state, Mapping) and model_state:
            try:
                built = _build_validation_model(spec, model)
                validation_model = built.model
                built_configuration = built.configuration
                _append_if_mismatch(
                    training_reasons,
                    actual=summary.get("model_hyperparameters"),
                    expected=built_configuration,
                    code="summary.model_configuration_mismatch",
                )
                _append_if_mismatch(
                    training_reasons,
                    actual=checkpoint.get("model_configuration"),
                    expected=built_configuration,
                    code="checkpoint.model_configuration_mismatch",
                )
                validation_model.load_state_dict(model_state, strict=True)
                actual_model_class = (
                    f"{validation_model.__class__.__module__}."
                    f"{validation_model.__class__.__qualname__}"
                )
                if summary.get("model_class") != actual_model_class:
                    training_reasons.append("summary.model_class_mismatch")
                parameter_count = _validation_parameter_count(spec, validation_model)
                if summary.get("parameter_count") != parameter_count:
                    training_reasons.append("summary.parameter_count_mismatch")
                built_factorization = getattr(built, "factorization", None)
                built_rank = getattr(built, "rank", None)
                if not _metadata_matches(
                    summary.get("factorization"), built_factorization
                ):
                    training_reasons.append("summary.factorization_mismatch")
                if not _metadata_matches(summary.get("rank"), built_rank):
                    training_reasons.append("summary.rank_mismatch")
            except Exception:
                training_reasons.append("checkpoint.strict_model_load_failed")

        if training_reasons:
            return RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                tuple(dict.fromkeys(training_reasons)),
                False,
                False,
            )
        if stale_training_source:
            return RunCompletionAudit(
                CompletionStatus.STALE_TRAINING_SOURCE,
                ("training.source_not_current",),
                True,
                False,
            )

        if training_only:
            # ``evaluation_classification`` and ``training_metric_columns`` are
            # intentionally computed above: a legacy evaluation can be
            # replaced, but its actual historical CSV must still be coherent.
            del evaluation_classification, training_metric_columns
            return RunCompletionAudit(
                CompletionStatus.COMPLETE,
                (),
                True,
                False,
            )

        if evaluation_record_path is not None:
            try:
                evaluation_record = _load_json_mapping(evaluation_record_path)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                return RunCompletionAudit(
                    CompletionStatus.INVALID_EVALUATION,
                    ("reevaluation.invalid_json",),
                    True,
                    False,
                )
            status, reasons = _validate_reevaluation_record(
                record=evaluation_record,
                summary=summary,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_digest,
                checkpoint_load_metadata=checkpoint_load_result.load_metadata,
                training_source_reuse=_training_source_reuse_metadata(
                    training_source_commit=str(training_source),
                    checkpoint_sha256=checkpoint_digest,
                    current_source_commit=current_source_commit,
                    allowed_historical_training_source_commit=(
                        allowed_historical_training_source_commit
                    ),
                    allowed_historical_checkpoint_sha256=(
                        allowed_historical_checkpoint_sha256
                    ),
                ),
                training_log_identity=_training_log_identity(
                    log_path,
                    fieldnames=fieldnames,
                    rows=rows,
                    metric_columns=training_metric_columns,
                ),
                training_evaluation_classification=evaluation_classification,
                spec=spec,
                model=model,
                seed=seed,
                current_evaluation_source_commit=current_source_commit,
            )
        else:
            status, reasons = _validate_native_evaluation(
                summary=summary,
                checkpoint=checkpoint,
                spec=spec,
                fieldnames=fieldnames,
                rows=rows,
                current_evaluation_source_commit=current_source_commit,
            )
        return RunCompletionAudit(status, reasons, True, not reasons)
    except Exception as exc:
        return RunCompletionAudit(
            CompletionStatus.INCOMPLETE,
            (f"audit.unexpected_error:{type(exc).__name__}",),
            False,
            False,
        )


def audit_run_completion(
    run_dir: Path,
    *,
    spec: Mapping[str, Any],
    model: str,
    seed: int,
    current_source_commit: str | None = None,
    evaluation_record_path: Path | None = None,
    allowed_historical_training_source_commit: str | None = None,
    allowed_historical_checkpoint_sha256: str | None = None,
    allow_legacy_torch_version: bool = False,
    allow_legacy_neuraloperator_metadata: bool = False,
) -> RunCompletionAudit:
    """Audit training plus an explicit or canonical versioned re-evaluation.

    A stored re-evaluation is never trusted merely because it exists: the shared
    validator below still checks its checkpoint hash, source, data, and metrics.
    Invalid records fail closed; they do not fall back to the native scores.
    """

    if evaluation_record_path is None:
        candidate = (
            run_dir
            / "evaluations"
            / str(spec["evaluation_protocol_version"])
            / "result.json"
        )
        if candidate.is_file():
            evaluation_record_path = candidate

    return _audit_run_artifacts(
        run_dir,
        spec=spec,
        model=model,
        seed=seed,
        current_source_commit=current_source_commit,
        evaluation_record_path=evaluation_record_path,
        allowed_historical_training_source_commit=(
            allowed_historical_training_source_commit
        ),
        allowed_historical_checkpoint_sha256=(
            allowed_historical_checkpoint_sha256
        ),
        allow_dirty_training_source=False,
        allow_legacy_torch_version=allow_legacy_torch_version,
        allow_legacy_neuraloperator_metadata=(
            allow_legacy_neuraloperator_metadata
        ),
        training_only=False,
    )


def validate_training_artifacts(
    run_dir: Path,
    *,
    spec: Mapping[str, Any],
    current_source_commit: str | None = None,
    allowed_historical_training_source_commit: str | None = None,
    allowed_historical_checkpoint_sha256: str | None = None,
    diagnostic_allow_dirty_training_source: bool = False,
    allow_legacy_torch_version: bool = False,
    allow_legacy_neuraloperator_metadata: bool = False,
) -> ValidatedTrainingArtifacts:
    """Validate and load a run for checkpoint re-evaluation.

    This is the sole public reader contract for summary/checkpoint/CSV bundles.
    ``diagnostic_allow_dirty_training_source`` is intentionally separate from
    any permission to execute an evaluator from a dirty checkout.  A caller
    using it must label its output diagnostic-only and not paper-eligible.
    """

    run_dir = run_dir.expanduser().resolve()
    seed_match = re.fullmatch(r"seed_(-?\d+)", run_dir.name)
    if seed_match is None or not run_dir.parent.name:
        audit = RunCompletionAudit(
            CompletionStatus.INCOMPLETE,
            ("run_directory.identity_mismatch",),
            False,
            False,
        )
        raise TrainingArtifactValidationError(audit)
    model = run_dir.parent.name
    seed = int(seed_match.group(1))
    audit = _audit_run_artifacts(
        run_dir,
        spec=spec,
        model=model,
        seed=seed,
        current_source_commit=current_source_commit,
        allowed_historical_training_source_commit=(
            allowed_historical_training_source_commit
        ),
        allowed_historical_checkpoint_sha256=(
            allowed_historical_checkpoint_sha256
        ),
        allow_dirty_training_source=diagnostic_allow_dirty_training_source,
        allow_legacy_torch_version=allow_legacy_torch_version,
        allow_legacy_neuraloperator_metadata=(
            allow_legacy_neuraloperator_metadata
        ),
        training_only=True,
    )
    if not audit.training_valid or audit.status is CompletionStatus.STALE_TRAINING_SOURCE:
        raise TrainingArtifactValidationError(audit)

    summary = _load_json_mapping(run_dir / "summary.json")
    checkpoint_load_result = load_weights_only_checkpoint(
        run_dir / "final_checkpoint.pt",
        map_location="cpu",
        allow_legacy_torch_version=allow_legacy_torch_version,
        allow_legacy_neuraloperator_metadata=(
            allow_legacy_neuraloperator_metadata
        ),
    )
    checkpoint = checkpoint_load_result.checkpoint
    if not isinstance(checkpoint, Mapping):  # Defensive; already audited above.
        raise TrainingArtifactValidationError(
            RunCompletionAudit(
                CompletionStatus.INCOMPLETE,
                ("checkpoint.not_mapping",),
                False,
                False,
            )
        )
    fieldnames, rows, _ = _read_training_log(
        run_dir / "training_log.csv", epochs=int(spec["epochs"])
    )
    classification, metric_columns, _ = _training_log_metric_contract(
        summary=summary,
        checkpoint=checkpoint,
        spec=spec,
        fieldnames=fieldnames,
    )
    training_dirty = checkpoint["training_source_dirty"]
    if training_dirty:
        validation_status = "valid_dirty_training_source_diagnostic"
    elif classification == "legacy":
        validation_status = "valid_training_legacy_evaluation"
    else:
        validation_status = "valid_training_current_evaluation"
    original_protocol = checkpoint.get("evaluation_protocol_version")
    training_source_reuse = _training_source_reuse_metadata(
        training_source_commit=str(checkpoint["training_source_commit"]),
        checkpoint_sha256=sha256_file(run_dir / "final_checkpoint.pt"),
        current_source_commit=current_source_commit,
        allowed_historical_training_source_commit=(
            allowed_historical_training_source_commit
        ),
        allowed_historical_checkpoint_sha256=(
            allowed_historical_checkpoint_sha256
        ),
    )
    return ValidatedTrainingArtifacts(
        summary=summary,
        checkpoint=checkpoint,
        model=model,
        seed=seed,
        experiment=str(checkpoint["experiment"]),
        training_protocol_version=str(checkpoint["training_protocol_version"]),
        training_protocol_identifier=str(
            checkpoint["training_protocol_identifier"]
        ),
        training_configuration=dict(checkpoint["training_configuration"]),
        data_configuration=dict(checkpoint["data_configuration"]),
        model_configuration=dict(checkpoint["model_configuration"]),
        dataset_identity=dict(checkpoint["dataset_identity"]),
        checkpoint_sha256=sha256_file(run_dir / "final_checkpoint.pt"),
        checkpoint_load_metadata=checkpoint_load_result.load_metadata,
        training_source_commit=str(checkpoint["training_source_commit"]),
        training_source_dirty=training_dirty,
        training_source_reuse=training_source_reuse,
        training_log_identity=_training_log_identity(
            run_dir / "training_log.csv",
            fieldnames=fieldnames,
            rows=rows,
            metric_columns=metric_columns,
        ),
        validation_status=validation_status,
        evaluation_classification=classification,
        original_evaluation_protocol_version=(
            original_protocol if isinstance(original_protocol, str) else None
        ),
    )


def is_complete(
    run_dir: Path,
    *,
    spec: Mapping[str, Any],
    model: str,
    seed: int,
    current_source_commit: str | None = None,
    allowed_historical_training_source_commit: str | None = None,
    allowed_historical_checkpoint_sha256: str | None = None,
    allow_legacy_torch_version: bool = False,
    allow_legacy_neuraloperator_metadata: bool = False,
) -> bool:
    """Backward-compatible bool wrapper around :func:`audit_run_completion`."""

    return audit_run_completion(
        run_dir,
        spec=spec,
        model=model,
        seed=seed,
        current_source_commit=current_source_commit,
        allowed_historical_training_source_commit=(
            allowed_historical_training_source_commit
        ),
        allowed_historical_checkpoint_sha256=(
            allowed_historical_checkpoint_sha256
        ),
        allow_legacy_torch_version=allow_legacy_torch_version,
        allow_legacy_neuraloperator_metadata=(
            allow_legacy_neuraloperator_metadata
        ),
    ).complete


def main() -> None:
    args = parse_args()
    reuse_training_source_commit = getattr(
        args, "reuse_training_source_commit", None
    )
    reuse_checkpoint_sha256 = getattr(args, "reuse_checkpoint_sha256", None)
    allow_legacy_neuraloperator_metadata = getattr(
        args, "allow_legacy_neuraloperator_metadata", False
    )
    allow_legacy_torch_version = getattr(
        args, "allow_legacy_torch_version", False
    )
    if (reuse_training_source_commit is None) != (
        reuse_checkpoint_sha256 is None
    ):
        raise SystemExit(
            "Historical reuse requires both --reuse-training-source-commit "
            "and --reuse-checkpoint-sha256."
        )
    if reuse_training_source_commit is not None and (
        _COMMIT_PATTERN.fullmatch(reuse_training_source_commit) is None
        or _SHA256_PATTERN.fullmatch(reuse_checkpoint_sha256) is None
    ):
        raise SystemExit("Historical reuse commit or checkpoint digest is invalid.")
    spec = DATASETS[args.dataset]
    models = tuple(args.models or spec["models"])
    unknown = sorted(set(models) - set(spec["models"]))
    if unknown:
        raise SystemExit(f"Unknown models: {', '.join(unknown)}")
    current_source_commit = resolve_current_source_commit(spec)
    results_root = args.results_root or spec["default_results"]

    for model in models:
        for seed in args.seeds:
            run_dir = results_root / model / f"seed_{seed}"
            if is_complete(
                run_dir,
                spec=spec,
                model=model,
                seed=seed,
                current_source_commit=current_source_commit,
                allowed_historical_training_source_commit=(
                    reuse_training_source_commit
                ),
                allowed_historical_checkpoint_sha256=(
                    reuse_checkpoint_sha256
                ),
                allow_legacy_neuraloperator_metadata=(
                    allow_legacy_neuraloperator_metadata
                ),
                allow_legacy_torch_version=allow_legacy_torch_version,
            ):
                print(f"COMPLETE: {model}/seed_{seed}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                audit = audit_run_completion(
                    run_dir,
                    spec=spec,
                    model=model,
                    seed=seed,
                    current_source_commit=current_source_commit,
                    allowed_historical_training_source_commit=(
                        reuse_training_source_commit
                    ),
                    allowed_historical_checkpoint_sha256=(
                        reuse_checkpoint_sha256
                    ),
                    allow_legacy_neuraloperator_metadata=(
                        allow_legacy_neuraloperator_metadata
                    ),
                    allow_legacy_torch_version=allow_legacy_torch_version,
                )
                raise SystemExit(
                    f"Incomplete run exists for {model}/seed_{seed}; refusing to skip "
                    "or overwrite it. Choose a new result root or review it manually. "
                    f"Audit: {audit.status.value}: {audit.describe()}"
                )
            command = [
                sys.executable,
                "-B",
                "-m",
                str(spec["module"]),
                "--model",
                model,
                "--seed",
                str(seed),
                "--data-root",
                str(args.data_root),
                "--results-root",
                str(results_root),
                "--device",
                args.device,
            ]
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            if not is_complete(
                run_dir,
                spec=spec,
                model=model,
                seed=seed,
                current_source_commit=current_source_commit,
                allowed_historical_training_source_commit=(
                    reuse_training_source_commit
                ),
                allowed_historical_checkpoint_sha256=(
                    reuse_checkpoint_sha256
                ),
                allow_legacy_neuraloperator_metadata=(
                    allow_legacy_neuraloperator_metadata
                ),
                allow_legacy_torch_version=allow_legacy_torch_version,
            ):
                audit = audit_run_completion(
                    run_dir,
                    spec=spec,
                    model=model,
                    seed=seed,
                    current_source_commit=current_source_commit,
                    allowed_historical_training_source_commit=(
                        reuse_training_source_commit
                    ),
                    allowed_historical_checkpoint_sha256=(
                        reuse_checkpoint_sha256
                    ),
                    allow_legacy_neuraloperator_metadata=(
                        allow_legacy_neuraloperator_metadata
                    ),
                    allow_legacy_torch_version=allow_legacy_torch_version,
                )
                raise SystemExit(
                    f"Completion audit failed for {model}/seed_{seed}: "
                    f"{audit.status.value}: {audit.describe()}"
                )


if __name__ == "__main__":
    main()
