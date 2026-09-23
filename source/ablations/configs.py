"""Ablation-owned configuration; parent dataset protocols remain unchanged."""
from __future__ import annotations

import importlib
from .contracts import DATASETS, MODEL_VARIANTS, FORMAL_EPOCHS


def model_configuration(dataset: str, variant: str) -> dict:
    from .repository import require_dataset_baseline
    require_dataset_baseline(dataset)
    module = importlib.import_module(DATASETS[dataset].config_module)
    return dict(module.model_constructor_kwargs(MODEL_VARIANTS[variant]))


def protocol_configuration(dataset: str) -> dict:
    module = importlib.import_module(DATASETS[dataset].config_module)
    if module.EPOCHS != FORMAL_EPOCHS:
        raise ValueError("The inherited dataset protocol must remain at 500 epochs")
    names = ("EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "SCHEDULER_T_MAX",
             "EXPECTED_SCHEDULER_T_MAX", "EVAL_INTERVAL", "NUM_WORKERS", "USE_AMP",
             "ROLLOUT", "SELECTION_POLICY", "NORMALIZATION_POLICY", "SPLIT_POLICY")
    return {name.lower(): getattr(module, name) for name in names if hasattr(module, name)}
