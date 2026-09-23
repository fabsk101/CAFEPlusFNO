"""Variant-aware, weights-only reload using the parent's checkpoint helper."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import math
from .contracts import ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS
from .runtime import contained, workspace_root


def validate_finite_tree(value, *, state: bool = False) -> None:
    """Inspect actual tensor values, including complex persistent buffers."""
    import torch
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError("checkpoint_nonfinite_tensor")
    elif isinstance(value, Mapping):
        for item in value.values():
            validate_finite_tree(item, state=state)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_finite_tree(item, state=state)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("checkpoint_nonfinite_scalar")
    elif state and value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("checkpoint_invalid_state_object")


def validate_checkpoint_payload(payload, model, *, training_configuration=None,
                                data_configuration=None, epochs=None):
    """Strictly restore weights and optionally the actual optimizer/scheduler.

    A weights-only payload is already normalized by the parent's reader. Do not
    rebuild its OrderedDict: the PyTorch ._metadata attribute must survive.
    """
    import torch
    weights = payload.get("model_state_dict")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("checkpoint_empty_or_invalid_model_state")
    if any(type(name) is not str or not torch.is_tensor(tensor) for name, tensor in weights.items()):
        raise ValueError("checkpoint_invalid_weight_entry")
    validate_finite_tree(weights, state=True)
    expected = model.state_dict()
    if set(weights) != set(expected) or any(
        weights[name].shape != expected[name].shape or weights[name].dtype != expected[name].dtype
        for name in weights
    ):
        raise ValueError("checkpoint_weight_key_shape_dtype_mismatch")
    try:
        model.load_state_dict(weights, strict=True)
    except (RuntimeError, ValueError, TypeError):
        raise ValueError("checkpoint_strict_restore_failed") from None
    for module in model.modules():
        clear = getattr(module, "clear_feature_cache", None)
        if callable(clear):
            clear()
    if training_configuration is None:
        for name in ("optimizer_state_dict", "scheduler_state_dict"):
            if name in payload:
                validate_finite_tree(payload[name], state=True)
        return None, None
    cfg = training_configuration
    state, scheduler_state = payload.get("optimizer_state_dict"), payload.get("scheduler_state_dict")
    if not isinstance(state, Mapping) or not state or not isinstance(scheduler_state, Mapping) or not scheduler_state:
        raise ValueError("checkpoint_missing_optimizer_scheduler_state")
    validate_finite_tree(state, state=True)
    validate_finite_tree(scheduler_state, state=True)
    if cfg["optimizer"] == "torch.optim.AdamW":
        optimizer_class = torch.optim.AdamW
    elif cfg["optimizer"] == "NeuralOperator AdamW":
        from neuralop.training import AdamW
        optimizer_class = AdamW
    else:
        raise ValueError("checkpoint_unapproved_optimizer")
    optimizer = optimizer_class(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"])
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state.get("state"), Mapping) or not state["state"]:
        raise ValueError("checkpoint_invalid_optimizer_structure")
    params = list(model.parameters())
    ids = groups[0].get("params")
    if not isinstance(ids, list) or len(ids) != len(params) or len(set(ids)) != len(ids):
        raise ValueError("checkpoint_optimizer_parameter_mismatch")
    if set(state["state"]) != set(ids):
        raise ValueError("checkpoint_optimizer_incomplete_parameter_state")
    template = optimizer.state_dict()["param_groups"][0]
    for key, value in template.items():
        if key not in {"params", "lr"} and groups[0].get(key) != value:
            raise ValueError("checkpoint_optimizer_configuration_mismatch")
    if not isinstance(groups[0].get("lr"), (int, float)) or isinstance(groups[0]["lr"], bool) or groups[0]["lr"] < 0:
        raise ValueError("checkpoint_optimizer_learning_rate_invalid")
    expected_optimizer_steps = None
    if epochs is not None and data_configuration is not None:
        count = data_configuration["n_train"]
        batch_size = cfg["batch_size"]
        batches = count // batch_size if cfg.get("train_loader_drop_last") else math.ceil(count / batch_size)
        expected_optimizer_steps = epochs * batches
    for identifier, parameter in zip(ids, params):
        entry = state["state"][identifier]
        if not isinstance(entry, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(entry):
            raise ValueError("checkpoint_optimizer_parameter_state_invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            if not torch.is_tensor(entry[name]) or entry[name].shape != parameter.shape:
                raise ValueError("checkpoint_optimizer_moment_shape_mismatch")
        step = entry["step"]
        step = step.item() if torch.is_tensor(step) and step.numel() == 1 else step
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step <= 0 or int(step) != step:
            raise ValueError("checkpoint_optimizer_step_invalid")
        if expected_optimizer_steps is not None and step != expected_optimizer_steps:
            raise ValueError("checkpoint_optimizer_step_count_mismatch")
    if not set(scheduler.state_dict()).issubset(scheduler_state):
        raise ValueError("checkpoint_scheduler_structure_invalid")
    if scheduler_state.get("T_max") != cfg["scheduler_t_max"] or scheduler_state.get("eta_min") != 0:
        raise ValueError("checkpoint_scheduler_configuration_mismatch")
    if epochs is not None:
        expected_steps = epochs
        if cfg.get("scheduler_step_policy") == "after_every_optimizer_step_plus_one_additional_step_at_epoch_end":
            count = data_configuration["n_train"]
            batches = math.ceil(count / cfg["batch_size"])
            expected_steps = epochs * (batches + 1)
        if scheduler_state.get("last_epoch") != expected_steps:
            raise ValueError("checkpoint_scheduler_step_mismatch")
        if scheduler_state.get("_step_count") != expected_steps + 1:
            raise ValueError("checkpoint_scheduler_step_count_mismatch")
    if scheduler_state.get("base_lrs") != [cfg["learning_rate"]]:
        raise ValueError("checkpoint_scheduler_base_learning_rate_mismatch")
    if scheduler_state.get("_last_lr") != [groups[0]["lr"]]:
        raise ValueError("checkpoint_scheduler_optimizer_learning_rate_mismatch")
    actual_lr = groups[0]["lr"]
    expected_lr = cfg["learning_rate"] * (1 + math.cos(math.pi * scheduler_state["last_epoch"] / cfg["scheduler_t_max"])) / 2
    if not math.isclose(actual_lr, expected_lr, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError("checkpoint_scheduler_learning_rate_mismatch")
    try:
        optimizer.load_state_dict(state)
        scheduler.load_state_dict(scheduler_state)
    except (RuntimeError, ValueError, TypeError, KeyError):
        raise ValueError("checkpoint_optimizer_scheduler_restore_failed") from None
    return optimizer, scheduler


def restore_model_checkpoint(path: str | Path, *, expected_condition: str,
                             expected_dataset: str, expected_model_variant: str,
                             expected_seed: int | None = None):
    from experiments.common.checkpoints import load_weights_only_checkpoint
    from .models import build_model_from_kwargs

    payload = load_weights_only_checkpoint(contained(path, workspace_root()), map_location="cpu").checkpoint
    if expected_condition not in ABLATION_CONDITIONS or expected_dataset not in DATASETS or expected_model_variant not in MODEL_VARIANTS:
        raise ValueError("Unknown expected checkpoint identity")
    expected = {"ablation_variant": expected_condition, "condition": expected_condition,
                "dataset": expected_dataset, "model_variant": expected_model_variant}
    if expected_seed is not None:
        expected["seed"] = expected_seed
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint dataset/variant/seed mismatch")
    kwargs = payload["base_configuration"]
    from .models import preserved_rng_state
    with preserved_rng_state():
        built = build_model_from_kwargs(spatial_dim=DATASETS[expected_dataset].spatial_dim,
                                        configuration=kwargs, condition=expected_condition, device="cpu")
    if built.factorization != expected_model_variant:
        raise ValueError("Checkpoint factorization mismatch")
    from .artifacts import canonical_json
    if canonical_json(payload.get("model_configuration")) != canonical_json(built.configuration):
        raise ValueError("Checkpoint resolved model configuration mismatch")
    validate_checkpoint_payload(payload, built.model)
    return built, payload
