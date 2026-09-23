"""Code-owned training/evaluation contracts derived from the pinned parent.

No data are loaded here. A TestContract is an explicit dependency injection for
synthetic diagnostics; it can never turn a short test into a formal paper run.
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any

from .contracts import DATASETS, FORMAL_EPOCHS


@dataclass(frozen=True)
class TestContract:
    label: str
    base_configuration: dict[str, Any]
    epochs: int = 1
    sample_count: int = 2
    horizon: int = 1
    optimizer: str = "torch.optim.AdamW"
    training_loss: str = "TEST_SYNTHETIC_MSE"
    scheduler_t_max: int | None = None
    scheduler_step_policy: str = "exactly_once_after_each_epoch_train_loop"

    def __post_init__(self):
        if not self.label.startswith("TEST_") or any(
            type(x) is not int or x <= 0 for x in (self.epochs, self.sample_count, self.horizon)
        ):
            raise ValueError("test_contract_invalid")
        if self.scheduler_t_max is not None and (type(self.scheduler_t_max) is not int or self.scheduler_t_max <= 0):
            raise ValueError("test_contract_scheduler_invalid")
        if self.scheduler_step_policy not in (
            "exactly_once_after_each_epoch_train_loop",
            "after_every_optimizer_step_plus_one_additional_step_at_epoch_end",
        ):
            raise ValueError("test_contract_scheduler_policy_invalid")


def pinned_contract(dataset: str, *, test_contract: TestContract | None = None) -> dict[str, Any]:
    """Reuse the parent's approved mapping, loss/protocol and data manifest."""
    from scripts import run_paper_sweep as sweep
    from scripts import summarize_results as results
    key = DATASETS[dataset].contract_key
    spec = sweep.DATASETS[key]
    if spec["epochs"] != FORMAL_EPOCHS:
        raise ValueError("pinned_protocol_not_500_epochs")
    cfg = importlib.import_module(DATASETS[dataset].config_module)
    adapter = results.DATASETS[key]
    if adapter.evaluation_kind == "corrected_rollout":
        from experiments.common.evaluation import DEFAULT_RELATIVE_L2_EPSILON
        epsilon, layout = DEFAULT_RELATIVE_L2_EPSILON, "B,T,C,*spatial"
    elif adapter.evaluation_kind == "burgers":
        epsilon, layout = 1e-12, "B,T,C=1,*spatial"
    else:
        # Read the actual pinned LpLoss default without importing optional
        # training adapters (which would demand unrelated data dependencies).
        import ast
        from .repository import resolve_repository_root
        source = resolve_repository_root() / "third_party/SirenFNO/neuralop/losses/data_losses.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LpLoss")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        defaults = dict(zip([x.arg for x in init.args.args][-len(init.args.defaults):], init.args.defaults))
        epsilon, layout = ast.literal_eval(defaults["eps"]), "B,C=1,*spatial"
    training = copy.deepcopy(spec["training_configuration"])
    data = copy.deepcopy(spec["data_configuration"])
    data["field_selection"] = {name.lower(): getattr(cfg, name) for name in (
        "SELECTED_FIELD", "SELECTED_HDF5_KEY", "TARGET_FIELD_INDEX", "PHYSICAL_CHANNELS",
        "INPUT_DIM", "OUTPUT_DIM", "EXPECTED_TIME_STEPS", "REDUCE_X", "REDUCE_T",
        "MAX_NORMALIZATION_TRAJECTORIES", "PUSHFORWARD_DETACH",
    ) if hasattr(cfg, name)}
    canonical_identity = results._canonical_dataset_identity(sweep, spec)
    horizon = spec["evaluation_metadata"]["evaluation_horizon"]
    if test_contract is not None:
        if not isinstance(test_contract, TestContract):
            raise ValueError("test_contract_not_explicit")
        training.update(epochs=test_contract.epochs, batch_size=test_contract.sample_count,
                        test_batch_size=test_contract.sample_count,
                        scheduler_t_max=test_contract.scheduler_t_max or test_contract.epochs,
                        optimizer=test_contract.optimizer, training_loss=test_contract.training_loss,
                        scheduler_step_policy=test_contract.scheduler_step_policy)
        data.update(n_train=test_contract.sample_count, n_test=test_contract.sample_count,
                    prediction_horizon=test_contract.horizon, split_policy="TEST_SYNTHETIC",
                    preprocessing_protocol="TEST_SYNTHETIC", normalization_policy="TEST_EXPLICIT")
        horizon = test_contract.horizon
        canonical_identity = {"dataset_id": spec["dataset_id"], "test_contract": test_contract.label,
                              "synthetic": True, "n_train": test_contract.sample_count,
                              "n_test": test_contract.sample_count}
    evaluation = results._build_evaluation_definition(
        adapter, sample_count=data["n_test"], horizon=horizon, epsilon=epsilon,
        layout=layout, metric_names=spec["record_metric_names"])
    if test_contract is not None:
        evaluation["test_contract"] = test_contract.label
    paper_metrics = {spec["paper_metric_columns"][name]: public
                     for name, public in results.PAPER_METRIC_FIELDS[key].items()}
    columns = ["epoch", *dict.fromkeys(spec["metric_columns"].values())]
    if adapter.evaluation_kind == "corrected_rollout":
        columns.extend(("train_evaluation_sample_count", "test_evaluation_sample_count", "evaluation_horizon"))
    columns.append("learning_rate")
    if dataset == "airfoil":
        columns.extend(("learning_rate_last_batch", "learning_rate_after_epoch_step"))
    columns.extend(("train_time_seconds", "epoch_time_seconds"))
    return {"dataset_id": spec["dataset_id"], "training_configuration": training,
            "data_configuration": data, "evaluation_definition": evaluation,
            "dataset_identity": canonical_identity, "paper_metrics": paper_metrics,
            "csv_columns": columns, "metric_columns": dict(spec["metric_columns"]),
            "training_protocol_version": spec["training_protocol_version"],
            "test_contract": test_contract.label if test_contract else None}
