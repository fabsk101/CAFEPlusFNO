"""Real tiny CPU checkpoint fixtures; never claims of formal dataset training."""
from __future__ import annotations

import csv
from pathlib import Path

from ablations.artifacts import build_artifact_contract, write_artifact
from ablations.contracts import EXPERIMENT_ID
from ablations.protocols import TestContract, pinned_contract
from ablations.tests.test_models import tiny_config


def tiny_contract(*, epochs=1, dataset="burgers1d", optimizer="torch.optim.AdamW",
                  training_loss="TEST_SYNTHETIC_MSE", factorization="dense"):
    from ablations.contracts import DATASETS
    return TestContract("TEST_REAL_TINY_CAFE", tiny_config(DATASETS[dataset].spatial_dim,
        factorization=factorization), epochs=epochs, optimizer=optimizer, training_loss=training_loss)


def make_artifact(root: Path, *, seed=0, condition="full", dataset="burgers1d",
                  model_variant="dense", test_contract=None):
    """One actual optimizer/scheduler update; other epochs only synthetic CSV.

    For epochs>1 the scheduler is stepped over real updates to remain a valid
    restoration contract, but CSV metrics are synthetic and explicitly labelled.
    """
    import torch
    from ablations.models import build_model_from_kwargs, preserved_rng_state
    from ablations.repository import resolve_repository_root, source_state
    from ablations.runtime_metadata import capture_runtime_metadata
    from ablations.contracts import DATASETS
    test_contract = test_contract or tiny_contract(dataset=dataset, factorization=model_variant)
    if test_contract.epochs > 3:
        raise ValueError("TEST_FIXTURE_TRAINING_BUDGET_EXCEEDED_USE_SYNTHETIC_CSV_ONLY")
    protocol = pinned_contract(dataset, test_contract=test_contract)
    run_dir = Path(root) / EXPERIMENT_ID / "diagnostic" / dataset / model_variant / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with preserved_rng_state():
        torch.manual_seed(seed)
        built = build_model_from_kwargs(spatial_dim=DATASETS[dataset].spatial_dim,
            configuration=test_contract.base_configuration, condition=condition)
        optimizer = torch.optim.AdamW(built.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=test_contract.epochs)
        shape = (2, 1, 7) if DATASETS[dataset].spatial_dim == 1 else (2, 1, 4, 5)
        x, target = torch.randn(shape), torch.randn(shape)
        for _ in range(test_contract.epochs):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(built.model(x), target).backward()
            optimizer.step()
            scheduler.step()
    rows = []
    for epoch in range(1, test_contract.epochs + 1):
        row = {name: 0.1 + seed / 10000 for name in protocol["csv_columns"]}
        row["epoch"] = epoch
        row["learning_rate"] = 0.001
        for name in ("train_evaluation_sample_count", "test_evaluation_sample_count"):
            if name in row:
                row[name] = test_contract.sample_count
        if "evaluation_horizon" in row:
            row["evaluation_horizon"] = test_contract.horizon
        rows.append(row)
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=protocol["csv_columns"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = build_artifact_contract(dataset, model_variant, condition,
        dataset_identity=protocol["dataset_identity"],
        runtime=capture_runtime_metadata(profile="diagnostic", device="cpu"),
        source=source_state(resolve_repository_root(), formal=False), test_contract=test_contract)
    metadata.update(seed=seed, result_kind="diagnostic", fixture_note="TEST_ONLY_SYNTHETIC_METRICS_REAL_TINY_UPDATE")
    path = write_artifact(run_dir, metadata=metadata, model=built.model,
        optimizer=optimizer, scheduler=scheduler, final_metrics=rows[-1], test_contract=test_contract)
    return path, test_contract
