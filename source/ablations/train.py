"""Train exactly one explicit CAFE+FNO ablation run."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

from .contracts import validate_condition_variant
from .contracts import (
    ABLATION_CONDITIONS,
    DATASETS,
    EXPERIMENT_ID,
    FORMAL_EPOCHS,
    MODEL_VARIANTS,
    seed_value,
    validate_seeds,
)
from .runtime import configure_runtime, apply_torch_runtime, contained, output_path, workspace_root
from .public import SafeArgumentParser, PublicError, emit_json, public_main
from .runtime_metadata import capture_runtime_metadata
from .repository import (
    activate_repository,
    import_audit,
    resolve_repository_root,
    source_state,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = SafeArgumentParser(
        description="Run one explicit CAFE+FNO ablation seed (fixed 500 epochs)."
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model-variant", required=True, choices=sorted(MODEL_VARIANTS)
    )
    parser.add_argument(
        "--condition", required=True, choices=sorted(ABLATION_CONDITIONS)
    )
    parser.add_argument("--seed", required=True, type=seed_value)
    parser.add_argument("--requested-seeds", nargs="+", type=seed_value,
                        help="Sweep plan only; excluded from model and reuse identity.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--diagnostic-source",
        action="store_true",
        help="Allow a Git-less/uncommitted development copy and label outputs diagnostic.",
    )
    args = parser.parse_args(argv)
    try:
        validate_condition_variant(args.condition, args.model_variant)
    except ValueError as exc:
        parser.error(str(exc))
    args.requested_seeds = args.requested_seeds if args.requested_seeds is not None else [args.seed]
    try:
        validate_seeds(args.requested_seeds)
        if args.seed not in args.requested_seeds:
            raise ValueError("RUN_SEED_NOT_IN_PLAN")
    except ValueError:
        parser.error("INVALID_SEED_PLAN")
    return args


def _prepare_run_paths(
    root: Path,
    *,
    dataset: str,
    model_variant: str,
    condition: str,
    seed: int,
    diagnostic: bool,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path]:
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS or condition not in ABLATION_CONDITIONS:
        raise ValueError("Unknown dataset/model/ablation tuple")
    seed_value(seed)
    run_dir = run_directory(root, dataset=dataset, model_variant=model_variant,
                            condition=condition, seed=seed, diagnostic=diagnostic)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_path(run_dir / "training_log.csv")
    summary_path = output_path(run_dir / "summary.json")
    checkpoint_path = output_path(run_dir / "final_checkpoint.pt")
    existing = [path for path in (csv_path, summary_path, checkpoint_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("ABLATION_OUTPUT_EXISTS")
    if overwrite:
        for path in existing:
            path.unlink()
    return run_dir, csv_path, summary_path, checkpoint_path


def run_directory(root: Path, *, dataset: str, model_variant: str, condition: str,
                  seed: int, diagnostic: bool = False) -> Path:
    if dataset not in DATASETS or model_variant not in MODEL_VARIANTS or condition not in ABLATION_CONDITIONS:
        raise PublicError("RUN_IDENTITY_INVALID")
    validate_condition_variant(condition, model_variant)
    seed_value(seed)
    return output_path(output_path(root) / EXPERIMENT_ID / ("diagnostic" if diagnostic else "formal")
                       / dataset / model_variant / condition / f"seed_{seed}")


def skip_completed_run(run_dir: Path, *, dataset: str, model_variant: str, condition: str,
                       seed: int, expected_context: dict, diagnostic: bool = False,
                       overwrite: bool = False, test_contract=None) -> bool:
    """Reuse complete compatible output; this does not resume partial epochs."""
    from .artifacts import validate_artifact
    run_dir = output_path(run_dir)
    candidates = [output_path(run_dir / name) for name in
                  ("summary.json", "final_checkpoint.pt", "training_log.csv")]
    existing = [path for path in candidates if path.exists()]
    if not existing or overwrite:
        return False
    if len(existing) != 3:
        raise PublicError("INCOMPLETE_RUN_REFUSES_OVERWRITE", dataset=dataset,
                          model_variant=model_variant, condition=condition, seed=seed)
    validate_artifact(run_dir / "summary.json", dataset=dataset, model_variant=model_variant,
                      condition=condition, seed=seed, expected_context=expected_context,
                      allow_diagnostic=diagnostic, test_contract=test_contract)
    return True


def sigma_optimizer_record(model: Any, optimizer: Any) -> dict:
    """Observe actual optimizer membership; no construction or RNG calls."""
    scalars = []
    for name, parameter in model.named_parameters():
        if not name.endswith("log_sigma"):
            continue
        groups = [(index, group) for index, group in enumerate(optimizer.param_groups)
                  if any(item is parameter for item in group["params"])]
        if len(groups) != 1:
            raise PublicError("SIGMA_OPTIMIZER_MEMBERSHIP_INVALID")
        index, group = groups[0]
        scalars.append({"parameter": name, "trainable_scalar_count": parameter.numel(),
                        "optimizer_included": True, "group_index": index,
                        "initial_learning_rate": float(group.get("initial_lr", group["lr"])),
                        "final_learning_rate": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"])})
    return {"encoders": scalars, "trainable_scalar_count": sum(item["trainable_scalar_count"] for item in scalars),
            "gaussian_G_trainable": any(name.endswith(".G") for name, _ in model.named_parameters())}


def _verify_dataset(train_module: Any, spec: Any, data_root: Path) -> dict:
    verifier = getattr(train_module, spec.verify_function)
    if spec.key == "darcy":
        return verifier(spec.contract_key, data_root)
    return verifier(data_root)


def _train_with_loop(
    train_module: Any,
    *,
    model: Any,
    data: Any,
    device: Any,
    csv_path: Path,
    test_contract: Any = None,
) -> tuple[list[dict], Any, Any]:
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_module.LEARNING_RATE),
        weight_decay=float(train_module.WEIGHT_DECAY),
    )
    if test_contract is not None:
        from .native_diagnostics import validate_diagnostic_budget
        validate_diagnostic_budget(test_contract)
        scheduler_t_max = test_contract.scheduler_t_max
    elif hasattr(train_module, "EXPECTED_SCHEDULER_T_MAX"):
        scheduler_t_max = int(train_module.EPOCHS) * len(data.train_loader)
        if scheduler_t_max != int(train_module.EXPECTED_SCHEDULER_T_MAX):
            raise RuntimeError("Airfoil scheduler/data-loader contract changed")
    else:
        scheduler_t_max = int(train_module.SCHEDULER_T_MAX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max
    )
    arguments = dict(model=model, data=data, optimizer=optimizer, scheduler=scheduler,
                     device=device, csv_path=csv_path)
    if test_contract is None:
        history = train_module.run_training_loop(**arguments)
    else:
        from .native_diagnostics import run_budgeted_native_loop
        history = run_budgeted_native_loop(train_module, test_contract, **arguments)
    return history, optimizer, scheduler


def _train_with_neuralop(
    train_module: Any,
    *,
    model: Any,
    data: tuple[Any, Any, Any],
    device: Any,
    csv_path: Path,
    test_contract: Any = None,
) -> tuple[list[dict], Any, Any]:
    import torch

    if test_contract is not None:
        from .native_diagnostics import validate_diagnostic_budget
        validate_diagnostic_budget(test_contract)

    train_loader, test_loaders, data_processor = data
    data_processor = data_processor.to(device)
    l2loss = train_module.LpLoss(d=2, p=2)
    h1loss = train_module.H1Loss(d=2)
    optimizer = train_module.AdamW(
        model.parameters(),
        lr=float(train_module.LEARNING_RATE),
        weight_decay=float(train_module.WEIGHT_DECAY),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(test_contract.scheduler_t_max if test_contract is not None
                          else int(train_module.SCHEDULER_T_MAX))
    )
    trainer = train_module.CsvLoggingTrainer(
        csv_path=csv_path,
        model=model,
        n_epochs=(test_contract.epochs if test_contract is not None else int(train_module.EPOCHS)),
        device=str(device),
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=int(train_module.EVAL_INTERVAL),
        use_distributed=False,
        verbose=test_contract is None,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=l2loss,
        eval_losses={"h1": h1loss, "l2": l2loss},
    )
    return trainer.history, optimizer, scheduler


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .repository import require_dataset_baseline
    require_dataset_baseline(args.dataset)
    repository_root = resolve_repository_root(args.repository_root)
    activate_repository(repository_root)
    provenance = source_state(repository_root, formal=not args.diagnostic_source)
    profile = "diagnostic" if args.diagnostic_source else "formal"
    configure_runtime(cpu=args.device == "cpu", profile=profile, threads=args.threads)

    import torch
    apply_torch_runtime()
    from .artifacts import build_artifact_contract, validate_artifact, write_artifact, canonical_json
    from .models import (
        build_ablation_model,
        count_parameters,
        describe_model,
        preserved_rng_state,
    )

    spec = DATASETS[args.dataset]
    train_module = importlib.import_module(spec.train_module)
    if int(train_module.EPOCHS) != FORMAL_EPOCHS:
        raise RuntimeError(
            f"The inherited protocol is no longer {FORMAL_EPOCHS} epochs."
        )
    device = train_module.resolve_device(args.device)
    data_root = contained(args.data_root, workspace_root())
    # The inherited loaders can create preprocessed files beside the input.
    output_path(data_root)
    import_audit(repository_root)
    dataset_identity = _verify_dataset(train_module, spec, data_root)
    runtime = capture_runtime_metadata(profile=profile, device=args.device, seed=args.seed)
    metadata = build_artifact_contract(args.dataset, args.model_variant, args.condition,
                                      dataset_identity=dataset_identity, runtime=runtime, source=provenance)
    metadata.update(seed=args.seed, requested_seeds=args.requested_seeds,
                    result_kind="diagnostic" if args.diagnostic_source else "formal")
    results_root = output_path(args.results_root or workspace_root() / "results" / "ablations")
    run_dir = run_directory(results_root, dataset=args.dataset, model_variant=args.model_variant,
                            condition=args.condition, seed=args.seed, diagnostic=args.diagnostic_source)
    if skip_completed_run(run_dir, dataset=args.dataset, model_variant=args.model_variant,
                          condition=args.condition, seed=args.seed,
                          diagnostic=args.diagnostic_source,
                          expected_context={"source": provenance, "runtime": runtime},
                          overwrite=args.overwrite):
        emit_json({"status": "SKIP_COMPLETE", "dataset": args.dataset, "model_variant": args.model_variant,
                   "condition": args.condition, "seed": args.seed, "requested_seeds": args.requested_seeds})
        return
    run_dir, csv_path, _, _ = _prepare_run_paths(
        results_root, dataset=args.dataset, model_variant=args.model_variant,
        condition=args.condition, seed=args.seed, diagnostic=args.diagnostic_source, overwrite=args.overwrite)

    # Preserve the exact main-run ordering: one seed before loader creation and
    # actual model construction. Reference counting restores RNG state exactly.
    train_module.set_seed(args.seed)
    data = getattr(train_module, spec.load_function)(data_root)
    if spec.key == "airfoil" and len(data.train_loader) != int(
        train_module.EXPECTED_TRAIN_BATCHES_PER_EPOCH
    ):
        raise RuntimeError("Airfoil training loader length changed")

    with preserved_rng_state():
        full_reference = build_ablation_model(
            args.dataset, args.model_variant, "full", device="cpu"
        )
        full_parameter_count = count_parameters(full_reference.model)
    del full_reference

    if device.type == "cuda":
        torch.cuda.empty_cache()
    built = build_ablation_model(
        args.dataset, args.model_variant, args.condition, device=device
    )
    model = built.model
    dimensions = describe_model(model)
    dimensions["full_parameter_count"] = full_parameter_count
    dimensions["parameter_delta_from_full"] = (
        dimensions["parameter_count"] - full_parameter_count
    )

    print(
        f"\n========== {EXPERIMENT_ID}: {args.dataset}/{args.model_variant}/"
        f"{args.condition}/seed_{args.seed} =========="
    )
    print(f"Device: {device}")
    print(f"Parameters: {dimensions['parameter_count']} ({dimensions['parameter_delta_from_full']:+d} vs Full)")
    emit_json({"status": "RUNNING", "dataset": args.dataset, "model_variant": args.model_variant,
               "condition": args.condition, "seed": args.seed})
    started = time.perf_counter()
    if spec.trainer_kind == "loop":
        history, optimizer, scheduler = _train_with_loop(
            train_module,
            model=model,
            data=data,
            device=device,
            csv_path=csv_path,
        )
    else:
        history, optimizer, scheduler = _train_with_neuralop(
            train_module,
            model=model,
            data=data,
            device=device,
            csv_path=csv_path,
        )
    elapsed = time.perf_counter() - started
    if len(history) != FORMAL_EPOCHS or int(history[-1]["epoch"]) != FORMAL_EPOCHS:
        raise RuntimeError("Training did not reach the fixed final epoch")

    import_audit(repository_root)
    if canonical_json(source_state(repository_root, formal=not args.diagnostic_source)) != canonical_json(provenance):
        raise PublicError("SOURCE_CHANGED_DURING_RUN", dataset=args.dataset, condition=args.condition, seed=args.seed)
    metadata.update(model_dimensions=dimensions, sigma_optimizer=sigma_optimizer_record(model, optimizer))
    summary_path = write_artifact(run_dir, metadata=metadata, model=model, optimizer=optimizer,
                                 scheduler=scheduler, final_metrics=history[-1],
                                 total_training_time_seconds=elapsed)
    validate_artifact(summary_path, dataset=args.dataset, model_variant=args.model_variant,
                      condition=args.condition, seed=args.seed, allow_diagnostic=args.diagnostic_source,
                      expected_context={"source": provenance, "runtime": runtime})
    emit_json({"status": "COMPLETE", "dataset": args.dataset, "condition": args.condition,
               "model_variant": args.model_variant, "seed": args.seed, "final_epoch": FORMAL_EPOCHS})


if __name__ == "__main__":
    public_main(main)
