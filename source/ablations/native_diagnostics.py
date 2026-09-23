"""Bounded CPU execution of the pinned dataset adapters on synthetic inputs.

Imports and evidence access are stdlib-only. The diagnostic loader ends the
five fixed-500 loops before the second training epoch's first batch; it never changes
parent globals, loss functions, factories, or the native scheduler cadence.
"""
from __future__ import annotations

import copy
import csv
import importlib
import importlib.util
import math
from pathlib import Path

REQUIRED_STAGES = ("native_training", "native_loss", "native_evaluation",
                   "checkpoint_write", "strict_reload", "aggregate")
ADDITIONAL_CASES = (("airfoil", "learnable_sigma"), ("burgers1d", "fourier_only"),
                    ("cfd2d", "chebyshev_only"), ("darcy", "without_linear_branches"))
_EVIDENCE: dict[str, dict] = {}


def reset_execution_evidence():
    _EVIDENCE.clear()


def execution_evidence():
    return copy.deepcopy(_EVIDENCE)


def _record(record):
    dataset = record["dataset"]
    if record["condition"] == "full":
        record["additional_conditions"] = _EVIDENCE.get(dataset, {}).get("additional_conditions", {})
        _EVIDENCE[dataset] = copy.deepcopy(record)
    else:
        item = _EVIDENCE.setdefault(dataset, {"dataset": dataset, "condition": "full",
            "status": "NOT_RUN", "executed": False,
            "reason_code": "NATIVE_FULL_NOT_EXECUTED", "stages": {key: False for key in REQUIRED_STAGES}})
        item.setdefault("additional_conditions", {})[record["condition"]] = copy.deepcopy(record)


class _DiagnosticBudgetComplete(RuntimeError):
    pass


class _ObservedLoader:
    """Transparent loader input with counters and an optional epoch budget."""
    def __init__(self, loader, *, epochs=None):
        self.loader, self.epochs = loader, epochs
        self.iterations = self.batches = self.samples = 0

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        if self.epochs is not None and self.iterations >= self.epochs:
            raise _DiagnosticBudgetComplete()
        self.iterations += 1
        for batch in self.loader:
            first = next(iter(batch.values())) if isinstance(batch, dict) else batch[0]
            self.batches += 1
            self.samples += len(first)
            yield batch


def validate_diagnostic_budget(contract):
    from .protocols import TestContract
    if (not isinstance(contract, TestContract) or contract.epochs not in (1, 2)
            or contract.sample_count > 4 or contract.scheduler_t_max is None):
        raise ValueError("NATIVE_DIAGNOSTIC_BUDGET_INVALID")


def run_budgeted_native_loop(module, contract, **arguments):
    """Allow the native epoch, evaluation and CSV writer to finish, then stop."""
    validate_diagnostic_budget(contract)
    loader = arguments["data"].train_loader
    if not isinstance(loader, _ObservedLoader) or loader.epochs != contract.epochs:
        raise ValueError("NATIVE_DIAGNOSTIC_LOADER_BUDGET_REQUIRED")
    try:
        module.run_training_loop(**arguments)
    except _DiagnosticBudgetComplete:
        pass
    else:
        raise ValueError("NATIVE_DIAGNOSTIC_BUDGET_NOT_REACHED")
    if loader.iterations != contract.epochs or loader.batches != contract.epochs * len(loader):
        raise ValueError("NATIVE_DIAGNOSTIC_BATCH_COUNT_INVALID")
    with arguments["csv_path"].open(encoding="utf-8", newline="") as handle:
        rows = [{key: (int(value) if key == "epoch" else float(value))
                 for key, value in row.items()} for row in csv.DictReader(handle)]
    if [row["epoch"] for row in rows] != list(range(1, contract.epochs + 1)):
        raise ValueError("NATIVE_DIAGNOSTIC_COMPLETED_EPOCH_MISSING")
    return rows


def _synthetic_data(dataset, module, contract):
    """Execute pinned normalization/rollout reshaping, with no data-file reads."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    n = contract.sample_count
    observers = {}

    def loader(values, name, *, budget=False):
        observed = _ObservedLoader(DataLoader(values, batch_size=n, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False),
            epochs=contract.epochs if budget else None)
        observers[name] = observed
        return observed

    if dataset in ("darcy", "ns2d"):
        from neuralop.data.transforms.data_processors import DefaultDataProcessor
        from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
        x, y = torch.rand(n, 1, 7, 9) + 0.5, torch.rand(n, 1, 7, 9) + 1.0
        # PTDataset fits encoders on raw training tensors; preprocessing applies
        # them at batch time. Darcy's pinned loader keeps inputs unencoded,
        # while NS explicitly encodes both inputs and targets channel-wise.
        input_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3]) if dataset == "ns2d" else None
        if input_normalizer is not None:
            input_normalizer.fit(x)
        output_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
        output_normalizer.fit(y)
        processor_evidence = {"training_normalized_batches": 0, "evaluation_decoded_batches": 0,
                              "input_policy_verified_batches": 0}

        class ObservedProcessor(DefaultDataProcessor):
            # Observe the pinned implementation's actual result; do not replace
            # its normalization, loss inputs or decoded evaluation outputs.
            def preprocess(self, data_dict, batched=True):
                raw_input = data_dict["x"].to(self.device).clone()
                target = data_dict["y"].to(self.device).clone()
                result = super().preprocess(data_dict, batched=batched)
                expected_input = (self.in_normalizer.transform(raw_input)
                                  if self.in_normalizer is not None else raw_input)
                if not torch.equal(result["x"], expected_input):
                    raise ValueError("NATIVE_DIAGNOSTIC_PROCESSOR_INPUT_MISMATCH")
                processor_evidence["input_policy_verified_batches"] += 1
                expected = self.out_normalizer.transform(target) if self.training else target
                if not torch.equal(result["y"], expected):
                    raise ValueError("NATIVE_DIAGNOSTIC_PROCESSOR_TARGET_MISMATCH")
                if self.training:
                    processor_evidence["training_normalized_batches"] += 1
                return result

            def postprocess(self, output, data_dict):
                expected = self.out_normalizer.inverse_transform(output) if not self.training else output
                result, sample = super().postprocess(output, data_dict)
                if not torch.equal(result, expected):
                    raise ValueError("NATIVE_DIAGNOSTIC_PROCESSOR_DECODE_MISMATCH")
                if not self.training:
                    processor_evidence["evaluation_decoded_batches"] += 1
                return result, sample

        processor = ObservedProcessor(in_normalizer=input_normalizer, out_normalizer=output_normalizer)
        samples = [{"x": x[index], "y": y[index]} for index in range(n)]
        data = (loader(samples, "train", budget=True),
                {"synthetic": loader(samples, "test_evaluation")}, processor)
        return data, observers, {"input_shape": list(x.shape), "target_shape": list(y.shape),
            "processor": "DefaultDataProcessor", "normalization": "UnitGaussianNormalizer channel-wise",
            "encode_input": input_normalizer is not None, "encode_output": True,
            "processor_execution": processor_evidence, "horizon": 1}
    if dataset == "airfoil":
        x, y = torch.rand(n, 7, 9, module.INPUT_DIM) + 0.5, torch.rand(n, 7, 9, module.OUTPUT_DIM) + 1.0
        samples = TensorDataset(x, y)
        data = module.AirfoilData(loader(samples, "train", budget=True),
            loader(samples, "test_evaluation"), n, tuple(x.shape[1:]), tuple(y.shape[1:]))
        return data, observers, {"input_shape": list(x.shape), "target_shape": list(y.shape),
            "processor": "native _channel_first", "normalization": "none", "horizon": 1}
    steps, horizon = int(module.INPUT_STEPS), int(module.ROLLOUT)
    if dataset == "burgers1d":
        raw = torch.rand(n, steps + horizon, 17) + 0.5
        mean, std = module.compute_mean_std_from_ram(raw, max_traj_for_stats=n)
        samples = module.RolloutRAMDataset1D(raw, steps, horizon, mean, std)
        data = module.BurgersData(loader(samples, "train", budget=True),
            loader(samples, "train_evaluation"), loader(samples, "test_evaluation"),
            float(mean), float(std), n, steps + horizon, 17, "TEST_SYNTHETIC")
        normalizing = "native compute_mean_std_from_ram and RolloutRAMDataset1D"
    else:
        spatial = (7, 9) if dataset == "cfd2d" else (17,)
        raw = torch.rand(n, steps + horizon, *spatial, int(module.FIELD_DIM)) + 0.5
        samples = module.RolloutRAMDataset(raw, steps, horizon, None, None)
        loaders = [loader(samples, "train", budget=True), loader(samples, "train_evaluation"),
                   loader(samples, "test_evaluation")]
        common = [*loaders, n, steps + horizon]
        if dataset == "cfd2d":
            data = module.CFD2DData(*common, spatial, module.SELECTED_FIELD)
        elif dataset == "cfd1d":
            data = module.CFD1DData(*common, spatial[0], module.SELECTED_FIELD)
        else:
            data = module.ReacDiffData(*common, spatial[0], "TEST_SYNTHETIC")
        normalizing = "none; native RolloutRAMDataset reshape"
    x, y = samples[0]
    return data, observers, {"input_shape": [n, *x.shape], "target_shape": [n, *y.shape],
        "processor": type(samples).__name__, "normalization": normalizing,
        "history": steps, "horizon": horizon, "pushforward_detach": bool(module.PUSHFORWARD_DETACH)}


def run_native_diagnostic(dataset, *, condition="full", output_root, seed=42):
    """Run actual native training, evaluation and artifact paths for one case."""
    from .contracts import DATASETS, ABLATION_CONDITIONS
    from .public import error_record
    if dataset not in DATASETS or condition not in ABLATION_CONDITIONS:
        raise ValueError("NATIVE_DIAGNOSTIC_IDENTITY_INVALID")
    record = {"dataset": dataset, "condition": condition, "model_variant": "dense",
        "status": "FAIL", "executed": False, "result_kind": "TEST/DIAGNOSTIC",
        "stages": {key: False for key in REQUIRED_STAGES}}
    prerequisites = ("torch", "numpy", "h5py", "tensorly", "tltorch", "opt_einsum", "scipy", "requests", "packaging")
    missing = [name for name in prerequisites if importlib.util.find_spec(name) is None]
    if missing:
        record.update(status="NOT_RUN", reason_code="NOT_RUN_NATIVE_ADAPTER_DEPENDENCIES_MISSING", missing_dependencies=missing)
        _record(record)
        return copy.deepcopy(record)
    try:
        _execute(dataset, condition, Path(output_root), seed, record)
    except Exception as exc:
        record.update(status="FAIL", reason_code=error_record(exc)["reason_code"])
        _record(record)
        raise
    record["status"] = "PASS"
    _record(record)
    return copy.deepcopy(record)


def _execute(dataset, condition, output_root, seed, record):
    from .contracts import DATASETS
    from .repository import resolve_repository_root, source_state, import_audit
    from .runtime import configure_runtime, apply_torch_runtime, output_path
    configure_runtime(cpu=True, profile="diagnostic", threads=1)
    import torch
    apply_torch_runtime()
    from .aggregate import aggregate
    from .artifacts import (build_artifact_contract, write_artifact, validate_artifact,
                            restore_validated_artifact)
    from .configs import model_configuration
    from .models import build_model_from_kwargs, preserved_rng_state
    from .protocols import TestContract, pinned_contract
    from .runtime_metadata import capture_runtime_metadata
    from .train import run_directory, _train_with_loop, _train_with_neuralop
    spec = DATASETS[dataset]
    module = importlib.import_module(spec.train_module)
    native = pinned_contract(dataset)["training_configuration"]
    configuration = model_configuration(dataset, "dense")
    contract = TestContract("TEST_NATIVE_" + dataset.upper() + "_" + condition.upper() + "_ONE_EPOCH",
        configuration, epochs=1, sample_count=2, horizon=int(getattr(module, "ROLLOUT", 1)),
        optimizer=native["optimizer"], training_loss=native["training_loss"],
        scheduler_t_max=int(native["scheduler_t_max"]), scheduler_step_policy=native["scheduler_step_policy"])
    validate_diagnostic_budget(contract)
    protocol = pinned_contract(dataset, test_contract=contract)
    run_dir = run_directory(output_path(output_root), dataset=dataset, model_variant="dense",
                            condition=condition, seed=seed, diagnostic=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    with preserved_rng_state():
        module.set_seed(seed)
        data, observers, shapes = _synthetic_data(dataset, module, contract)
        built = build_model_from_kwargs(spatial_dim=spec.spatial_dim,
            configuration=configuration, condition=condition, device="cpu")
        model = built.model
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        gradients = {"finite_tensors": 0, "nonzero_tensors": 0}

        def observe_gradient(gradient):
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("NATIVE_DIAGNOSTIC_NONFINITE_GRADIENT")
            gradients["finite_tensors"] += 1
            gradients["nonzero_tensors"] += int(bool(torch.count_nonzero(gradient)))

        handles = [parameter.register_hook(observe_gradient) for parameter in model.parameters()
                   if parameter.requires_grad]
        record["executed"] = True
        try:
            trainer = _train_with_loop if spec.trainer_kind == "loop" else _train_with_neuralop
            history, optimizer, scheduler = trainer(module, model=model, data=data,
                device=torch.device("cpu"), csv_path=run_dir / "training_log.csv", test_contract=contract)
        finally:
            for handle in handles:
                handle.remove()
        changed = sum(not torch.equal(initial[name], parameter)
                      for name, parameter in model.named_parameters())
        counts = {name: {"iterations": value.iterations, "batches": value.batches, "samples": value.samples}
                  for name, value in observers.items()}
        if (len(history) != 1 or counts["train"]["batches"] != 1 or counts["train"]["samples"] != 2
                or not changed or not gradients["nonzero_tensors"]
                or counts["test_evaluation"]["samples"] != 2):
            raise ValueError("NATIVE_DIAGNOSTIC_EXECUTION_EVIDENCE_INCOMPLETE")
        expected_scheduler_steps = 2 if dataset == "airfoil" else 1
        if scheduler.last_epoch != expected_scheduler_steps or int(module.EPOCHS) != 500:
            raise ValueError("NATIVE_DIAGNOSTIC_SCHEDULER_OR_PARENT_PROTOCOL_CHANGED")
        optimizer_steps = sorted({int(state["step"]) for state in optimizer.state.values()})
        if optimizer_steps != [1]:
            raise ValueError("NATIVE_DIAGNOSTIC_OPTIMIZER_STEPS_INVALID")
        if dataset in ("darcy", "ns2d") and any(
                value < 1 for value in shapes["processor_execution"].values()):
            raise ValueError("NATIVE_DIAGNOSTIC_PROCESSOR_NOT_EXECUTED")
        record["stages"].update(native_training=True, native_loss=True, native_evaluation=True)
        source = source_state(resolve_repository_root(), formal=False)
        runtime = capture_runtime_metadata(profile="diagnostic", device="cpu", seed=seed)
        metadata = build_artifact_contract(dataset, "dense", condition,
            dataset_identity=protocol["dataset_identity"], runtime=runtime, source=source, test_contract=contract)
        metadata.update(seed=seed, result_kind="diagnostic", fixture_note=contract.label)
        summary = write_artifact(run_dir, metadata=metadata, model=model, optimizer=optimizer,
            scheduler=scheduler, final_metrics=history[-1], test_contract=contract)
        record["stages"]["checkpoint_write"] = True
        identity = dict(dataset=dataset, model_variant="dense", condition=condition,
                        seed=seed, allow_diagnostic=True, test_contract=contract)
        verified = validate_artifact(summary, **identity)
        restored, restored_artifact = restore_validated_artifact(summary, **identity)
        if any(not torch.equal(value, restored.model.state_dict()[name])
               for name, value in model.state_dict().items()):
            raise ValueError("NATIVE_DIAGNOSTIC_STRICT_RESTORE_MISMATCH")
        if restored_artifact.checkpoint_sha256 != verified.checkpoint_sha256:
            raise ValueError("NATIVE_DIAGNOSTIC_CHECKPOINT_IDENTITY_CHANGED")
        record["stages"]["strict_reload"] = True
        report = aggregate(results_root=output_root, dataset=dataset, model_variant="dense",
                            condition=condition, seeds=[seed], test_contract=contract)
        if (report["result_kind"] != "diagnostic_test" or report["seed_count"] != 1
                or any(row["sample_std"] is not None for row in report["metrics"].values())
                or not all(math.isfinite(value) for value in verified.metrics.values())):
            raise ValueError("NATIVE_DIAGNOSTIC_AGGREGATE_INVALID")
        record["stages"]["aggregate"] = True
        record.update(test_contract=contract.label, native_model_configuration=configuration,
            synthetic_input=shapes, loader_execution=counts, gradients=gradients,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            changed_parameter_tensors=changed, native_training_loss=native["training_loss"],
            optimizer=native["optimizer"], optimizer_step_count=optimizer_steps[0],
            optimizer_implementation=type(optimizer).__module__ + "." + type(optimizer).__name__,
            scheduler_t_max=scheduler.T_max, scheduler_step_count=scheduler.last_epoch,
            native_scheduler_step_policy=native["scheduler_step_policy"], completed_epochs=1,
            parent_epochs=500, finite_metrics=verified.metrics, checkpoint_sha256=verified.checkpoint_sha256,
            artifact_validator=True, aggregate_seed_count=1, sample_std=None,
            source={key: source[key] for key in ("base_source_commit", "sirenfno_commit",
                "ablation_source_sha256", "source_mode")}, runtime=runtime,
            import_sources=import_audit(resolve_repository_root()))


def main(argv=None):
    from .public import SafeArgumentParser, emit_json
    from .repository import activate_repository, resolve_repository_root
    from .runtime import output_path
    from .contracts import DATASETS
    parser = SafeArgumentParser(description="Execute eleven bounded native CPU synthetic diagnostics; no formal data.")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    activate_repository(resolve_repository_root(args.repository_root))
    reset_execution_evidence()
    cases = [(dataset, "full") for dataset in DATASETS] + list(ADDITIONAL_CASES)
    failures = []
    for dataset, condition in cases:
        try:
            run_native_diagnostic(dataset, condition=condition, output_root=args.output_root)
        except Exception as exc:
            # A failed native path is FAIL, never a dependency skip. Retain its
            # evidence and continue only the other independent one-epoch cases.
            from .public import error_record
            failures.append({"dataset": dataset, "condition": condition, **error_record(exc)})
    evidence = execution_evidence()
    statuses = [item["status"] for item in evidence.values()]
    statuses.extend(extra["status"] for item in evidence.values()
                    for extra in item.get("additional_conditions", {}).values())
    status = ("FAIL" if failures or "FAIL" in statuses or len(statuses) != len(cases)
              else "NOT_RUN" if "NOT_RUN" in statuses else "PASS")
    report = {"status": status, "result_kind": "TEST/DIAGNOSTIC", "native_adapters": evidence,
              "case_count": len(cases), "case_failures": failures}
    from .artifacts import canonical_json
    destination = output_path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(report) + "\n")
    emit_json(report)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    from .public import public_main
    # The train seam imports this module by its canonical package name. Use
    # those same loader/budget classes when runpy invokes the public CLI.
    from .native_diagnostics import main as canonical_main
    public_main(canonical_main)
