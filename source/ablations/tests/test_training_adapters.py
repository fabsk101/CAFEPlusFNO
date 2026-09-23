from __future__ import annotations

import importlib
import inspect
import unittest
import importlib.util
import ast
import tempfile
from pathlib import Path

from ablations.contracts import DATASETS, FORMAL_EPOCHS
from ablations.configs import protocol_configuration
from ablations.repository import resolve_repository_root


class TrainingAdapterContractTests(unittest.TestCase):
    def test_every_dataset_import_callable_signature_only(self) -> None:
        """This test does not claim that any training/evaluation ran."""
        missing = [name for name in ("numpy", "h5py", "tensorly", "tltorch", "opt_einsum", "scipy", "requests", "packaging")
                   if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest("NOT_RUN_ADAPTER_IMPORT_DEPENDENCIES_" + "_".join(name.upper() for name in missing))
        for key, spec in DATASETS.items():
            with self.subTest(dataset=key):
                module = importlib.import_module(spec.train_module)
                self.assertEqual(int(module.EPOCHS), FORMAL_EPOCHS)
                self.assertTrue(callable(getattr(module, spec.verify_function)))
                self.assertTrue(callable(getattr(module, spec.load_function)))
                self.assertTrue(callable(module.resolve_device))
                self.assertTrue(callable(module.set_seed))
                if spec.trainer_kind == "loop":
                    signature = inspect.signature(module.run_training_loop)
                    self.assertTrue(
                        {"model", "data", "optimizer", "scheduler", "device", "csv_path"}
                        <= set(signature.parameters)
                    )
                else:
                    self.assertTrue(callable(module.CsvLoggingTrainer))
                    self.assertTrue(callable(module.LpLoss))
                    self.assertTrue(callable(module.H1Loss))
                    self.assertTrue(callable(module.AdamW))

    def test_all_seven_protocols_and_adapter_signatures_in_source(self):
        root = resolve_repository_root()
        for key, spec in DATASETS.items():
            with self.subTest(dataset=key):
                protocol = protocol_configuration(key)
                self.assertEqual(protocol["epochs"], 500)
                tree = ast.parse((root / Path(*spec.train_module.split('.'))).with_suffix('.py').read_text(encoding='utf-8'))
                nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                self.assertIn(spec.load_function, nodes)
                if spec.key != 'darcy':
                    self.assertIn(spec.verify_function, nodes)
                if spec.trainer_kind == 'loop':
                    arguments = nodes['run_training_loop'].args
                    self.assertTrue({'model', 'data', 'optimizer', 'scheduler', 'device', 'csv_path'} <=
                                    {arg.arg for arg in arguments.args + arguments.kwonlyargs})
                else:
                    self.assertIn('CsvLoggingTrainer', nodes)

    def test_darcy_native_cpu_training_evaluator_writer_validator_aggregate(self):
        """One real native trainer epoch under an explicit synthetic contract.

        Uses the native normalized LpLoss, decoded L2/H1 evaluator, optimizer,
        scheduler and CSV callback; no parent globals or validator are patched.
        Synthetic data and one epoch never qualify as formal dataset evidence.
        """
        missing = [name for name in ("numpy", "h5py", "tensorly", "tltorch", "opt_einsum", "scipy", "requests", "packaging")
                   if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest("NOT_RUN_DARCY_NATIVE_ADAPTER_DEPENDENCIES_" + "_".join(name.upper() for name in missing))
        # Only the declared prerequisite check above may skip. Nested import
        # errors from an installed dependency or parent implementation must fail.
        module = importlib.import_module("experiments.train_darcy")
        import torch
        from torch.utils.data import DataLoader
        from neuralop.data.transforms.data_processors import DefaultDataProcessor
        from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
        from ablations.aggregate import aggregate
        from ablations.artifacts import build_artifact_contract, validate_artifact, write_artifact
        from ablations.models import build_model_from_kwargs, preserved_rng_state
        from ablations.protocols import TestContract, pinned_contract
        from ablations.repository import source_state
        from ablations.runtime_metadata import capture_runtime_metadata
        from ablations.tests.test_models import tiny_config
        from ablations.train import run_directory

        contract = TestContract("TEST_DARCY_NATIVE_ADAPTER_ONE_EPOCH", tiny_config(2), epochs=1,
                                optimizer="NeuralOperator AdamW", training_loss="LpLoss(d=2,p=2)")
        protocol = pinned_contract("darcy", test_contract=contract)
        with tempfile.TemporaryDirectory() as directory, preserved_rng_state():
            root = Path(directory)
            run_dir = run_directory(root, dataset="darcy", model_variant="dense", condition="full",
                                    seed=42, diagnostic=True)
            run_dir.mkdir(parents=True)
            module.set_seed(42)
            x = torch.rand(2, 1, 5, 6) + 0.5
            y = torch.rand(2, 1, 5, 6) + 1.0
            loader = DataLoader([{"x": x[i], "y": y[i]} for i in range(2)], batch_size=2,
                                shuffle=False, num_workers=0)
            output_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
            output_normalizer.fit(y)
            # Pinned Darcy load_darcy_flow_small defaults: raw inputs and
            # channel-wise output encoding, applied by DefaultDataProcessor.
            processor = DefaultDataProcessor(in_normalizer=None,
                                             out_normalizer=output_normalizer).to("cpu")
            built = build_model_from_kwargs(spatial_dim=2, configuration=contract.base_configuration,
                                            condition="full", device="cpu")
            before = {name: parameter.detach().clone() for name, parameter in built.model.named_parameters()}
            optimizer = module.AdamW(built.model.parameters(), lr=module.LEARNING_RATE,
                                     weight_decay=module.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            trainer = module.CsvLoggingTrainer(csv_path=run_dir / "training_log.csv", model=built.model,
                n_epochs=1, device="cpu", data_processor=processor, wandb_log=False,
                eval_interval=1, use_distributed=False, verbose=False)
            trainer.train(train_loader=loader, test_loaders={"synthetic": loader}, optimizer=optimizer,
                          scheduler=scheduler, regularizer=False, training_loss=module.LpLoss(d=2, p=2),
                          eval_losses={"h1": module.H1Loss(d=2), "l2": module.LpLoss(d=2, p=2)})
            self.assertEqual(len(trainer.history), 1)
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in built.model.named_parameters()))
            metadata = build_artifact_contract("darcy", "dense", "full",
                dataset_identity=protocol["dataset_identity"],
                runtime=capture_runtime_metadata(profile="diagnostic", device="cpu"),
                source=source_state(resolve_repository_root(), formal=False), test_contract=contract)
            metadata.update(seed=42, result_kind="diagnostic",
                            fixture_note="TEST_NATIVE_ADAPTER_ONE_EPOCH_SYNTHETIC_INPUTS")
            summary_path = write_artifact(run_dir, metadata=metadata, model=built.model,
                optimizer=optimizer, scheduler=scheduler, final_metrics=trainer.history[-1], test_contract=contract)
            artifact = validate_artifact(summary_path, dataset="darcy", model_variant="dense",
                condition="full", seed=42, allow_diagnostic=True, test_contract=contract)
            report = aggregate(results_root=root, dataset="darcy", model_variant="dense", condition="full",
                               seeds=[42], test_contract=contract)
            self.assertEqual(report["result_kind"], "diagnostic_test")
            self.assertEqual(report["seed_count"], 1)
            self.assertTrue(artifact.metrics)
            self.assertTrue(all(row["sample_std"] is None for row in report["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
