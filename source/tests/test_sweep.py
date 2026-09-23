"""Paper-sweep dispatch and fail-closed artifact completion tests."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.common.checkpoints import (
    load_weights_only_checkpoint,
    normalize_environment_metadata,
    prepare_weights_only_checkpoint,
)
from experiments.common.dataset_provenance import dataset_manifest
from experiments.common.sirenfno_backend import upstream_manifest
from experiments.configs.airfoil import (
    MODEL_CHOICES as AIRFOIL_MODEL_CHOICES,
    PAPER_SEEDS as AIRFOIL_PAPER_SEEDS,
)
from experiments.configs.seeds import PAPER_SEEDS
from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D
from scripts import run_cafe_darcy_sweep
from scripts import run_paper_sweep


class TinyPaperModel(torch.nn.Module):
    """Small real model used to exercise strict state/buffer loading."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.register_buffer("required_scale", torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value) * self.required_scale


TINY_MODEL_CONFIGURATION = {
    "in_features": 2,
    "out_features": 1,
    "bias": True,
    "required_buffer": "required_scale",
}


def tiny_model_configuration(model: str) -> dict[str, object]:
    if model != "tiny":
        raise KeyError(model)
    return dict(TINY_MODEL_CONFIGURATION)


def tiny_model_builder(model: str, device: torch.device) -> SimpleNamespace:
    if model != "tiny":
        raise KeyError(model)
    return SimpleNamespace(
        model=TinyPaperModel().to(device),
        configuration=dict(TINY_MODEL_CONFIGURATION),
        factorization="dense",
        rank=None,
    )


CAFE_CHECKPOINT_CONFIGURATION = {
    "width": 4,
    "input_dim": 2,
    "output_dim": 1,
    "num_layers": 1,
    "ffn_expansion": 2,
    "rff_basis": 3,
    "cheb_basis": 4,
    "cafe_branches": 2,
    "cafe_branch_dim": 5,
    "kernel_hidden_dim": 6,
    "factor_rff_basis": 16,
    "factor_cheb_basis": 8,
    "factor_branch_dim": 12,
    "factor_hidden_dim": 16,
    "factor_output_init_mode": "xavier",
    "factor_kernel_target_rms": 1e-3,
    "factorization": "cp",
    "rank": 2,
    "input_layout": "channels_first",
    "output_layout": "channels_first",
}


class UnsupportedCheckpointMetadata:
    """Pickleable test sentinel which restricted checkpoint loading must reject."""


class CheckpointSerializationTests(unittest.TestCase):
    def assert_exact_builtin_metadata(self, value: object) -> None:
        value_type = type(value)
        if value is None or value_type in {str, bool, int, float}:
            return
        if value_type is dict:
            for key, item in value.items():
                self.assertIs(type(key), str)
                self.assert_exact_builtin_metadata(item)
            return
        if value_type in {list, tuple}:
            for item in value:
                self.assert_exact_builtin_metadata(item)
            return
        self.fail(f"non-built-in metadata type: {value_type!r}")

    def assert_nested_state_equal(self, expected: object, actual: object) -> None:
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            return
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_state_equal(expected[key], actual[key])
            return
        if isinstance(expected, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self.assert_nested_state_equal(expected_item, actual_item)
            return
        self.assertEqual(expected, actual)

    def test_all_seven_environment_records_use_exact_builtin_types(self) -> None:
        from experiments import (
            train_airfoil,
            train_burgers,
            train_cfd1d,
            train_cfd2d,
            train_darcy,
            train_ns2d,
            train_reacdiff,
        )

        writers = (
            train_airfoil,
            train_burgers,
            train_cfd1d,
            train_cfd2d,
            train_darcy,
            train_ns2d,
            train_reacdiff,
        )
        source_state = {
            "source_repository_commit": "a" * 40,
            "source_repository_dirty": False,
        }
        for writer in writers:
            with self.subTest(writer=writer.__name__):
                environment = writer.environment_metadata(source_state)
                self.assertIs(type(environment), dict)
                self.assertIs(type(environment["pytorch_version"]), str)
                self.assertEqual(
                    environment["pytorch_version"], str(torch.__version__)
                )
                self.assert_exact_builtin_metadata(environment)

    def test_all_seven_writers_save_the_prepared_checkpoint_value(self) -> None:
        from experiments import (
            train_airfoil,
            train_burgers,
            train_cfd1d,
            train_cfd2d,
            train_darcy,
            train_ns2d,
            train_reacdiff,
        )

        writers = (
            train_airfoil,
            train_burgers,
            train_cfd1d,
            train_cfd2d,
            train_darcy,
            train_ns2d,
            train_reacdiff,
        )
        for writer in writers:
            with self.subTest(writer=writer.__name__):
                module_tree = ast.parse(
                    Path(writer.__file__).read_text(encoding="utf-8")
                )
                main = next(
                    node
                    for node in module_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main"
                )
                save_sites: list[tuple[int, ast.Call]] = []
                for index, statement in enumerate(main.body):
                    if not isinstance(statement, ast.Expr):
                        continue
                    call = statement.value
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "torch"
                        and call.func.attr == "save"
                    ):
                        save_sites.append((index, call))
                self.assertEqual(len(save_sites), 1)
                save_index, save_call = save_sites[0]
                self.assertTrue(save_call.args)
                self.assertIsInstance(save_call.args[0], ast.Name)
                checkpoint_name = save_call.args[0].id

                reaching_assignments = []
                for statement in main.body[:save_index]:
                    if not isinstance(statement, ast.Assign):
                        continue
                    if any(
                        isinstance(target, ast.Name)
                        and target.id == checkpoint_name
                        for target in statement.targets
                    ):
                        reaching_assignments.append(statement)
                self.assertTrue(reaching_assignments)
                reaching_value = reaching_assignments[-1].value
                self.assertIsInstance(reaching_value, ast.Call)
                self.assertIsInstance(reaching_value.func, ast.Name)
                self.assertEqual(
                    reaching_value.func.id, "prepare_weights_only_checkpoint"
                )

    def test_actual_cafe_model_optimizer_scheduler_weights_only_roundtrip(
        self,
    ) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            model = CAFEPlusFNO1D(**CAFE_CHECKPOINT_CONFIGURATION)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=4
            )
            inputs = torch.linspace(-1.0, 1.0, 36).reshape(2, 2, 9)
            targets = torch.linspace(0.5, -0.5, 18).reshape(2, 1, 9)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.assertTrue(optimizer.state)
            self.assertEqual(scheduler.last_epoch, 1)

            model.eval()
            with torch.no_grad():
                expected_output = model(inputs).clone()
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            environment = normalize_environment_metadata(
                {
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                }
            )
            checkpoint = prepare_weights_only_checkpoint(
                {
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer_state,
                    "scheduler_state_dict": scheduler_state,
                    "model_configuration": CAFE_CHECKPOINT_CONFIGURATION,
                    "epoch": 1,
                    "environment": environment,
                }
            )

            # Model state is shallow-copied so upstream mapping metadata can be
            # isolated without mutating the original. Tensor identity and the
            # optimizer/scheduler containers remain unchanged.
            self.assertIsNot(checkpoint["model_state_dict"], model_state)
            self.assertEqual(
                checkpoint["model_state_dict"].keys(), model_state.keys()
            )
            for key in model_state:
                self.assertIs(checkpoint["model_state_dict"][key], model_state[key])
            self.assertIs(checkpoint["optimizer_state_dict"], optimizer_state)
            self.assertIs(checkpoint["scheduler_state_dict"], scheduler_state)

            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "roundtrip.pt"
                torch.save(checkpoint, checkpoint_path)
                direct = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                loaded_result = load_weights_only_checkpoint(checkpoint_path)

            self.assertEqual(
                loaded_result.load_metadata,
                {
                    "checkpoint_load_mode": "weights_only",
                    "legacy_torch_version_compatibility_used": False,
                    "legacy_neuraloperator_metadata_compatibility_used": False,
                },
            )
            self.assertIs(
                type(direct["environment"]["pytorch_version"]), str
            )
            self.assert_exact_builtin_metadata(direct["environment"])
            loaded = loaded_result.checkpoint

            restored_model = CAFEPlusFNO1D(**loaded["model_configuration"])
            incompatible = restored_model.load_state_dict(
                loaded["model_state_dict"], strict=True
            )
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=2e-3
            )
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=4
            )
            restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
            restored_scheduler.load_state_dict(loaded["scheduler_state_dict"])

            self.assert_nested_state_equal(
                optimizer_state, restored_optimizer.state_dict()
            )
            self.assert_nested_state_equal(
                scheduler_state, restored_scheduler.state_dict()
            )
            expected_parameters = dict(model.named_parameters())
            actual_parameters = dict(restored_model.named_parameters())
            self.assertEqual(expected_parameters.keys(), actual_parameters.keys())
            for name in expected_parameters:
                torch.testing.assert_close(
                    expected_parameters[name],
                    actual_parameters[name],
                    rtol=0,
                    atol=0,
                )
            expected_buffers = dict(model.named_buffers())
            actual_buffers = dict(restored_model.named_buffers())
            self.assertTrue(expected_buffers)
            self.assertEqual(expected_buffers.keys(), actual_buffers.keys())
            for name in expected_buffers:
                torch.testing.assert_close(
                    expected_buffers[name], actual_buffers[name], rtol=0, atol=0
                )
            restored_model.eval()
            with torch.no_grad():
                actual_output = restored_model(inputs)
            torch.testing.assert_close(
                expected_output, actual_output, rtol=0, atol=0
            )

    def test_pinned_fno_and_tfno_cp_1d_2d_safe_roundtrip(self) -> None:
        from experiments.common.sirenfno_backend import (
            bootstrap_sirenfno_backend,
        )

        bootstrap_sirenfno_backend()
        from neuralop.models import FNO

        for dimensions in (1, 2):
            for factorization in (None, "cp"):
                with self.subTest(
                    dimensions=dimensions,
                    factorization=factorization or "dense",
                ), torch.random.fork_rng():
                    torch.manual_seed(3100 + dimensions)
                    configuration = {
                        "n_modes": (3,) * dimensions,
                        "in_channels": 1,
                        "out_channels": 1,
                        "hidden_channels": 4,
                        "n_layers": 1,
                        "lifting_channels": 8,
                        "projection_channels": 8,
                    }
                    if factorization is not None:
                        configuration.update(
                            {
                                "implementation": "factorized",
                                "factorization": factorization,
                                "rank": 0.5,
                            }
                        )
                    model = FNO(**configuration)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=4
                    )
                    shape = (2, 1) + (8,) * dimensions
                    inputs = torch.randn(shape)
                    targets = torch.randn(shape)
                    optimizer.zero_grad(set_to_none=True)
                    (model(inputs) - targets).square().mean().backward()
                    optimizer.step()
                    scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        expected_output = model(inputs).clone()
                    original_model_state = model.state_dict()
                    original_mapping_metadata = copy.deepcopy(
                        original_model_state._metadata
                    )
                    optimizer_state = optimizer.state_dict()
                    scheduler_state = scheduler.state_dict()
                    checkpoint = prepare_weights_only_checkpoint(
                        {
                            "model_state_dict": original_model_state,
                            "optimizer_state_dict": optimizer_state,
                            "scheduler_state_dict": scheduler_state,
                            "model_configuration": configuration,
                            "epoch": 1,
                        }
                    )

                    self.assertIn("_metadata", original_model_state)
                    self.assertNotIn("_metadata", checkpoint["model_state_dict"])
                    self.assertEqual(
                        checkpoint["model_state_dict"]._metadata,
                        original_mapping_metadata,
                    )
                    self.assertEqual(
                        checkpoint["model_initialization_metadata"][
                            "conv_module"
                        ],
                        "neuralop.layers.spectral_convolution.SpectralConv",
                    )
                    self.assertEqual(
                        checkpoint["model_initialization_metadata"][
                            "non_linearity"
                        ],
                        "torch._C._nn.gelu",
                    )
                    self.assertIs(
                        checkpoint["optimizer_state_dict"], optimizer_state
                    )
                    self.assertIs(
                        checkpoint["scheduler_state_dict"], scheduler_state
                    )

                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        safe_path = root / "safe.pt"
                        torch.save(checkpoint, safe_path)
                        direct = torch.load(
                            safe_path, map_location="cpu", weights_only=True
                        )
                        loaded = load_weights_only_checkpoint(
                            safe_path
                        ).checkpoint

                        legacy_path = root / "legacy.pt"
                        torch.save(
                            {
                                "model_state_dict": original_model_state,
                                "optimizer_state_dict": optimizer_state,
                                "scheduler_state_dict": scheduler_state,
                                "model_configuration": configuration,
                            },
                            legacy_path,
                        )
                        legacy_bytes = legacy_path.read_bytes()
                        with self.assertRaises(pickle.UnpicklingError):
                            load_weights_only_checkpoint(legacy_path)
                        safe_globals_before = set(
                            torch.serialization.get_safe_globals()
                        )
                        legacy_result = load_weights_only_checkpoint(
                            legacy_path,
                            allow_legacy_neuraloperator_metadata=True,
                        )
                        self.assertEqual(
                            set(torch.serialization.get_safe_globals()),
                            safe_globals_before,
                        )
                        self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
                        self.assertTrue(
                            legacy_result.load_metadata[
                                "legacy_neuraloperator_metadata_compatibility_used"
                            ]
                        )
                        self.assertNotIn(
                            "_metadata",
                            legacy_result.checkpoint["model_state_dict"],
                        )

                    self.assertNotIn("_metadata", direct["model_state_dict"])
                    restored = FNO(**loaded["model_configuration"])
                    incompatible = restored.load_state_dict(
                        loaded["model_state_dict"], strict=True
                    )
                    self.assertEqual(incompatible.missing_keys, [])
                    self.assertEqual(incompatible.unexpected_keys, [])
                    restored_optimizer = torch.optim.AdamW(
                        restored.parameters(), lr=2e-3
                    )
                    restored_scheduler = (
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            restored_optimizer, T_max=4
                        )
                    )
                    restored_optimizer.load_state_dict(
                        loaded["optimizer_state_dict"]
                    )
                    restored_scheduler.load_state_dict(
                        loaded["scheduler_state_dict"]
                    )
                    self.assert_nested_state_equal(
                        optimizer_state, restored_optimizer.state_dict()
                    )
                    self.assert_nested_state_equal(
                        scheduler_state, restored_scheduler.state_dict()
                    )
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            parameter,
                            dict(restored.named_parameters())[name],
                            rtol=0,
                            atol=0,
                        )
                    for name, buffer in model.named_buffers():
                        torch.testing.assert_close(
                            buffer,
                            dict(restored.named_buffers())[name],
                            rtol=0,
                            atol=0,
                        )
                    restored.eval()
                    with torch.no_grad():
                        actual_output = restored(inputs)
                    torch.testing.assert_close(
                        expected_output, actual_output, rtol=0, atol=0
                    )

    def test_model_state_rejects_unexpected_non_tensor_entries(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a Tensor"):
            prepare_weights_only_checkpoint(
                {
                    "model_state_dict": {
                        "weight": torch.ones(1),
                        "unexpected": "not-a-weight",
                    }
                }
            )
        with self.assertRaises(TypeError):
            prepare_weights_only_checkpoint(
                {
                    "model_state_dict": {
                        "weight": torch.ones(1),
                        "_metadata": {"unexpected_callable": torch._C._nn.gelu},
                    }
                }
            )

    def test_legacy_torch_version_retry_is_explicit_and_reported(self) -> None:
        legacy = {
            "model_state_dict": {"weight": torch.ones(1)},
            "environment": {"pytorch_version": torch.__version__},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"
            torch.save(legacy, checkpoint_path)
            with self.assertRaises(pickle.UnpicklingError):
                load_weights_only_checkpoint(checkpoint_path)
            result = load_weights_only_checkpoint(
                checkpoint_path, allow_legacy_torch_version=True
            )

        self.assertEqual(
            result.load_metadata,
            {
                "checkpoint_load_mode": "weights_only",
                "legacy_torch_version_compatibility_used": True,
                "legacy_neuraloperator_metadata_compatibility_used": False,
            },
        )
        version = result.checkpoint["environment"]["pytorch_version"]
        self.assertIs(type(version), str)
        self.assertEqual(version, str(torch.__version__))

    def test_legacy_retry_does_not_allow_unrelated_pickle_globals(self) -> None:
        payload = {
            "model_state_dict": {"weight": torch.ones(1)},
            "metadata": UnsupportedCheckpointMetadata(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "unsupported.pt"
            torch.save(payload, checkpoint_path)
            with self.assertRaises(pickle.UnpicklingError):
                load_weights_only_checkpoint(
                    checkpoint_path,
                    allow_legacy_torch_version=True,
                    allow_legacy_neuraloperator_metadata=True,
                )


class SweepDispatchTests(unittest.TestCase):
    def test_all_seven_datasets_share_the_guarded_completion_contract(self) -> None:
        self.assertEqual(
            set(run_paper_sweep.DATASETS),
            {
                "airfoil221x51",
                "burgers1024",
                "cfd1d1024",
                "cfd2d128",
                "darcy128",
                "ns128",
                "reacdiff1024",
            },
        )
        for dataset, spec in run_paper_sweep.DATASETS.items():
            with self.subTest(dataset=dataset):
                self.assertIs(spec["require_current_source_commit"], True)
                self.assertIn("training_protocol_version", spec)
                self.assertIn("evaluation_protocol_version", spec)
                self.assertIn("training_configuration", spec)
                self.assertIn("data_configuration", spec)
                self.assertTrue(callable(spec["model_configuration_factory"]))
                self.assertTrue(spec["metric_columns"])
                self.assertTrue(spec["paper_metric_columns"])

    def test_airfoil_sweep_spec_is_eleven_models_by_five_seeds(self) -> None:
        spec = run_paper_sweep.DATASETS["airfoil221x51"]
        self.assertEqual(tuple(spec["models"]), AIRFOIL_MODEL_CHOICES)
        self.assertEqual(len(AIRFOIL_MODEL_CHOICES) * len(AIRFOIL_PAPER_SEEDS), 55)

    def test_burgers_amfno64_uses_a_separate_default_result_root(self) -> None:
        spec = run_paper_sweep.DATASETS["burgers1024"]
        self.assertIn("protocol_v2", str(spec["default_results"]).lower())
        self.assertIn("amfno64", spec["training_protocol_version"].lower())

    def test_writer_contract_separates_checkpoint_hash_and_environment(self) -> None:
        source = {
            "source_repository_commit": "a" * 40,
            "source_repository_dirty": False,
        }
        contract = run_paper_sweep.run_contract_metadata("cfd1d1024", source)
        self.assertEqual(
            contract["training_source_commit"],
            contract["evaluation_source_commit"],
        )
        self.assertEqual(
            contract["evaluation_protocol_version"],
            run_paper_sweep.CORRECTED_RELATIVE_L2_VERSION,
        )
        self.assertNotIn("checkpoint_sha256", contract)
        environment = run_paper_sweep.run_environment_provenance(contract)
        self.assertEqual(environment["training_source_commit"], "a" * 40)
        self.assertNotIn("evaluation_source_commit", environment)

    def test_unknown_model_override_is_rejected(self) -> None:
        args = argparse.Namespace(
            dataset="airfoil221x51",
            data_root=Path("data"),
            results_root=None,
            models=["ufno"],
            seeds=[42],
            device="cpu",
        )
        with patch.object(run_paper_sweep, "parse_args", return_value=args):
            with self.assertRaisesRegex(SystemExit, "Unknown models: ufno"):
                run_paper_sweep.main()

    def test_default_airfoil_sweep_dispatches_fifty_five_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                dataset="airfoil221x51",
                data_root=Path("data"),
                results_root=Path(directory),
                models=None,
                seeds=list(AIRFOIL_PAPER_SEEDS),
                device="cpu",
            )
            completions = [value for _ in range(55) for value in (False, True)]
            source = {
                "source_repository_commit": "a" * 40,
                "source_repository_dirty": False,
            }
            with (
                patch.object(run_paper_sweep, "parse_args", return_value=args),
                patch.object(
                    run_paper_sweep, "source_repository_state", return_value=source
                ) as source_state,
                patch.object(
                    run_paper_sweep, "is_complete", side_effect=completions
                ),
                patch.object(run_paper_sweep.subprocess, "run") as dispatched,
            ):
                dispatched.return_value.returncode = 0
                run_paper_sweep.main()
        source_state.assert_called_once_with(allow_dirty=False)
        self.assertEqual(dispatched.call_count, 55)
        for call in dispatched.call_args_list:
            command = call.args[0]
            self.assertIn("experiments.train_airfoil", command)
            self.assertNotIn("ufno", command)

    def test_dirty_source_rejects_every_paper_sweep_before_dispatch(self) -> None:
        for dataset, spec in run_paper_sweep.DATASETS.items():
            with self.subTest(dataset=dataset):
                args = argparse.Namespace(
                    dataset=dataset,
                    data_root=Path("data"),
                    results_root=Path("results"),
                    models=[spec["models"][0]],
                    seeds=[42],
                    device="cpu",
                )
                source = {
                    "source_repository_commit": "a" * 40,
                    "source_repository_dirty": True,
                }
                with (
                    patch.object(run_paper_sweep, "parse_args", return_value=args),
                    patch.object(
                        run_paper_sweep,
                        "source_repository_state",
                        return_value=source,
                    ),
                    patch.object(run_paper_sweep.subprocess, "run") as dispatched,
                ):
                    with self.assertRaisesRegex(RuntimeError, "must be clean"):
                        run_paper_sweep.main()
                    dispatched.assert_not_called()


class PaperSweepCompletionFixture(unittest.TestCase):
    """Create one internally consistent three-epoch paper artifact set."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model = "tiny"
        self.seed = 42
        self.training_commit = "a" * 40
        self.spec = copy.deepcopy(run_paper_sweep.DATASETS["cfd1d1024"])
        self.spec.update(
            {
                "module": "unused.test.module",
                "models": (self.model,),
                "epochs": 3,
                "experiment": "tiny_completion_fixture",
                "training_protocol_version": "tiny_training_v1",
                "evaluation_protocol_version": (
                    run_paper_sweep.CORRECTED_RELATIVE_L2_VERSION
                ),
                "model_configuration_factory": tiny_model_configuration,
                "model_builder": tiny_model_builder,
            }
        )
        self.spec["training_configuration"] = {
            **self.spec["training_configuration"],
            "epochs": 3,
            "batch_size": 2,
        }
        self.run_dir = (
            Path(self.temporary_directory.name)
            / self.model
            / f"seed_{self.seed}"
        )
        self.run_dir.mkdir(parents=True)

        canonical = copy.deepcopy(
            dataset_manifest()[str(self.spec["dataset_id"])]
        )
        canonical["dataset_id"] = self.spec["dataset_id"]
        self.dataset_identity = canonical

        model = TinyPaperModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones(2, 2)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        scheduler.step()
        model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        upstream_commit = upstream_manifest()["SirenFNO"]["commit"]

        shared = {
            "completion_schema_version": run_paper_sweep.COMPLETION_SCHEMA_VERSION,
            "experiment": self.spec["experiment"],
            "dataset_id": self.spec["dataset_id"],
            "dataset_identity": copy.deepcopy(self.dataset_identity),
            "data_configuration": copy.deepcopy(self.spec["data_configuration"]),
            "model": self.model,
            "model_class": model_class,
            "factorization": "dense",
            "rank": None,
            "parameter_count": parameter_count,
            "seed": self.seed,
            "experiment_seed": self.seed,
            "training_protocol_version": self.spec["training_protocol_version"],
            "training_protocol_identifier": (
                run_paper_sweep.training_protocol_identifier(self.spec)
            ),
            "training_configuration": copy.deepcopy(
                self.spec["training_configuration"]
            ),
            **{
                key: copy.deepcopy(self.spec["training_configuration"][key])
                for key in (
                    "epochs",
                    "batch_size",
                    "optimizer",
                    "learning_rate",
                    "weight_decay",
                    "scheduler",
                    "scheduler_t_max",
                    "eval_interval",
                    "training_loss",
                )
            },
            "checkpoint_selection_policy": self.spec["training_configuration"][
                "checkpoint_selection_policy"
            ],
            "training_source_commit": self.training_commit,
            "training_source_dirty": False,
            "evaluation_schema_version": run_paper_sweep.EVALUATION_SCHEMA_VERSION,
            "evaluation_protocol_version": self.spec["evaluation_protocol_version"],
            "evaluation_source_commit": self.training_commit,
            "evaluation_source_dirty": False,
            **copy.deepcopy(self.spec["evaluation_metadata"]),
            "source_repository_commit": self.training_commit,
            "source_repository_dirty": False,
            "sirenfno_upstream_commit": upstream_commit,
        }
        self.metrics = {
            "final_train_corrected_step_relative_l2": 0.4,
            "final_train_corrected_trajectory_relative_l2": 0.3,
            "final_test_corrected_step_relative_l2": 0.5,
            "final_test_corrected_trajectory_relative_l2": 0.25,
        }
        self.summary = {
            **copy.deepcopy(shared),
            **self.metrics,
            "model_hyperparameters": copy.deepcopy(TINY_MODEL_CONFIGURATION),
            "final_epoch": self.spec["epochs"],
            "final_checkpoint": "final_checkpoint.pt",
            "training_log": "training_log.csv",
        }
        self.checkpoint = {
            **copy.deepcopy(shared),
            **self.metrics,
            "model_configuration": copy.deepcopy(TINY_MODEL_CONFIGURATION),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
            "epoch": self.spec["epochs"],
            "environment": {
                "training_source_commit": self.training_commit,
                "training_source_dirty": False,
                "source_repository_commit": self.training_commit,
                "source_repository_dirty": False,
                "sirenfno_upstream_commit": upstream_commit,
            },
        }
        self.fieldnames = (
            "epoch",
            *self.spec["metric_columns"].values(),
            "learning_rate",
            "train_time_seconds",
            "epoch_time_seconds",
        )
        self.rows = []
        for epoch in range(1, self.spec["epochs"] + 1):
            self.rows.append(
                {
                    "epoch": epoch,
                    **{
                        csv_name: (
                            self.metrics[metric_name]
                            if epoch == self.spec["epochs"]
                            else self.metrics[metric_name] + 0.1
                        )
                        for metric_name, csv_name in self.spec[
                            "metric_columns"
                        ].items()
                    },
                    "learning_rate": 1e-3,
                    "train_time_seconds": 0.01,
                    "epoch_time_seconds": 0.02,
                }
            )
        self._write_artifacts()

    def _write_artifacts(self) -> None:
        checkpoint_path = self.run_dir / "final_checkpoint.pt"
        torch.save(self.checkpoint, checkpoint_path)
        self.summary["checkpoint_sha256"] = run_paper_sweep.sha256_file(
            checkpoint_path
        )
        (self.run_dir / "summary.json").write_text(
            json.dumps(self.summary, allow_nan=False), encoding="utf-8"
        )
        with (self.run_dir / "training_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def _audit(self, **kwargs: object) -> run_paper_sweep.RunCompletionAudit:
        return run_paper_sweep.audit_run_completion(
            self.run_dir,
            spec=self.spec,
            model=self.model,
            seed=self.seed,
            current_source_commit=self.training_commit,
            **kwargs,
        )


class PaperSweepCompletionTests(PaperSweepCompletionFixture):
    def test_complete_fixture_strict_loads_and_passes(self) -> None:
        audit = self._audit()
        self.assertEqual(audit.status, run_paper_sweep.CompletionStatus.COMPLETE)
        self.assertTrue(audit.training_valid)
        self.assertTrue(audit.evaluation_valid)
        self.assertTrue(audit.complete)
        self.assertTrue(
            run_paper_sweep.is_complete(
                self.run_dir,
                spec=self.spec,
                model=self.model,
                seed=self.seed,
                current_source_commit=self.training_commit,
            )
        )

    def test_metadata_only_or_empty_checkpoint_fails(self) -> None:
        for model_state in (None, {}):
            with self.subTest(model_state=model_state):
                saved = self.checkpoint.pop("model_state_dict")
                if model_state is not None:
                    self.checkpoint["model_state_dict"] = model_state
                self._write_artifacts()
                audit = self._audit()
                self.assertEqual(
                    audit.status, run_paper_sweep.CompletionStatus.INCOMPLETE
                )
                self.assertFalse(audit.training_valid)
                self.checkpoint["model_state_dict"] = saved

    def test_missing_key_and_shape_mismatch_fail_strict_loading(self) -> None:
        original_state = copy.deepcopy(self.checkpoint["model_state_dict"])
        del self.checkpoint["model_state_dict"]["required_scale"]
        self._write_artifacts()
        self.assertIn("checkpoint.strict_model_load_failed", self._audit().reasons)

        self.checkpoint["model_state_dict"] = copy.deepcopy(original_state)
        self.checkpoint["model_state_dict"]["linear.weight"] = torch.zeros(2, 2)
        self._write_artifacts()
        self.assertIn("checkpoint.strict_model_load_failed", self._audit().reasons)

    def test_nan_and_inf_checkpoint_tensors_fail(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.checkpoint["model_state_dict"]["linear.weight"][0, 0] = value
                self._write_artifacts()
                self.assertIn(
                    "checkpoint.nonfinite_tensor_or_value", self._audit().reasons
                )
                self.checkpoint["model_state_dict"]["linear.weight"][0, 0] = 0.0

    def test_nan_and_inf_metrics_fail(self) -> None:
        metric_name = "final_test_corrected_trajectory_relative_l2"
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.checkpoint[metric_name] = value
                self._write_artifacts()
                audit = self._audit()
                self.assertEqual(
                    audit.status, run_paper_sweep.CompletionStatus.INCOMPLETE
                )
                self.checkpoint[metric_name] = self.metrics[metric_name]

    def test_nonfinite_csv_metric_fails(self) -> None:
        column = "test_corrected_trajectory_relative_l2"
        self.rows[-1][column] = "NaN"
        self._write_artifacts()
        audit = self._audit()
        self.assertEqual(
            audit.status, run_paper_sweep.CompletionStatus.INCOMPLETE
        )
        self.assertIn(f"csv.nonfinite_or_invalid:{column}", audit.reasons)

    def test_source_config_data_and_evaluation_mismatches_fail(self) -> None:
        mutations = (
            (self.summary, "training_source_commit", "b" * 40),
            (self.checkpoint["environment"], "training_source_commit", "b" * 40),
            (self.checkpoint, "training_protocol_version", "tiny_training_v0"),
            (self.summary["training_configuration"], "epochs", 99),
            (self.summary, "optimizer", "not_the_recorded_optimizer"),
            (self.checkpoint["data_configuration"], "prediction_horizon", 99),
            (self.summary, "evaluation_protocol_version", "unknown_v99"),
        )
        for container, key, value in mutations:
            with self.subTest(key=key, value=value):
                original = container[key]
                container[key] = value
                self._write_artifacts()
                self.assertFalse(self._audit().complete)
                container[key] = original

    def test_checkpoint_hash_mismatch_fails(self) -> None:
        self.summary["checkpoint_sha256"] = "0" * 64
        (self.run_dir / "summary.json").write_text(
            json.dumps(self.summary), encoding="utf-8"
        )
        self.assertIn("summary.checkpoint_sha256_mismatch", self._audit().reasons)

    def test_training_provenance_requires_every_explicit_strict_boolean(self) -> None:
        del self.summary["training_source_dirty"]
        self._write_artifacts()
        self.assertIn(
            "summary.invalid_training_source_dirty",
            self._audit().reasons,
        )

        self.summary["training_source_dirty"] = False
        self.checkpoint["environment"]["training_source_dirty"] = 0
        self._write_artifacts()
        self.assertIn(
            "environment.invalid_training_source_dirty",
            self._audit().reasons,
        )

    def test_source_alias_is_required_in_all_three_artifacts(self) -> None:
        del self.checkpoint["environment"]["source_repository_commit"]
        self._write_artifacts()
        self.assertIn(
            "environment.invalid_source_repository_commit",
            self._audit().reasons,
        )

    def test_dirty_training_requires_separate_diagnostic_permission(self) -> None:
        for artifact in (
            self.summary,
            self.checkpoint,
            self.checkpoint["environment"],
        ):
            artifact["training_source_dirty"] = True
            artifact["source_repository_dirty"] = True
        self._write_artifacts()

        with self.assertRaises(run_paper_sweep.TrainingArtifactValidationError):
            run_paper_sweep.validate_training_artifacts(
                self.run_dir,
                spec=self.spec,
                current_source_commit=self.training_commit,
            )
        validated = run_paper_sweep.validate_training_artifacts(
            self.run_dir,
            spec=self.spec,
            current_source_commit=self.training_commit,
            diagnostic_allow_dirty_training_source=True,
        )
        self.assertTrue(validated.training_source_dirty)
        self.assertEqual(
            validated.validation_status,
            "valid_dirty_training_source_diagnostic",
        )

    def test_common_reader_preserves_checkpoint_and_csv_identity(self) -> None:
        validated = run_paper_sweep.validate_training_artifacts(
            self.run_dir,
            spec=self.spec,
            current_source_commit=self.training_commit,
        )
        identity = validated.training_log_identity
        self.assertEqual(identity["row_count"], self.spec["epochs"])
        self.assertEqual(identity["first_epoch"], 1)
        self.assertEqual(identity["final_epoch"], self.spec["epochs"])
        self.assertEqual(
            identity["sha256"],
            run_paper_sweep.sha256_file(self.run_dir / "training_log.csv"),
        )
        self.assertEqual(
            validated.checkpoint_sha256,
            self.summary["checkpoint_sha256"],
        )
        self.assertEqual(
            validated.checkpoint_load_metadata,
            {
                "checkpoint_load_mode": "weights_only",
                "legacy_torch_version_compatibility_used": False,
                "legacy_neuraloperator_metadata_compatibility_used": False,
            },
        )

    def test_common_reader_legacy_torch_version_is_explicit_and_recorded(self) -> None:
        self.checkpoint["legacy_torch_version"] = torch.__version__
        self._write_artifacts()
        with self.assertRaises(run_paper_sweep.TrainingArtifactValidationError):
            run_paper_sweep.validate_training_artifacts(
                self.run_dir,
                spec=self.spec,
                current_source_commit=self.training_commit,
            )
        validated = run_paper_sweep.validate_training_artifacts(
            self.run_dir,
            spec=self.spec,
            current_source_commit=self.training_commit,
            allow_legacy_torch_version=True,
        )
        self.assertEqual(
            validated.checkpoint_load_metadata,
            {
                "checkpoint_load_mode": "weights_only",
                "legacy_torch_version_compatibility_used": True,
                "legacy_neuraloperator_metadata_compatibility_used": False,
            },
        )

    def test_missing_duplicate_and_incomplete_csv_histories_fail(self) -> None:
        original_rows = copy.deepcopy(self.rows)
        original_fields = self.fieldnames

        self.rows = [self.rows[0], self.rows[2]]
        self._write_artifacts()
        self.assertIn("csv.incomplete_epoch_history", self._audit().reasons)

        self.rows = copy.deepcopy(original_rows)
        self.rows[1]["epoch"] = 1
        self._write_artifacts()
        self.assertIn(
            "csv.non_contiguous_or_duplicate_epochs", self._audit().reasons
        )

        self.rows = copy.deepcopy(original_rows)
        missing = "test_corrected_trajectory_relative_l2"
        self.fieldnames = tuple(name for name in original_fields if name != missing)
        for row in self.rows:
            row.pop(missing)
        self._write_artifacts()
        audit = self._audit()
        self.assertIn(f"csv.missing_column:{missing}", audit.reasons)

    def test_header_plus_last_epoch_is_not_complete(self) -> None:
        self.rows = [self.rows[-1]]
        self._write_artifacts()
        self.assertFalse(self._audit().complete)

    def test_partial_artifact_set_fails_with_explicit_reason(self) -> None:
        (self.run_dir / "final_checkpoint.pt").unlink()
        audit = self._audit()
        self.assertEqual(audit.status, run_paper_sweep.CompletionStatus.INCOMPLETE)
        self.assertIn("artifact.missing:checkpoint", audit.reasons)

    def test_legacy_evaluation_is_distinct_from_incomplete_training(self) -> None:
        legacy = self.spec["legacy_evaluation_protocol_versions"][0]
        self.summary["evaluation_protocol_version"] = legacy
        self.checkpoint["evaluation_protocol_version"] = legacy
        self._write_artifacts()
        audit = self._audit()
        self.assertEqual(
            audit.status, run_paper_sweep.CompletionStatus.LEGACY_EVALUATION
        )
        self.assertTrue(audit.training_valid)
        self.assertFalse(audit.evaluation_valid)
        self.assertFalse(audit.complete)

    @patch.object(
        run_paper_sweep,
        "historical_source_compatibility_evidence",
        return_value={
            "verified_change_scope": (
                "checkpoint_storage_loading_validation_only"
            ),
            "source_diff_sha256": "d" * 64,
            "changed_paths": ["experiments/common/checkpoints.py"],
        },
    )
    def test_provenance_preserving_historical_reevaluation_is_allowed(
        self,
        _compatibility_evidence,
    ) -> None:
        legacy = self.spec["legacy_evaluation_protocol_versions"][0]
        self.summary["evaluation_protocol_version"] = legacy
        self.checkpoint["evaluation_protocol_version"] = legacy
        self._write_artifacts()

        current_evaluation_commit = "b" * 40
        with self.assertRaises(run_paper_sweep.TrainingArtifactValidationError):
            run_paper_sweep.validate_training_artifacts(
                self.run_dir,
                spec=self.spec,
                current_source_commit=current_evaluation_commit,
                allowed_historical_training_source_commit=self.training_commit,
                allowed_historical_checkpoint_sha256="0" * 64,
            )
        validated = run_paper_sweep.validate_training_artifacts(
            self.run_dir,
            spec=self.spec,
            current_source_commit=current_evaluation_commit,
            allowed_historical_training_source_commit=self.training_commit,
            allowed_historical_checkpoint_sha256=self.summary[
                "checkpoint_sha256"
            ],
        )
        self.assertEqual(
            validated.validation_status,
            "valid_training_legacy_evaluation",
        )

        record = {
            "record_type": "checkpoint_evaluation",
            "schema_version": 2,
            "evaluation_schema_version": run_paper_sweep.EVALUATION_SCHEMA_VERSION,
            "evaluation_protocol_version": self.spec["evaluation_protocol_version"],
            "evaluation_source_commit": current_evaluation_commit,
            "checkpoint_sha256": self.summary["checkpoint_sha256"],
            "checkpoint_load_metadata": dict(
                validated.checkpoint_load_metadata
            ),
            "dataset_id": self.spec["dataset_id"],
            "experiment": self.spec["experiment"],
            "model": self.model,
            "seed": self.seed,
            "training_protocol_version": self.spec["training_protocol_version"],
            "training_protocol_identifier": (
                run_paper_sweep.training_protocol_identifier(self.spec)
            ),
            "training_source_commit": self.training_commit,
            "training_source_dirty": False,
            "training_source_reuse": dict(validated.training_source_reuse),
            "training_artifact_validation_status": (
                "valid_training_legacy_evaluation"
            ),
            "training_artifact_evaluation_classification": "legacy",
            "original_training_evaluation_protocol_version": legacy,
            "training_log_identity": dict(validated.training_log_identity),
            "diagnostic_only": False,
            "paper_eligible": True,
            "diagnostic_reasons": [],
            "training_protocol": copy.deepcopy(
                self.spec["training_configuration"]
            ),
            "training_protocol_sha256": run_paper_sweep.canonical_metadata_sha256(
                self.spec["training_configuration"]
            ),
            "training_configuration": copy.deepcopy(
                self.spec["training_configuration"]
            ),
            "data_configuration": copy.deepcopy(self.spec["data_configuration"]),
            "model_configuration": copy.deepcopy(TINY_MODEL_CONFIGURATION),
            "model_configuration_sha256": (
                run_paper_sweep.canonical_metadata_sha256(
                    TINY_MODEL_CONFIGURATION
                )
            ),
            "dataset_identity": copy.deepcopy(self.dataset_identity),
            "dataset_identity_sha256": run_paper_sweep.canonical_metadata_sha256(
                self.dataset_identity
            ),
            "fixed_final_epoch": self.spec["epochs"],
            "checkpoint_selection_policy": self.spec["training_configuration"][
                "checkpoint_selection_policy"
            ],
            "factorization": "dense",
            "rank": None,
            "parameter_count": self.summary["parameter_count"],
            "evaluation_source_dirty": False,
            "evaluation_space": self.spec["reevaluation_space"],
            **{
                key: copy.deepcopy(value)
                for key, value in self.spec["evaluation_metadata"].items()
                if key != "evaluation_sample_count"
            },
            "evaluation_sample_count": {
                "test": self.spec["evaluation_metadata"][
                    "evaluation_sample_count"
                ]["test"]
            },
            **{
                metric: self.metrics[metric]
                for metric in self.spec["paper_metric_columns"]
            },
            "metrics": {
                "corrected_step_relative_l2": self.metrics[
                    "final_test_corrected_step_relative_l2"
                ],
                "corrected_trajectory_relative_l2": self.metrics[
                    "final_test_corrected_trajectory_relative_l2"
                ],
            },
        }
        evaluation_path = self.run_dir / "reevaluation.json"
        evaluation_path.write_text(
            json.dumps(record, allow_nan=False), encoding="utf-8"
        )
        audit = run_paper_sweep.audit_run_completion(
            self.run_dir,
            spec=self.spec,
            model=self.model,
            seed=self.seed,
            current_source_commit=current_evaluation_commit,
            evaluation_record_path=evaluation_path,
            allowed_historical_training_source_commit=self.training_commit,
            allowed_historical_checkpoint_sha256=self.summary[
                "checkpoint_sha256"
            ],
        )
        self.assertEqual(
            audit.status,
            run_paper_sweep.CompletionStatus.COMPLETE_REEVALUATED,
            msg=audit.reasons,
        )
        self.assertEqual(record["training_source_commit"], self.training_commit)
        self.assertNotEqual(
            record["training_source_commit"], record["evaluation_source_commit"]
        )

    def test_legacy_metadata_cannot_impersonate_current_training_source(self) -> None:
        current = "b" * 40
        self.summary["training_source_commit"] = current
        self.checkpoint["training_source_commit"] = current
        self.checkpoint["environment"]["training_source_commit"] = current
        # Independently recorded source aliases retain the real old source.
        self._write_artifacts()
        audit = run_paper_sweep.audit_run_completion(
            self.run_dir,
            spec=self.spec,
            model=self.model,
            seed=self.seed,
            current_source_commit=current,
        )
        self.assertEqual(audit.status, run_paper_sweep.CompletionStatus.INCOMPLETE)
        self.assertTrue(
            any(
                reason.endswith("source_repository_commit_disagreement")
                for reason in audit.reasons
            )
        )


class ActualProjectArtifactIntegrationTests(unittest.TestCase):
    """Connect a short real-model run to the production artifact readers.

    The one-epoch contract exists only inside this test.  Once that injected
    contract is removed, the resulting evaluation record must be rejected by
    the real 500-epoch paper aggregator.
    """

    def test_short_burgers_run_roundtrips_but_is_not_paper_eligible(self) -> None:
        from experiments import train_burgers
        from scripts import summarize_results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root = root / "runs"
            evaluation_root = root / "evaluations"
            run_dir = results_root / "cp_cafe_plus_fno" / "seed_0"
            source_state = {
                "source_repository_commit": "d" * 40,
                "source_repository_dirty": False,
            }

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(314159)
                train_values = torch.randn(2, 20, 32)
                test_values = torch.randn(2, 20, 32)
            with patch.object(train_burgers, "BATCH_SIZE", 2):
                loaders = train_burgers.build_data_loaders(
                    train_values,
                    test_values,
                    0.0,
                    1.0,
                )
            data = train_burgers.BurgersData(
                train_loader=loaders[0],
                train_eval_loader=loaders[1],
                test_eval_loader=loaders[2],
                mean=0.0,
                std=1.0,
                total_trajectories=4,
                time_steps=20,
                resolution=32,
                reduced_filename="synthetic-fixture.hdf5",
            )

            manifests = copy.deepcopy(dataset_manifest())
            manifest = manifests["burgers1d"]
            manifest["n_train"] = 2
            manifest["n_test"] = 2
            identity = {
                "dataset_id": "burgers1d",
                "source": manifest["source"],
                "resolution": manifest["resolution"],
                "n_train": 2,
                "n_test": 2,
                "files": {
                    filename: {
                        "path": f"data/{filename}",
                        "sha256": metadata["sha256"],
                        "size_bytes": metadata["size_bytes"],
                    }
                    for filename, metadata in manifest["files"].items()
                },
            }

            short_spec = copy.deepcopy(
                run_paper_sweep.DATASETS["burgers1024"]
            )
            short_spec["epochs"] = 1
            short_spec["training_configuration"]["epochs"] = 1
            short_spec["training_configuration"]["batch_size"] = 2
            short_spec["training_configuration"]["scheduler_t_max"] = 1
            short_spec["data_configuration"]["n_train"] = 2
            short_spec["data_configuration"]["n_test"] = 2
            short_spec["evaluation_metadata"]["evaluation_sample_count"] = {
                "train": 2,
                "test": 2,
            }

            with (
                patch.dict(
                    run_paper_sweep.DATASETS,
                    {"burgers1024": short_spec},
                ),
                patch.object(
                    run_paper_sweep,
                    "dataset_manifest",
                    return_value=manifests,
                ),
                patch.object(train_burgers, "EPOCHS", 1),
                patch.object(train_burgers, "BATCH_SIZE", 2),
                patch.object(train_burgers, "SCHEDULER_T_MAX", 1),
                patch.object(train_burgers, "N_TRAIN", 2),
                patch.object(train_burgers, "N_TEST", 2),
                patch.object(
                    train_burgers,
                    "source_repository_state",
                    return_value=source_state,
                ),
                patch.object(
                    train_burgers,
                    "verify_burgers_dataset",
                    return_value=identity,
                ),
                patch.object(
                    train_burgers,
                    "load_burgers",
                    return_value=data,
                ),
            ):
                train_burgers.main(
                    [
                        "--model",
                        "cp_cafe_plus_fno",
                        "--seed",
                        "0",
                        "--data-root",
                        str(root / "data"),
                        "--results-root",
                        str(results_root),
                        "--device",
                        "cpu",
                    ]
                )

                checkpoint_path = run_dir / "final_checkpoint.pt"
                summary_path = run_dir / "summary.json"
                csv_path = run_dir / "training_log.csv"
                self.assertTrue(checkpoint_path.is_file())
                self.assertTrue(summary_path.is_file())
                self.assertTrue(csv_path.is_file())
                direct = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertIs(
                    type(direct["environment"]["pytorch_version"]), str
                )

                validated = run_paper_sweep.validate_training_artifacts(
                    run_dir,
                    spec=short_spec,
                    current_source_commit=source_state[
                        "source_repository_commit"
                    ],
                )
                self.assertEqual(validated.validation_status,
                                 "valid_training_current_evaluation")
                self.assertEqual(validated.training_log_identity["row_count"], 1)
                self.assertEqual(
                    validated.checkpoint_load_metadata[
                        "legacy_torch_version_compatibility_used"
                    ],
                    False,
                )

                original_checkpoint_hash = run_paper_sweep.sha256_file(
                    checkpoint_path
                )
                original_summary = summary_path.read_bytes()
                original_csv = csv_path.read_bytes()
                result_path = summarize_results.reevaluate_checkpoint(
                    dataset="burgers1024",
                    checkpoint_path=checkpoint_path,
                    data_root=root / "data",
                    batch_size=1,
                    device_name="cpu",
                    output_root=evaluation_root,
                    allow_dirty_source=False,
                    diagnostic_allow_dirty_training_artifact=False,
                    allow_legacy_torch_version=False,
                    overwrite=False,
                )
                reevaluation_audit = run_paper_sweep.audit_run_completion(
                    run_dir,
                    spec=short_spec,
                    model="cp_cafe_plus_fno",
                    seed=0,
                    current_source_commit=source_state[
                        "source_repository_commit"
                    ],
                    evaluation_record_path=result_path,
                    allowed_historical_training_source_commit=source_state[
                        "source_repository_commit"
                    ],
                    allowed_historical_checkpoint_sha256=original_checkpoint_hash,
                )
                self.assertEqual(
                    reevaluation_audit.status,
                    run_paper_sweep.CompletionStatus.COMPLETE_REEVALUATED,
                    msg=reevaluation_audit.reasons,
                )

            self.assertEqual(
                run_paper_sweep.sha256_file(checkpoint_path),
                original_checkpoint_hash,
            )
            self.assertEqual(summary_path.read_bytes(), original_summary)
            self.assertEqual(csv_path.read_bytes(), original_csv)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(record["fixed_final_epoch"], 1)
            self.assertEqual(record["evaluation_sample_count"], {"test": 2})
            self.assertEqual(record["training_log_identity"]["row_count"], 1)
            self.assertEqual(
                record["checkpoint_load_metadata"],
                {
                    "checkpoint_load_mode": "weights_only",
                    "legacy_torch_version_compatibility_used": False,
                    "legacy_neuraloperator_metadata_compatibility_used": False,
                },
            )
            with self.assertRaisesRegex(
                summarize_results.ResultValidationError,
                "epoch 500",
            ):
                summarize_results.validate_evaluation_record(record)
            with self.assertRaises(summarize_results.ResultValidationError):
                summarize_results.aggregate_records(
                    [record],
                    required_seeds=(0,),
                    allow_partial=True,
                )


class CafeDarcyRunnerTests(unittest.TestCase):
    def test_default_matrix_is_exactly_four_cafe_models_by_five_seeds(self) -> None:
        runner = run_cafe_darcy_sweep
        self.assertEqual(
            runner.CAFE_MODELS,
            (
                "cafe_plus_fno",
                "cp_cafe_plus_fno",
                "tt_cafe_plus_fno",
                "tucker_cafe_plus_fno",
            ),
        )
        self.assertEqual(PAPER_SEEDS, (0, 42, 73, 108, 202))
        self.assertEqual(len(runner.CAFE_MODELS) * len(PAPER_SEEDS), 20)

    def test_command_rejects_every_baseline(self) -> None:
        runner = run_cafe_darcy_sweep
        command = runner.build_command(
            model="cp_cafe_plus_fno",
            seed=42,
            data_root=Path("data"),
            results_root=runner.DEFAULT_RESULTS_ROOT,
            device="cuda",
            overwrite=True,
        )
        self.assertEqual(command[1:4], ["-B", "-m", "experiments.train_darcy"])
        self.assertIn("cp_cafe_plus_fno", command)
        self.assertIn("--overwrite", command)
        for baseline in (
            "fno",
            "ufno",
            "tfno_cp",
            "amfno",
            "sirenfno",
            "cpsirenfno",
            "ttsirenfno",
            "tuckersirenfno",
        ):
            with self.subTest(baseline=baseline):
                with self.assertRaises(ValueError):
                    runner.build_command(
                        model=baseline,
                        seed=42,
                        data_root=Path("data"),
                        results_root=runner.DEFAULT_RESULTS_ROOT,
                        device="cpu",
                        overwrite=False,
                    )

    def test_main_dispatches_only_the_twenty_expected_subprocesses(self) -> None:
        runner = run_cafe_darcy_sweep
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                runner, "resolve_current_source_commit", return_value="a" * 40
            ),
            patch.object(runner, "is_complete", side_effect=[False, True] * 20),
            patch.object(
                runner.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as mocked_run,
        ):
            runner.main(
                [
                    "--data-root",
                    "data",
                    "--results-root",
                    directory,
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(mocked_run.call_count, 20)
        actual_pairs = []
        for call in mocked_run.call_args_list:
            command = call.args[0]
            actual_pairs.append(
                (
                    command[command.index("--model") + 1],
                    int(command[command.index("--seed") + 1]),
                )
            )
        self.assertEqual(
            actual_pairs,
            [(model, seed) for model in runner.CAFE_MODELS for seed in PAPER_SEEDS],
        )

    def test_incomplete_run_is_fail_closed_without_overwrite(self) -> None:
        runner = run_cafe_darcy_sweep
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "cafe_plus_fno" / "seed_0"
            run_dir.mkdir(parents=True)
            (run_dir / "partial.txt").write_text("incomplete", encoding="utf-8")
            with (
                patch.object(
                    runner, "resolve_current_source_commit", return_value="a" * 40
                ),
                patch.object(runner, "is_complete", return_value=False),
                patch.object(runner.subprocess, "run") as mocked_run,
            ):
                with self.assertRaises(SystemExit):
                    runner.main(["--results-root", directory])
                mocked_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
