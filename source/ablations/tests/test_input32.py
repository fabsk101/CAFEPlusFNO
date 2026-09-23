"""Dense-only, direct-32 feature condition; never changes the old ablations."""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from ablations.contracts import (
    ABLATION_CONDITIONS, DATASETS, INPUT32_CONDITION,
    conditions_for_variant, validate_condition_variant,
)
from ablations.models import build_model_from_kwargs, describe_model
from ablations.tests.test_models import tiny_config


def input32_config(dim: int, *, full_input: int = 32) -> dict:
    config = tiny_config(dim)
    config.update(rff_basis=10, cheb_basis=16, cafe_branch_dim=full_input)
    return config


class DirectInput32Tests(unittest.TestCase):
    def test_supported_condition_matrix_keeps_old_five_for_factors(self):
        old = ("full", "learnable_sigma", "fourier_only", "chebyshev_only",
               "without_linear_branches")
        self.assertEqual(conditions_for_variant("dense"), (*old, INPUT32_CONDITION))
        for variant in ("cp", "tt", "tucker"):
            self.assertEqual(conditions_for_variant(variant), old)
            with self.assertRaisesRegex(ValueError, "INPUT32_DENSE_ONLY"):
                validate_condition_variant(INPUT32_CONDITION, variant)
        self.assertEqual(len(ABLATION_CONDITIONS), 6)

    def test_exact_selected_features_gaussian_prefix_and_hidden_preservation(self):
        from models import Cafe_Plus_FNO1D, Cafe_Plus_FNO2D
        for dim in (1, 2):
            for full_input in (16, 32):
                with self.subTest(dim=dim, full_input=full_input):
                    config = input32_config(dim, full_input=full_input)
                    saved_config = copy.deepcopy(config)
                    torch.manual_seed(42)
                    full = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition="full")
                    full_rng = torch.get_rng_state().clone()
                    torch.manual_seed(42)
                    built = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition=INPUT32_CONDITION)
                    self.assertEqual(config, saved_config)
                    if full_input == 32:
                        self.assertTrue(torch.equal(full_rng, torch.get_rng_state()))
                    # Native get_config exposes active family counts, while
                    # ablation restoration uses base_configuration + condition.
                    expected_native = dict(full.model.get_config())
                    expected_native.update(rff_basis=8, cheb_basis=16 // dim)
                    self.assertEqual(built.model.get_config(), expected_native)
                    info = describe_model(built.model)
                    self.assertEqual(info["embedding_dim"], 32)
                    self.assertEqual(info["kernel_mlp_input_dim"], 32)
                    self.assertEqual(info["rff_feature_dim_per_encoder"], 16)
                    self.assertEqual(info["chebyshev_feature_dim_per_encoder"], 16)
                    self.assertEqual(info["linear_branch_parameters"], 0)
                    self.assertFalse(info["hadamard_combination"])
                    helpers = Cafe_Plus_FNO1D if dim == 1 else Cafe_Plus_FNO2D
                    for full_layer, layer in zip(full.model.operator_layers, built.model.operator_layers):
                        fg = full_layer.spectral.kernel_generator
                        g = layer.spectral.kernel_generator
                        enc = g.encoder
                        self.assertEqual(tuple(enc.G.shape), (dim, 8))
                        self.assertEqual((enc.rff_basis, enc.cheb_basis), (8, 16 // dim))
                        self.assertEqual((enc.source_rff_basis, enc.source_cheb_basis), (10, 16))
                        self.assertEqual(len(enc.branches), 0)
                        self.assertEqual(g.head.joint_mlp.linear_in.in_features, 32)
                        self.assertEqual(g.head.joint_mlp.linear_in.out_features,
                                         fg.head.joint_mlp.linear_in.out_features)
                        torch.testing.assert_close(enc.G, fg.encoder.G[:, :8], rtol=0, atol=0)
                        for key, value in fg.head.joint_mlp.state_dict().items():
                            if full_input == 32 or not key.startswith("linear_in."):
                                torch.testing.assert_close(value, g.head.joint_mlp.state_dict()[key], rtol=0, atol=0)
                        coords = torch.linspace(0, 1, 6 * dim).reshape(6, dim)
                        phase = fg.encoder.phase_scale * (coords @ (fg.encoder.sigma * fg.encoder.G[:, :8]))
                        cheb = torch.cat([helpers.chebyshev_embedding(coords[:, axis], 16 // dim)
                                          for axis in range(dim)], dim=-1)
                        expected = torch.cat((cheb, torch.cos(phase), torch.sin(phase)), dim=-1)
                        torch.testing.assert_close(enc.forward_coordinates(coords), expected, rtol=0, atol=0)
                    spec = built.configuration["direct_input32_basis"]
                    self.assertEqual(spec["version"], "direct_input32_v1")
                    self.assertEqual(spec["kernel_mlp_input_dim"], 32)
                    self.assertNotIn("direct_input32_basis", full.configuration)

    def test_only_expected_parameters_change_and_fixed_gaussian_is_not_optimized(self):
        for dim in (1, 2):
            config = input32_config(dim)
            torch.manual_seed(73)
            full = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition="full").model
            torch.manual_seed(73)
            new = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition=INPUT32_CONDITION).model
            full_state = full.state_dict()
            for name, value in new.state_dict().items():
                if ".encoder.G" not in name:
                    torch.testing.assert_close(value, full_state[name], rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in new.parameters()),
                             describe_model(full)["linear_branch_parameters"])
            self.assertFalse(any(".branches." in n or n.endswith(".G") or n.endswith("log_sigma")
                                 for n, _ in new.named_parameters()))
            g_before = {n: b.clone() for n, b in new.named_buffers() if n.endswith(".G")}
            x = torch.randn((2, 1, 9) if dim == 1 else (2, 1, 5, 6))
            output = new(x)
            self.assertEqual(output.shape, x.shape)
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()))
            torch.optim.AdamW(new.parameters(), lr=1e-3).step()
            for n, value in g_before.items():
                torch.testing.assert_close(value, dict(new.named_buffers())[n], rtol=0, atol=0)

    def test_original_branch_free_condition_retains_original_embedding(self):
        for dim in (1, 2):
            config = input32_config(dim)
            old = build_model_from_kwargs(spatial_dim=dim, configuration=config,
                                          condition="without_linear_branches")
            expected = 2 * config["rff_basis"] + dim * config["cheb_basis"]
            self.assertEqual(describe_model(old.model)["kernel_mlp_input_dim"], expected)
            self.assertNotIn("direct_input32_basis", old.configuration)

    def test_insufficient_source_basis_and_factorized_requests_are_rejected(self):
        for dim in (1, 2):
            for field, too_small in (("rff_basis", 7), ("cheb_basis", 16 // dim - 1)):
                config = input32_config(dim)
                config[field] = too_small
                with self.assertRaisesRegex(ValueError, "INPUT32_SOURCE_BASIS_TOO_SMALL"):
                    build_model_from_kwargs(spatial_dim=dim, configuration=config, condition=INPUT32_CONDITION)
            for variant in ("cp", "tt", "tucker"):
                config = input32_config(dim)
                config["factorization"] = variant
                state = torch.get_rng_state().clone()
                with self.assertRaisesRegex(ValueError, "INPUT32_DENSE_ONLY"):
                    build_model_from_kwargs(spatial_dim=dim, configuration=config, condition=INPUT32_CONDITION)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_all_seven_current_configs_build_input32_without_parent_config_changes(self):
        expected = {
            "darcy": (32, 32, 312577), "ns2d": (32, 64, 578945),
            "burgers1d": (32, 32, 312833), "airfoil": (32, 32, 312609),
            "reacdiff1d": (16, 32, 312833), "cfd1d": (16, 32, 312833),
            "cfd2d": (16, 32, 312705),
        }
        for dataset, spec in DATASETS.items():
            with self.subTest(dataset=dataset):
                module = importlib.import_module(spec.config_module)
                config = module.model_constructor_kwargs("cafe_plus_fno")
                original = copy.deepcopy(config)
                full_input, hidden, count = expected[dataset]
                self.assertEqual(config["cafe_branch_dim"], full_input)
                self.assertEqual(config["kernel_hidden_dim"], hidden)
                built = build_model_from_kwargs(spatial_dim=spec.spatial_dim, configuration=config,
                                                condition=INPUT32_CONDITION)
                description = describe_model(built.model)
                self.assertEqual((description["embedding_dim"], description["kernel_mlp_input_dim"]), (32, 32))
                self.assertEqual(description["parameter_count"], count)
                for layer in built.model.operator_layers:
                    mlp = layer.spectral.kernel_generator.head.joint_mlp
                    self.assertEqual((mlp.linear_in.in_features, mlp.linear_in.out_features), (32, hidden))
                    self.assertEqual(mlp.linear_out.out_features, 2 * config["width"] ** 2)
                self.assertEqual(config, original)
                self.assertEqual(module.model_constructor_kwargs("cafe_plus_fno"), original)

    def test_cli_rejects_factorized_request_before_io(self):
        from ablations.run import parse_args as parse_run
        from ablations.train import parse_args as parse_train, run_directory
        from ablations.smoke import parse_args as parse_smoke
        from ablations.check import main as check_main
        for variant in ("cp", "tt", "tucker"):
            args = ["--dataset", "darcy", "--model-variant", variant, "--condition", INPUT32_CONDITION]
            cases = (
                (parse_run, args + ["--data-root", "data", "--dry-run"]),
                (parse_train, args + ["--data-root", "data", "--seed", "0"]),
                (parse_smoke, args + ["--kind", "cuda", "--device", "cuda", "--seed", "0", "--report", "never.json"]),
                (check_main, args),
            )
            for entry, argv in cases:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    entry(argv)
                self.assertEqual(caught.exception.code, 2)
            with self.assertRaisesRegex(ValueError, "INPUT32_DENSE_ONLY"):
                run_directory(Path("never_created"), dataset="darcy", model_variant=variant,
                              condition=INPUT32_CONDITION, seed=0)

    def test_checkpoint_condition_and_resolved_basis_tampering_rejected(self):
        from experiments.common.checkpoints import prepare_weights_only_checkpoint
        from ablations.checkpoints import restore_model_checkpoint
        config = input32_config(1)
        built = build_model_from_kwargs(spatial_dim=1, configuration=config, condition=INPUT32_CONDITION)
        payload = prepare_weights_only_checkpoint({
            "base_configuration": config, "model_configuration": built.configuration,
            "ablation_variant": INPUT32_CONDITION, "condition": INPUT32_CONDITION,
            "dataset": "burgers1d", "model_variant": "dense", "seed": 0,
            "model_state_dict": built.model.state_dict(),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_input32.pt"
            torch.save(payload, path)
            restore_model_checkpoint(path, expected_dataset="burgers1d", expected_model_variant="dense",
                                     expected_condition=INPUT32_CONDITION, expected_seed=0)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                restore_model_checkpoint(path, expected_dataset="burgers1d", expected_model_variant="dense",
                                         expected_condition="without_linear_branches", expected_seed=0)
            payload["model_configuration"]["direct_input32_basis"]["rff_basis"] = 7
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                restore_model_checkpoint(path, expected_dataset="burgers1d", expected_model_variant="dense",
                                         expected_condition=INPUT32_CONDITION, expected_seed=0)
