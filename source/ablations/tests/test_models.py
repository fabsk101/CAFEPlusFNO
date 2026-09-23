from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from ablations.contracts import DEFAULT_SEEDS

from ablations.models import (
    build_model_from_kwargs,
    count_parameters,
    describe_model,
)


def tiny_config(spatial_dim: int, *, factorization: str = "dense") -> dict:
    config = {
        "width": 4,
        "input_dim": 1,
        "output_dim": 1,
        "padding": 0,
        "mlp_dropout": 0.0,
        "add_grid": True,
        "num_layers": 1,
        "ffn_expansion": 2,
        "rff_basis": 3,
        "cheb_basis": 2,
        "cafe_branches": 2,
        "cafe_branch_dim": None,
        "kernel_hidden_dim": 5,
        "kernel_mlp_type": "joint",
        "kernel_activation": "gelu",
        "kernel_output_init_std": 1e-3,
        "sigma_init": 1.0,
        "learnable_sigma": False,
        "enforce_hermitian": True,
        "input_layout": "channels_first",
        "output_layout": "channels_first",
        "factorization": factorization,
        "rank": 2,
    }
    if factorization != "dense":
        config.update(
            {
                "factor_rff_basis": 2,
                "factor_cheb_basis": 2,
                "factor_branch_dim": 3,
                "factor_hidden_dim": 4,
                "factor_output_init_mode": "xavier",
                "factor_kernel_target_rms": 1e-3,
            }
        )
    return config


class FullParityTests(unittest.TestCase):
    def _run_parity(self, spatial_dim: int, factorization: str = "dense", seed: int = 314159) -> None:
        from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D
        from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D

        cls = CAFEPlusFNO1D if spatial_dim == 1 else CAFEPlusFNO2D
        config = tiny_config(spatial_dim, factorization=factorization)
        torch.manual_seed(seed)
        reference = cls(**config)
        reference_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(seed)
        candidate = build_model_from_kwargs(
            spatial_dim=spatial_dim,
            configuration=config,
            condition="full",
        ).model
        candidate_rng = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(reference_rng, candidate_rng), "RNG consumption")

        reference_state = reference.state_dict()
        candidate_state = candidate.state_dict()
        self.assertEqual(reference_state.keys(), candidate_state.keys())
        for name in reference_state:
            torch.testing.assert_close(
                reference_state[name], candidate_state[name], rtol=0.0, atol=0.0,
                msg=lambda message, n=name: f"{n}: {message}",
            )

        shape = (2, 1, 9) if spatial_dim == 1 else (2, 1, 5, 6)
        torch.manual_seed(2718)
        x = torch.randn(shape)
        target = torch.randn(shape)
        y_reference = reference(x)
        y_candidate = candidate(x)
        torch.testing.assert_close(y_reference, y_candidate, rtol=0.0, atol=0.0)
        loss_reference = torch.nn.functional.mse_loss(y_reference, target)
        loss_candidate = torch.nn.functional.mse_loss(y_candidate, target)
        torch.testing.assert_close(loss_reference, loss_candidate, rtol=0.0, atol=0.0)
        loss_reference.backward()
        loss_candidate.backward()
        reference_grads = dict(reference.named_parameters())
        candidate_grads = dict(candidate.named_parameters())
        self.assertEqual(reference_grads.keys(), candidate_grads.keys())
        for name in reference_grads:
            torch.testing.assert_close(
                reference_grads[name].grad,
                candidate_grads[name].grad,
                rtol=0.0,
                atol=0.0,
            )
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3)
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=2e-3)
        reference_optimizer.step()
        candidate_optimizer.step()
        for (name_a, value_a), (name_b, value_b) in zip(
            reference.state_dict().items(), candidate.state_dict().items()
        ):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(value_a, value_b, rtol=0.0, atol=0.0)

    def test_full_1d_exact_parity(self) -> None:
        self._run_parity(1)

    def test_full_2d_exact_parity(self) -> None:
        self._run_parity(2)

    def test_full_parity_all_factorizations_and_default_seeds(self):
        for dimension in (1, 2):
            for factorization in ('dense', 'cp', 'tt', 'tucker'):
                for seed in DEFAULT_SEEDS:
                    with self.subTest(dimension=dimension, factorization=factorization, seed=seed):
                        self._run_parity(dimension, factorization, seed)


class AblationBehaviorTests(unittest.TestCase):
    def test_each_condition_forward_backward_and_checkpoint_restore(self) -> None:
        from experiments.common.checkpoints import prepare_weights_only_checkpoint

        for condition in (
            "full",
            "learnable_sigma",
            "fourier_only",
            "chebyshev_only",
            "without_linear_branches",
        ):
            with self.subTest(condition=condition):
                torch.manual_seed(123)
                config = tiny_config(1)
                built = build_model_from_kwargs(
                    spatial_dim=1, configuration=config, condition=condition
                )
                model = built.model
                x = torch.randn(2, 1, 9)
                loss = model(x).square().mean()
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
                optimizer.step()
                scheduler.step()
                checkpoint = prepare_weights_only_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "model_configuration": built.configuration,
                        "epoch": 1,
                        "result_kind": "diagnostic",
                    }
                )
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "diagnostic_checkpoint.pt"
                    torch.save(checkpoint, path)
                    loaded = torch.load(path, map_location="cpu", weights_only=True)
                torch.manual_seed(456)
                restored = build_model_from_kwargs(
                    spatial_dim=1, configuration=config, condition=condition
                ).model
                restored.load_state_dict(loaded["model_state_dict"], strict=True)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x), restored(x), rtol=0.0, atol=0.0)

    def test_dimensions_parameters_and_nonzero_branchless_kernel(self) -> None:
        config = tiny_config(1)
        models = {
            condition: build_model_from_kwargs(
                spatial_dim=1, configuration=config, condition=condition
            ).model
            for condition in (
                "full",
                "learnable_sigma",
                "fourier_only",
                "chebyshev_only",
                "without_linear_branches",
            )
        }
        descriptions = {name: describe_model(model) for name, model in models.items()}
        self.assertEqual(descriptions["full"]["embedding_dim"], 8)
        self.assertEqual(descriptions["fourier_only"]["embedding_dim"], 6)
        self.assertEqual(descriptions["chebyshev_only"]["embedding_dim"], 2)
        self.assertEqual(descriptions["without_linear_branches"]["embedding_dim"], 8)
        self.assertEqual(descriptions["without_linear_branches"]["linear_branch_parameters"], 0)
        self.assertGreater(descriptions["full"]["linear_branch_parameters"], 0)
        self.assertEqual(
            count_parameters(models["learnable_sigma"]),
            count_parameters(models["full"]) + descriptions["full"]["encoder_count"],
        )
        self.assertLess(count_parameters(models["fourier_only"]), count_parameters(models["full"]))
        self.assertLess(count_parameters(models["chebyshev_only"]), count_parameters(models["full"]))
        self.assertLess(
            count_parameters(models["without_linear_branches"]),
            count_parameters(models["full"]),
        )
        for description in descriptions.values():
            self.assertFalse(description["gaussian_basis_trainable"])
        generator = models["without_linear_branches"].operator_layers[0].spectral.kernel_generator
        with torch.no_grad():
            kernel = generator((9,))
        self.assertGreater(float(kernel.abs().sum()), 0.0)

    def test_factorized_2d_branchless_path(self) -> None:
        config = tiny_config(2, factorization="cp")
        model = build_model_from_kwargs(
            spatial_dim=2,
            configuration=config,
            condition="without_linear_branches",
        ).model
        x = torch.randn(2, 1, 5, 6)
        output = model(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        output.square().mean().backward()
        description = describe_model(model)
        self.assertEqual(description["linear_branch_parameters"], 0)
        self.assertEqual(description["embedding_dim"], 6)

    def test_all_factorizations_and_conditions_forward_backward(self) -> None:
        for spatial_dim in (1, 2):
            shape = (1, 1, 7) if spatial_dim == 1 else (1, 1, 4, 5)
            for factorization in ("cp", "tt", "tucker"):
                for condition in (
                    "full",
                    "learnable_sigma",
                    "fourier_only",
                    "chebyshev_only",
                    "without_linear_branches",
                ):
                    with self.subTest(
                        spatial_dim=spatial_dim,
                        factorization=factorization,
                        condition=condition,
                    ):
                        torch.manual_seed(99)
                        model = build_model_from_kwargs(
                            spatial_dim=spatial_dim,
                            configuration=tiny_config(
                                spatial_dim, factorization=factorization
                            ),
                            condition=condition,
                        ).model
                        output = model(torch.randn(shape))
                        self.assertEqual(tuple(output.shape), shape)
                        output.square().mean().backward()
                        self.assertTrue(
                            all(
                                parameter.grad is not None
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            )
                        )


if __name__ == "__main__":
    unittest.main()
