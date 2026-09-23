"""Role-based tests consolidated from 2 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_cafe_global_rng.py
import inspect

import unittest

from pathlib import Path

import torch

from experiments.common.seed import set_seed

from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D, CAFEPlusKernelGenerator1D, CAFEPlusModeEncoder, CAFEPlusScalarFactorBlock, CAFEPlusSpectralConv1D

from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D, CAFEPlusKernelGenerator2D, CAFEPlusModeEncoder2D, CAFEPlusSpectralConv2D

_caferng_FACTORIZATIONS = ('dense', 'cp', 'tt', 'tucker')

_caferng_MODEL_CASES = (('1D', CAFEPlusFNO1D), ('2D', CAFEPlusFNO2D))

_caferng_COMMON = {'width': 4, 'input_dim': 2, 'output_dim': 1, 'num_layers': 2, 'ffn_expansion': 2, 'rff_basis': 5, 'cheb_basis': 4, 'cafe_branches': 2, 'cafe_branch_dim': 6, 'kernel_hidden_dim': 7, 'rank': 2, 'input_layout': 'channels_first', 'output_layout': 'channels_first'}

def _caferng__build(model_class, factorization: str):
    return model_class(factorization=factorization, **_caferng_COMMON)

def _g_buffers(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class GlobalRFFReproducibilityTests(unittest.TestCase):

    def test_encoder_g_is_the_next_global_torch_randn_draw(self) -> None:
        set_seed(42)
        expected_1d = torch.randn(1, 5, dtype=torch.float32)
        set_seed(42)
        encoder_1d = CAFEPlusModeEncoder(rff_basis=5, cheb_basis=4, num_branches=2)
        torch.testing.assert_close(encoder_1d.G, expected_1d, rtol=0, atol=0)
        set_seed(73)
        expected_2d = torch.randn(2, 5, dtype=torch.float32)
        set_seed(73)
        encoder_2d = CAFEPlusModeEncoder2D(spatial_dim=2, rff_basis=5, cheb_basis=4, num_branches=2)
        torch.testing.assert_close(encoder_2d.G, expected_2d, rtol=0, atol=0)

    def test_same_global_seed_reproduces_g_and_entire_state(self) -> None:
        for dimension, model_class in _caferng_MODEL_CASES:
            for factorization in _caferng_FACTORIZATIONS:
                with self.subTest(dimension=dimension, factorization=factorization):
                    set_seed(42)
                    first = _caferng__build(model_class, factorization)
                    first_state = first.state_dict()
                    set_seed(42)
                    second = _caferng__build(model_class, factorization)
                    second_state = second.state_dict()
                    self.assertEqual(tuple(first_state), tuple(second_state))
                    self.assertTrue(_g_buffers(first))
                    for key in first_state:
                        torch.testing.assert_close(first_state[key], second_state[key], rtol=0, atol=0)

    def test_different_global_seeds_change_at_least_one_g(self) -> None:
        for dimension, model_class in _caferng_MODEL_CASES:
            for factorization in _caferng_FACTORIZATIONS:
                with self.subTest(dimension=dimension, factorization=factorization):
                    set_seed(42)
                    first = _g_buffers(_caferng__build(model_class, factorization))
                    set_seed(73)
                    second = _g_buffers(_caferng__build(model_class, factorization))
                    self.assertEqual(tuple(first), tuple(second))
                    self.assertTrue(any((not torch.equal(first[key], second[key]) for key in first)))

    def test_operator_layers_receive_distinct_global_rng_draws(self) -> None:
        for dimension, model_class in _caferng_MODEL_CASES:
            for factorization in _caferng_FACTORIZATIONS:
                with self.subTest(dimension=dimension, factorization=factorization):
                    set_seed(42)
                    model = _caferng__build(model_class, factorization)
                    first = model.operator_layers[0].spectral.kernel_generator
                    second = model.operator_layers[1].spectral.kernel_generator
                    if factorization == 'dense':
                        first_g = first.encoder.G
                        second_g = second.encoder.G
                    else:
                        first_g = first.factor_blocks['input_real'].encoder.G
                        second_g = second.factor_blocks['input_real'].encoder.G
                    self.assertFalse(torch.equal(first_g, second_g))

    def test_g_is_persistent_fixed_and_sigma_policy_is_unchanged(self) -> None:
        for learnable_sigma in (True, False):
            set_seed(42)
            model = CAFEPlusFNO1D(factorization='cp', learnable_sigma=learnable_sigma, **_caferng_COMMON)
            state = model.state_dict()
            parameters = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            g_names = [name for name in buffers if name.endswith('.G')]
            self.assertTrue(g_names)
            self.assertTrue(all((name in state for name in g_names)))
            self.assertTrue(all((not buffers[name].requires_grad for name in g_names)))
            sigma_names = [name for name in state if name.endswith('.log_sigma')]
            self.assertTrue(sigma_names)
            if learnable_sigma:
                self.assertTrue(all((name in parameters for name in sigma_names)))
            else:
                self.assertTrue(all((name in buffers for name in sigma_names)))

    def test_public_and_internal_initializers_have_no_rff_seed_argument(self) -> None:
        for initializer in (CAFEPlusModeEncoder, CAFEPlusModeEncoder2D, CAFEPlusScalarFactorBlock, CAFEPlusKernelGenerator1D, CAFEPlusKernelGenerator2D, CAFEPlusSpectralConv1D, CAFEPlusSpectralConv2D, CAFEPlusFNO1D, CAFEPlusFNO2D):
            self.assertNotIn('rff_seed', inspect.signature(initializer).parameters)

    def test_no_internal_reseeding_patterns_remain(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        for relative_path in ('models/Cafe_Plus_FNO1D.py', 'models/Cafe_Plus_FNO2D.py'):
            source = (repository_root / relative_path).read_text(encoding='utf-8')
            for forbidden in ('torch.Generator(', 'manual_seed(', 'rff_seed + layer_index', 'rff_seed: ', 'rff_seed='):
                self.assertNotIn(forbidden, source)

    def test_training_entry_point_seeds_before_model_construction(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        source = (repository_root / 'experiments/train_darcy.py').read_text(encoding='utf-8')
        self.assertLess(source.index('set_seed(args.seed)'), source.index('build_model(args.model, device)'))
        self.assertNotIn('--cafe-rff-seed', source)
        self.assertIn('"rff_rng_policy": "global_torch_rng"', source)

    def test_config_and_summary_expose_global_rng_policy(self) -> None:
        for _, model_class in _caferng_MODEL_CASES:
            set_seed(42)
            model = _caferng__build(model_class, 'tucker')
            config = model.get_config()
            summary = model.architecture_summary()
            self.assertNotIn('rff_seed', config)
            self.assertNotIn('rff_rng', config)
            self.assertNotIn('rff_seed_source', config)
            self.assertEqual(summary['rff_rng'], 'torch_global')
            self.assertEqual(summary['rff_seed_source'], 'global_experiment_seed')

# Migrated from tests/test_cafe_sigma_api.py


from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D

from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D

_cafesigma_FACTORIZATIONS = ('dense', 'cp', 'tt', 'tucker')

_cafesigma_MODEL_CASES = (('1D', CAFEPlusFNO1D), ('2D', CAFEPlusFNO2D))

ROLES = {'1D': ('input_real', 'input_imag', 'output_real', 'output_imag', 'frequency_real', 'frequency_imag'), '2D': ('input_real', 'input_imag', 'output_real', 'output_imag', 'kx_real', 'kx_imag', 'ky_real', 'ky_imag')}

_cafesigma_COMMON = {'width': 4, 'input_dim': 2, 'output_dim': 1, 'num_layers': 2, 'ffn_expansion': 2, 'rff_basis': 5, 'cheb_basis': 4, 'cafe_branches': 2, 'cafe_branch_dim': 6, 'kernel_hidden_dim': 7, 'rank': 2, 'input_layout': 'channels_first', 'output_layout': 'channels_first'}

def _cafesigma__build(model_class, factorization: str, **overrides):
    arguments = {**_cafesigma_COMMON, **overrides, 'factorization': factorization}
    torch.manual_seed(42)
    return model_class(**arguments)

def _generators(model):
    return [layer.spectral.kernel_generator for layer in model.operator_layers]

class SigmaDiagnosticTests(unittest.TestCase):

    def test_regularization_is_finite_scalar_for_all_eight_variants(self) -> None:
        for dimension, model_class in _cafesigma_MODEL_CASES:
            for factorization in _cafesigma_FACTORIZATIONS:
                with self.subTest(dimension=dimension, factorization=factorization):
                    model = _cafesigma__build(model_class, factorization, sigma_init=1.75)
                    regularization = model.sigma_regularization()
                    self.assertEqual(regularization.ndim, 0)
                    self.assertTrue(torch.isfinite(regularization).item())
                    expected = torch.stack([encoder.log_sigma.square() for encoders in model._sigma_encoders_by_role().values() for encoder in encoders]).mean()
                    torch.testing.assert_close(regularization, expected)

    def test_dense_regularization_preserves_original_numerical_definition(self) -> None:
        for dimension, model_class in _cafesigma_MODEL_CASES:
            with self.subTest(dimension=dimension):
                model = _cafesigma__build(model_class, 'dense', sigma_init=1.75)
                original = torch.stack([layer.spectral.kernel_generator.encoder.log_sigma.square() for layer in model.operator_layers]).mean()
                torch.testing.assert_close(model.sigma_regularization(), original)

    def test_factorized_regularization_reaches_every_learnable_sigma(self) -> None:
        for dimension, model_class in _cafesigma_MODEL_CASES:
            for factorization in _cafesigma_FACTORIZATIONS[1:]:
                with self.subTest(dimension=dimension, factorization=factorization):
                    model = _cafesigma__build(model_class, factorization, sigma_init=2.0, learnable_sigma=True)
                    model.sigma_regularization().backward()
                    for generator in _generators(model):
                        self.assertEqual(tuple(generator.factor_blocks.keys()), ROLES[dimension])
                        for block in generator.factor_blocks.values():
                            gradient = block.encoder.log_sigma.grad
                            self.assertIsNotNone(gradient)
                            self.assertTrue(torch.isfinite(gradient).item())
                            self.assertNotEqual(float(gradient.item()), 0.0)

    def test_fixed_sigma_is_safe_and_remains_non_trainable(self) -> None:
        cases = ((CAFEPlusFNO1D, 'dense'), (CAFEPlusFNO2D, 'cp'))
        for model_class, factorization in cases:
            with self.subTest(model=model_class.__name__, factorization=factorization):
                model = _cafesigma__build(model_class, factorization, sigma_init=1.75, learnable_sigma=False)
                regularization = model.sigma_regularization()
                self.assertEqual(regularization.ndim, 0)
                self.assertTrue(torch.isfinite(regularization).item())
                for encoders in model._sigma_encoders_by_role().values():
                    for encoder in encoders:
                        self.assertFalse(encoder.log_sigma.requires_grad)
                        self.assertIsNone(encoder.log_sigma.grad)

    def test_sigma_values_dense_shape_and_detach_semantics(self) -> None:
        for dimension, model_class in _cafesigma_MODEL_CASES:
            with self.subTest(dimension=dimension):
                model = _cafesigma__build(model_class, 'dense', learnable_sigma=True)
                detached = model.sigma_values(detach=True)
                attached = model.sigma_values(detach=False)
                self.assertIsInstance(detached, torch.Tensor)
                self.assertEqual(tuple(detached.shape), (_cafesigma_COMMON['num_layers'],))
                self.assertFalse(detached.requires_grad)
                self.assertTrue(attached.requires_grad)
                self.assertEqual(detached.device, attached.device)
                torch.testing.assert_close(detached, attached.detach())

    def test_factorized_sigma_values_cover_every_role_and_layer(self) -> None:
        for dimension, model_class in _cafesigma_MODEL_CASES:
            for factorization in _cafesigma_FACTORIZATIONS[1:]:
                with self.subTest(dimension=dimension, factorization=factorization):
                    model = _cafesigma__build(model_class, factorization, learnable_sigma=True)
                    detached = model.sigma_values(detach=True)
                    attached = model.sigma_values(detach=False)
                    self.assertIsInstance(detached, dict)
                    self.assertEqual(tuple(detached.keys()), ROLES[dimension])
                    self.assertEqual(tuple(attached.keys()), ROLES[dimension])
                    for role in ROLES[dimension]:
                        self.assertEqual(tuple(detached[role].shape), (_cafesigma_COMMON['num_layers'],))
                        self.assertFalse(detached[role].requires_grad)
                        self.assertTrue(attached[role].requires_grad)
                        self.assertEqual(detached[role].device, attached[role].device)
                        expected = torch.stack([generator.factor_blocks[role].sigma for generator in _generators(model)])
                        torch.testing.assert_close(attached[role], expected)
                        torch.testing.assert_close(detached[role], expected.detach())
