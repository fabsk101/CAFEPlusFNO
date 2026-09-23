"""Role-based tests consolidated from 6 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_airfoil_models.py
import gc

import unittest

import torch

from experiments import train_airfoil

from experiments.common.seed import set_seed

from experiments.configs.airfoil import CANONICAL_SEED as _afmodel_CANONICAL_SEED, EXPECTED_CAFE_PARAMETER_COUNTS as _afmodel_EXPECTED_CAFE_PARAMETER_COUNTS, EXPECTED_PINNED_PARAMETER_COUNTS as _afmodel_EXPECTED_PINNED_PARAMETER_COUNTS, MODEL_CHOICES as _afmodel_MODEL_CHOICES, RESOLUTION as _afmodel_RESOLUTION, model_constructor_kwargs as _afmodel_model_constructor_kwargs

from neuralop.losses import LpLoss as _afmodel_LpLoss

from neuralop.utils import count_model_params

class AirfoilModelTests(unittest.TestCase):

    def tearDown(self) -> None:
        gc.collect()

    def test_all_eleven_constructors_and_parameter_counts(self) -> None:
        expected = {**_afmodel_EXPECTED_PINNED_PARAMETER_COUNTS, **_afmodel_EXPECTED_CAFE_PARAMETER_COUNTS}
        self.assertEqual(set(expected), set(_afmodel_MODEL_CHOICES))
        for index, model_name in enumerate(_afmodel_MODEL_CHOICES):
            with self.subTest(model=model_name):
                set_seed(_afmodel_CANONICAL_SEED + index)
                built = train_airfoil.build_model(model_name, torch.device('cpu'))
                try:
                    count = count_model_params(built.model)
                    self.assertEqual(count, expected[model_name])
                    train_airfoil.validate_model_parameter_count(model_name, count)
                finally:
                    del built
                    gc.collect()

    def test_all_eleven_models_preserve_full_airfoil_shape(self) -> None:
        sample = torch.randn(1, 2, *_afmodel_RESOLUTION)
        for index, model_name in enumerate(_afmodel_MODEL_CHOICES):
            with self.subTest(model=model_name):
                set_seed(_afmodel_CANONICAL_SEED + index)
                built = train_airfoil.build_model(model_name, torch.device('cpu'))
                try:
                    built.model.eval()
                    with torch.no_grad():
                        prediction = built.model(sample)
                    self.assertEqual(tuple(prediction.shape), (1, 1, *_afmodel_RESOLUTION))
                    self.assertTrue(torch.isfinite(prediction).all().item())
                finally:
                    del built
                    gc.collect()

    def test_all_eleven_models_backward_on_small_synthetic_tensor(self) -> None:
        sample = torch.randn(1, 2, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        loss_function = _afmodel_LpLoss(d=2, p=2, reduction='sum')
        for index, model_name in enumerate(_afmodel_MODEL_CHOICES):
            with self.subTest(model=model_name):
                set_seed(_afmodel_CANONICAL_SEED + index)
                built = train_airfoil.build_model(model_name, torch.device('cpu'))
                try:
                    prediction = built.model(sample)
                    self.assertEqual(tuple(prediction.shape), tuple(target.shape))
                    self.assertTrue(torch.isfinite(prediction).all().item())
                    loss = loss_function(prediction, target)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    gradients = [parameter.grad for parameter in built.model.parameters() if parameter.requires_grad and parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                finally:
                    del built
                    gc.collect()

    def test_resolved_siren_kwargs_are_exact(self) -> None:
        for name in ('sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno'):
            with self.subTest(model=name):
                set_seed(_afmodel_CANONICAL_SEED)
                built = train_airfoil.build_model(name, torch.device('cpu'))
                self.assertEqual(built.configuration, _afmodel_model_constructor_kwargs(name))

    def test_no_airfoil_ufno_constructor_exists(self) -> None:
        with self.assertRaises(KeyError):
            _afmodel_model_constructor_kwargs('ufno')
        with self.assertRaises(KeyError):
            train_airfoil.build_model('ufno', torch.device('cpu'))

# Migrated from tests/test_burgers_models.py
import contextlib


import json


from unittest.mock import patch




from experiments.configs.burgers1d import CAFEPLUSFNO_FACTORIZATIONS as _bgmodel_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _bgmodel_CANONICAL_SEED, EXPECTED_PINNED_PARAMETER_COUNTS as _bgmodel_EXPECTED_PINNED_PARAMETER_COUNTS, MODEL_CHOICES as _bgmodel_MODEL_CHOICES, model_constructor_kwargs as _bgmodel_model_constructor_kwargs

from experiments.train_burgers import build_model as _bgmodel_build_model, validate_baseline_parameter_count as _bgmodel_validate_baseline_parameter_count

from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D

_bgmodel_CAFE_MODELS = tuple(_bgmodel_CAFEPLUSFNO_FACTORIZATIONS)

@contextlib.contextmanager
def _bgmodel__released_amfno_cpu_compatibility(model_name: str):
    """Neutralize the released constructor's unused ``.cuda()`` on CPU CI."""
    if model_name == 'amfno' and (not torch.cuda.is_available()):
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            yield
    else:
        yield

def _bgmodel__g_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class BurgersModelTests(unittest.TestCase):

    def test_all_eight_baseline_parameter_counts(self) -> None:
        device = torch.device('cpu')
        for model_name, expected in _bgmodel_EXPECTED_PINNED_PARAMETER_COUNTS.items():
            with self.subTest(model=model_name), _bgmodel__released_amfno_cpu_compatibility(model_name):
                set_seed(_bgmodel_CANONICAL_SEED)
                built = _bgmodel_build_model(model_name, device)
                actual = count_model_params(built.model)
                self.assertEqual(actual, expected)
                _bgmodel_validate_baseline_parameter_count(model_name, actual)
                del built
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def test_all_twelve_models_forward_backward_on_small_tensor(self) -> None:
        device = torch.device('cpu')
        x = torch.randn(1, 10, 64, device=device)
        for index, model_name in enumerate(_bgmodel_MODEL_CHOICES):
            with self.subTest(model=model_name), _bgmodel__released_amfno_cpu_compatibility(model_name):
                set_seed(_bgmodel_CANONICAL_SEED + index)
                built = _bgmodel_build_model(model_name, device)
                model = built.model
                prediction = model(x)
                self.assertEqual(tuple(prediction.shape), (1, 1, 64))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = prediction.square().mean()
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                if model_name in {'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'}:
                    generator = model.operator_layers[0].spectral.kernel_generator
                    kernel = generator((64,))
                    kernel_rms = kernel.abs().square().mean().sqrt()
                    self.assertAlmostEqual(float(kernel_rms.detach().item()), 0.001, places=8)
                print('BURGERS_SYNTHETIC_SMOKE ' + json.dumps({'model': model_name, 'parameter_count': count_model_params(model), 'factorization': built.factorization, 'rank': built.rank, 'shape': list(prediction.shape)}, sort_keys=True))
                del model, built, prediction, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def test_cafe_requested_and_resolved_configs_match(self) -> None:
        for model_name in _bgmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_bgmodel_CANONICAL_SEED)
                built = _bgmodel_build_model(model_name, torch.device('cpu'))
                requested = _bgmodel_model_constructor_kwargs(model_name)
                resolved = built.model.get_config()
                for key, requested_value in requested.items():
                    expected = (requested_value,) if key == 'padding' else requested_value
                    self.assertEqual(resolved[key], expected)
                self.assertFalse(resolved['learnable_sigma'])
                expected_rank = _bgmodel_CAFEPLUSFNO_FACTORIZATIONS[model_name]['rank']
                self.assertEqual(built.rank, expected_rank)

    def test_cafe_checkpoint_configuration_reconstructs_strictly(self) -> None:
        sample = torch.randn(1, 10, 32)
        for model_name in _bgmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_bgmodel_CANONICAL_SEED)
                built = _bgmodel_build_model(model_name, torch.device('cpu'))
                expected = built.model(sample).detach()
                state = built.model.state_dict()
                reconstructed = CAFEPlusFNO1D(**built.configuration)
                reconstructed.load_state_dict(state, strict=True)
                actual = reconstructed(sample).detach()
                self.assertTrue(torch.equal(actual, expected))

    def test_cafe_random_features_follow_global_seed(self) -> None:
        for model_name in _bgmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                first = _bgmodel__g_buffers(_bgmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(42)
                second = _bgmodel__g_buffers(_bgmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(73)
                different = _bgmodel__g_buffers(_bgmodel_build_model(model_name, torch.device('cpu')).model)
                self.assertEqual(tuple(first), tuple(second))
                self.assertTrue(all((torch.equal(first[key], second[key]) for key in first)))
                self.assertTrue(any((not torch.equal(first[key], different[key]) for key in first)))

# Migrated from tests/test_cfd1d_models.py






from experiments import train_cfd1d as _c1model_train_module


from experiments.configs.cfd1d import CAFEPLUSFNO_FACTORIZATIONS as _c1model_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _c1model_CANONICAL_SEED, EXPECTED_PINNED_PARAMETER_COUNTS as _c1model_EXPECTED_PINNED_PARAMETER_COUNTS, MODEL_CHOICES as _c1model_MODEL_CHOICES, model_constructor_kwargs as _c1model_model_constructor_kwargs

from experiments.train_cfd1d import build_model as _c1model_build_model, validate_baseline_parameter_count as _c1model_validate_baseline_parameter_count


_c1model_CAFE_MODELS = tuple(_c1model_CAFEPLUSFNO_FACTORIZATIONS)

@contextlib.contextmanager
def _c1model__released_amfno_cpu_compatibility(model_name: str):
    """Neutralize the released constructor's unused ``.cuda()`` in CPU tests."""
    if model_name == 'amfno':
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            yield
    else:
        yield

def _c1model__g_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class CFD1DModelTests(unittest.TestCase):

    def test_all_eight_released_baseline_parameter_counts(self) -> None:
        device = torch.device('cpu')
        for model_name, expected in _c1model_EXPECTED_PINNED_PARAMETER_COUNTS.items():
            with self.subTest(model=model_name), _c1model__released_amfno_cpu_compatibility(model_name):
                set_seed(_c1model_CANONICAL_SEED)
                built = _c1model_build_model(model_name, device)
                actual = _c1model_train_module.count_model_params(built.model)
                self.assertEqual(actual, expected)
                _c1model_validate_baseline_parameter_count(model_name, actual)
                del built
                gc.collect()

    def test_all_twelve_models_forward_backward_on_small_tensor(self) -> None:
        device = torch.device('cpu')
        x = torch.randn(1, 10, 64, device=device)
        for index, model_name in enumerate(_c1model_MODEL_CHOICES):
            with self.subTest(model=model_name), _c1model__released_amfno_cpu_compatibility(model_name):
                set_seed(_c1model_CANONICAL_SEED + index)
                built = _c1model_build_model(model_name, device)
                model = built.model
                prediction = model(x)
                self.assertEqual(tuple(prediction.shape), (1, 1, 64))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = prediction.square().mean()
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                if model_name in {'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'}:
                    generator = model.operator_layers[0].spectral.kernel_generator
                    kernel = generator((64,))
                    kernel_rms = kernel.abs().square().mean().sqrt()
                    self.assertAlmostEqual(float(kernel_rms.detach().item()), 0.001, places=8)
                print('CFD1D_SYNTHETIC_SMOKE ' + json.dumps({'model': model_name, 'parameter_count': _c1model_train_module.count_model_params(model), 'factorization': built.factorization, 'rank': built.rank, 'shape': list(prediction.shape)}, sort_keys=True))
                del model, built, prediction, loss
                gc.collect()

    def test_cafe_requested_and_resolved_configs_match(self) -> None:
        for model_name in _c1model_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_c1model_CANONICAL_SEED)
                built = _c1model_build_model(model_name, torch.device('cpu'))
                requested = _c1model_model_constructor_kwargs(model_name)
                resolved = built.model.get_config()
                for key, requested_value in requested.items():
                    expected = (requested_value,) if key == 'padding' else requested_value
                    self.assertEqual(resolved[key], expected)
                self.assertEqual(resolved['input_dim'], 10)
                self.assertEqual(resolved['output_dim'], 1)
                self.assertFalse(resolved['learnable_sigma'])
                self.assertEqual(built.rank, _c1model_CAFEPLUSFNO_FACTORIZATIONS[model_name]['rank'])

    def test_cafe_checkpoint_configuration_reconstructs_strictly(self) -> None:
        sample = torch.randn(1, 10, 32)
        for model_name in _c1model_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_c1model_CANONICAL_SEED)
                built = _c1model_build_model(model_name, torch.device('cpu'))
                expected = built.model(sample).detach()
                state = built.model.state_dict()
                reconstructed = CAFEPlusFNO1D(**built.configuration)
                reconstructed.load_state_dict(state, strict=True)
                actual = reconstructed(sample).detach()
                self.assertTrue(torch.equal(actual, expected))

    def test_cafe_random_features_follow_global_seed(self) -> None:
        for model_name in _c1model_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                first = _c1model__g_buffers(_c1model_build_model(model_name, torch.device('cpu')).model)
                set_seed(42)
                second = _c1model__g_buffers(_c1model_build_model(model_name, torch.device('cpu')).model)
                set_seed(73)
                different = _c1model__g_buffers(_c1model_build_model(model_name, torch.device('cpu')).model)
                self.assertEqual(tuple(first), tuple(second))
                self.assertTrue(all((torch.equal(first[key], second[key]) for key in first)))
                self.assertTrue(any((not torch.equal(first[key], different[key]) for key in first)))

# Migrated from tests/test_cfd2d_models.py





from experiments import train_cfd2d as _c2model_train_module


from experiments.configs.cfd2d import CAFEPLUSFNO_FACTORIZATIONS as _c2model_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _c2model_CANONICAL_SEED, EXPECTED_PINNED_PARAMETER_COUNTS as _c2model_EXPECTED_PINNED_PARAMETER_COUNTS, EXPECTED_CAFE_PARAMETER_COUNTS as _c2model_EXPECTED_CAFE_PARAMETER_COUNTS, MODEL_CHOICES as _c2model_MODEL_CHOICES, model_constructor_kwargs as _c2model_model_constructor_kwargs

from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D

_c2model_CAFE_MODELS = tuple(_c2model_CAFEPLUSFNO_FACTORIZATIONS)

@contextlib.contextmanager
def _c2model__released_amfno_cpu_compatibility(model_name: str):
    """Neutralize the released constructor's unused ``.cuda()`` in CPU tests."""
    if model_name == 'amfno':
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            yield
    else:
        yield

def _c2model__g_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class CFD2DModelTests(unittest.TestCase):

    def test_all_eight_released_baseline_parameter_counts(self) -> None:
        for model_name, expected in _c2model_EXPECTED_PINNED_PARAMETER_COUNTS.items():
            with self.subTest(model=model_name), _c2model__released_amfno_cpu_compatibility(model_name):
                set_seed(_c2model_CANONICAL_SEED)
                built = _c2model_train_module.build_model(model_name, torch.device('cpu'))
                actual = _c2model_train_module.count_model_params(built.model)
                self.assertEqual(actual, expected)
                _c2model_train_module.validate_model_parameter_count(model_name, actual)
                del built
                gc.collect()

    def test_all_four_cafe_parameter_counts(self) -> None:
        for model_name, expected in _c2model_EXPECTED_CAFE_PARAMETER_COUNTS.items():
            with self.subTest(model=model_name):
                set_seed(_c2model_CANONICAL_SEED)
                built = _c2model_train_module.build_model(model_name, torch.device('cpu'))
                actual = _c2model_train_module.count_model_params(built.model)
                self.assertEqual(actual, expected)
                _c2model_train_module.validate_model_parameter_count(model_name, actual)

    def test_all_twelve_models_forward_backward_on_small_tensor(self) -> None:
        x = torch.randn(1, 5, 16, 16)
        for index, model_name in enumerate(_c2model_MODEL_CHOICES):
            with self.subTest(model=model_name), _c2model__released_amfno_cpu_compatibility(model_name):
                set_seed(_c2model_CANONICAL_SEED + index)
                built = _c2model_train_module.build_model(model_name, torch.device('cpu'))
                prediction = built.model(x)
                self.assertEqual(tuple(prediction.shape), (1, 1, 16, 16))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = prediction.square().mean()
                loss.backward()
                gradients = [value.grad for value in built.model.parameters() if value.requires_grad and value.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(item).all().item() for item in gradients)))
                if model_name in _c2model_CAFE_MODELS and model_name != 'cafe_plus_fno':
                    generator = built.model.operator_layers[0].spectral.kernel_generator
                    kernel = generator((16, 16))
                    self.assertAlmostEqual(float(kernel.abs().square().mean().sqrt().item()), 0.001, places=8)
                del built, prediction, loss
                gc.collect()

    def test_cafe_requested_resolved_and_checkpoint_configs_match(self) -> None:
        sample = torch.randn(1, 5, 16, 16)
        for model_name in _c2model_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_c2model_CANONICAL_SEED)
                built = _c2model_train_module.build_model(model_name, torch.device('cpu'))
                requested = _c2model_model_constructor_kwargs(model_name)
                resolved = built.model.get_config()
                for key, requested_value in requested.items():
                    expected = (requested_value, requested_value) if key == 'padding' else requested_value
                    self.assertEqual(resolved[key], expected)
                expected = built.model(sample).detach()
                reconstructed = CAFEPlusFNO2D(**built.configuration)
                reconstructed.load_state_dict(built.model.state_dict(), strict=True)
                actual = reconstructed(sample).detach()
                self.assertTrue(torch.equal(actual, expected))

    def test_cafe_random_features_follow_only_the_global_seed(self) -> None:
        for model_name in _c2model_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                first = _c2model__g_buffers(_c2model_train_module.build_model(model_name, torch.device('cpu')).model)
                set_seed(42)
                second = _c2model__g_buffers(_c2model_train_module.build_model(model_name, torch.device('cpu')).model)
                set_seed(73)
                different = _c2model__g_buffers(_c2model_train_module.build_model(model_name, torch.device('cpu')).model)
                self.assertEqual(tuple(first), tuple(second))
                self.assertTrue(all((torch.equal(first[key], second[key]) for key in first)))
                self.assertTrue(any((not torch.equal(first[key], different[key]) for key in first)))

# Migrated from tests/test_train_ns.py


import os


from pathlib import Path


from experiments import train_ns2d as train_ns_module

from neuralop import LpLoss as _nsmodel_LpLoss

from neuralop.training import AdamW



from experiments.configs.ns2d import CAFEPLUSFNO_FACTORIZATIONS as _nsmodel_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _nsmodel_CANONICAL_SEED, MODEL_CHOICES as _nsmodel_MODEL_CHOICES, RESOLUTION as _nsmodel_RESOLUTION, model_constructor_kwargs as _nsmodel_model_constructor_kwargs

from experiments.train_ns2d import build_model as _nsmodel_build_model, load_ns, resolve_device

_nsmodel_CAFE_MODELS = tuple(_nsmodel_CAFEPLUSFNO_FACTORIZATIONS)

def _nsmodel__g_buffers(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class NSModelBuildTests(unittest.TestCase):

    def test_author_local_neuraloperator_source(self) -> None:
        audited = train_ns_module.author_local_import_audit()
        self.assertTrue(audited['neuralop'].startswith('third_party/SirenFNO/neuralop/'))
        self.assertNotIn('site-packages', audited['neuralop'].lower())

    def test_all_twelve_models_forward_backward_on_small_tensor(self) -> None:
        x = torch.randn(1, 1, 16, 16)
        target = torch.randn_like(x)
        for index, model_name in enumerate(_nsmodel_MODEL_CHOICES):
            with self.subTest(model=model_name):
                set_seed(_nsmodel_CANONICAL_SEED + index)
                built = _nsmodel_build_model(model_name, torch.device('cpu'))
                model = built.model
                try:
                    prediction = model(x=x, y=target)
                    self.assertEqual(tuple(prediction.shape), tuple(target.shape))
                    self.assertTrue(torch.isfinite(prediction).all().item())
                    loss = _nsmodel_LpLoss(d=2, p=2)(prediction, y=target)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                    if model_name in {'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'}:
                        generator = model.operator_layers[0].spectral.kernel_generator
                        kernel = generator((16, 16))
                        kernel_rms = kernel.abs().square().mean().sqrt()
                        self.assertTrue(torch.isfinite(kernel).all().item())
                        self.assertAlmostEqual(float(kernel_rms.detach().item()), 0.001, places=8)
                        factor_gradients = [parameter.grad for parameter in generator.parameters() if parameter.requires_grad and parameter.grad is not None]
                        self.assertTrue(factor_gradients)
                        self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in factor_gradients)))
                        self.assertTrue(any((torch.count_nonzero(gradient).item() > 0 for gradient in factor_gradients)))
                    print('NS_SYNTHETIC_SMOKE ' + json.dumps({'model': model_name, 'parameter_count': count_model_params(model), 'factorization': built.factorization, 'rank': built.rank, 'shape': list(prediction.shape)}, sort_keys=True))
                finally:
                    del model, built
                    gc.collect()

    def test_cafe_requested_and_resolved_configs_match(self) -> None:
        for model_name in _nsmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_nsmodel_CANONICAL_SEED)
                built = _nsmodel_build_model(model_name, torch.device('cpu'))
                requested = _nsmodel_model_constructor_kwargs(model_name)
                resolved = built.model.get_config()
                for key, requested_value in requested.items():
                    expected = requested_value
                    if key == 'padding' and isinstance(requested_value, int):
                        expected = (requested_value, requested_value)
                    self.assertEqual(resolved[key], expected)
                self.assertFalse(resolved['learnable_sigma'])
                self.assertEqual(built.rank, 16)

    def test_cafe_g_follows_global_experiment_seed(self) -> None:
        for model_name in _nsmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                first = _nsmodel__g_buffers(_nsmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(42)
                second = _nsmodel__g_buffers(_nsmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(73)
                different = _nsmodel__g_buffers(_nsmodel_build_model(model_name, torch.device('cpu')).model)
                self.assertEqual(tuple(first), tuple(second))
                self.assertTrue(all((torch.equal(first[key], second[key]) for key in first)))
                self.assertTrue(any((not torch.equal(first[key], different[key]) for key in first)))

# Migrated from tests/test_reacdiff_models.py





from experiments import train_reacdiff as _rdmodel_train_module


from experiments.configs.reacdiff1d import CAFEPLUSFNO_FACTORIZATIONS as _rdmodel_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _rdmodel_CANONICAL_SEED, EXPECTED_CAFE_PARAMETER_COUNTS as _rdmodel_EXPECTED_CAFE_PARAMETER_COUNTS, EXPECTED_PINNED_PARAMETER_COUNTS as _rdmodel_EXPECTED_PINNED_PARAMETER_COUNTS, MODEL_CHOICES as _rdmodel_MODEL_CHOICES, model_constructor_kwargs as _rdmodel_model_constructor_kwargs

from experiments.train_reacdiff import build_model as _rdmodel_build_model, validate_model_parameter_count


_rdmodel_CAFE_MODELS = tuple(_rdmodel_CAFEPLUSFNO_FACTORIZATIONS)

@contextlib.contextmanager
def _rdmodel__released_amfno_cpu_compatibility(model_name: str):
    if model_name == 'amfno':
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            yield
    else:
        yield

def _rdmodel__g_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

class ReacDiffModelTests(unittest.TestCase):

    def test_all_twelve_exact_parameter_counts(self) -> None:
        expected = {**_rdmodel_EXPECTED_PINNED_PARAMETER_COUNTS, **_rdmodel_EXPECTED_CAFE_PARAMETER_COUNTS}
        self.assertEqual(set(expected), set(_rdmodel_MODEL_CHOICES))
        for model_name in _rdmodel_MODEL_CHOICES:
            with self.subTest(model=model_name), _rdmodel__released_amfno_cpu_compatibility(model_name):
                set_seed(_rdmodel_CANONICAL_SEED)
                built = _rdmodel_build_model(model_name, torch.device('cpu'))
                actual = _rdmodel_train_module.count_model_params(built.model)
                self.assertEqual(actual, expected[model_name])
                validate_model_parameter_count(model_name, actual)
                del built
                gc.collect()

    def test_all_twelve_forward_backward_and_optimizer_step(self) -> None:
        x = torch.randn(1, 10, 64)
        for index, model_name in enumerate(_rdmodel_MODEL_CHOICES):
            with self.subTest(model=model_name), _rdmodel__released_amfno_cpu_compatibility(model_name):
                set_seed(_rdmodel_CANONICAL_SEED + index)
                built = _rdmodel_build_model(model_name, torch.device('cpu'))
                model = built.model
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                prediction = model(x)
                self.assertEqual(tuple(prediction.shape), (1, 1, 64))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = prediction.square().mean()
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                optimizer.step()
                self.assertTrue(all((torch.isfinite(parameter).all().item() for parameter in model.parameters())))
                del optimizer, model, built, prediction, loss
                gc.collect()

    def test_cafe_requested_resolved_and_checkpoint_configs_match(self) -> None:
        sample = torch.randn(1, 10, 32)
        for model_name in _rdmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(_rdmodel_CANONICAL_SEED)
                built = _rdmodel_build_model(model_name, torch.device('cpu'))
                requested = _rdmodel_model_constructor_kwargs(model_name)
                resolved = built.model.get_config()
                for key, requested_value in requested.items():
                    expected = (requested_value,) if key == 'padding' and isinstance(requested_value, int) else requested_value
                    self.assertEqual(resolved[key], expected)
                expected_output = built.model(sample).detach()
                reconstructed = CAFEPlusFNO1D(**built.configuration)
                reconstructed.load_state_dict(built.model.state_dict(), strict=True)
                actual_output = reconstructed(sample).detach()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test_cafe_random_features_follow_global_seed(self) -> None:
        for model_name in _rdmodel_CAFE_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                first = _rdmodel__g_buffers(_rdmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(42)
                second = _rdmodel__g_buffers(_rdmodel_build_model(model_name, torch.device('cpu')).model)
                set_seed(73)
                different = _rdmodel__g_buffers(_rdmodel_build_model(model_name, torch.device('cpu')).model)
                self.assertEqual(tuple(first), tuple(second))
                self.assertTrue(all((torch.equal(first[key], second[key]) for key in first)))
                self.assertTrue(any((not torch.equal(first[key], different[key]) for key in first)))
