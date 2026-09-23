"""Role-based tests consolidated from 8 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_airfoil_pipeline.py
import os

import tempfile

import unittest

from pathlib import Path

from unittest.mock import patch

import numpy as np

import torch

from torch.utils.data import RandomSampler, SequentialSampler

from experiments import train_airfoil

from experiments.common.seed import set_seed

from experiments.configs.airfoil import CANONICAL_SEED as _afpipe_CANONICAL_SEED, DATASET_FILENAMES, N_TEST as _afpipe_N_TEST, N_TRAIN as _afpipe_N_TRAIN, RESOLUTION as _afpipe_RESOLUTION

@unittest.skipUnless(os.environ.get('AIRFOIL_DATA_ROOT'), 'Set AIRFOIL_DATA_ROOT to run the official Airfoil data smoke suite.')
class AirfoilOfficialDataSmokeTests(unittest.TestCase):

    def test_official_hash_shape_split_and_target_extraction(self) -> None:
        root = Path(os.environ['AIRFOIL_DATA_ROOT']).expanduser()
        with patch('urllib.request.urlopen', side_effect=AssertionError('network access is forbidden')) as network:
            identity = train_airfoil.verify_airfoil_dataset(root)
            set_seed(_afpipe_CANONICAL_SEED)
            data = train_airfoil.load_airfoil(root)
        network.assert_not_called()
        self.assertEqual(set(identity['files']), set(DATASET_FILENAMES))
        self.assertEqual(data.total_samples, 2490)
        self.assertEqual(data.input_shape, (*_afpipe_RESOLUTION, 2))
        self.assertEqual(data.target_shape, (*_afpipe_RESOLUTION, 1))
        self.assertEqual(len(data.train_loader.dataset), _afpipe_N_TRAIN)
        self.assertEqual(len(data.test_loader.dataset), _afpipe_N_TEST)
        q_values = np.load(root / 'NACA_Cylinder_Q.npy', mmap_mode='r', allow_pickle=False)
        train_target = data.train_loader.dataset.tensors[1]
        test_target = data.test_loader.dataset.tensors[1]
        np.testing.assert_array_equal(train_target[0, ..., 0].numpy(), q_values[0, 4].astype(np.float32))
        np.testing.assert_array_equal(test_target[0, ..., 0].numpy(), q_values[_afpipe_N_TRAIN, 4].astype(np.float32))

# Migrated from tests/test_burgers_pipeline.py
import csv

import inspect




from argparse import Namespace


from types import SimpleNamespace





from experiments import train_burgers as _bgpipe_train_module


from experiments.configs.burgers1d import BATCH_SIZE as _bgpipe_BATCH_SIZE, CANONICAL_SEED as _bgpipe_CANONICAL_SEED, DATASET_FILENAME as _bgpipe_DATASET_FILENAME, INPUT_STEPS as _bgpipe_INPUT_STEPS, N_TEST as _bgpipe_N_TEST, N_TRAIN as _bgpipe_N_TRAIN, RESOLUTION as _bgpipe_RESOLUTION, ROLLOUT as _bgpipe_ROLLOUT

from experiments.train_burgers import BurgersData, build_data_loaders as _bgpipe_build_data_loaders, fixed_split_indices as _bgpipe_fixed_split_indices, load_burgers, run_training_loop as _bgpipe_run_training_loop, verify_burgers_dataset

class _bgpipe__StopAfterBuild(RuntimeError):
    pass

class _bgpipe__TrackingAdamW(torch.optim.AdamW):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_step_learning_rates: list[float] = []

    def step(self, closure=None):
        self.training_step_learning_rates.append(float(self.param_groups[0]['lr']))
        return super().step(closure)

@unittest.skipUnless(os.environ.get('BURGERS_DATA_ROOT'), 'Set BURGERS_DATA_ROOT to run the verified offline Burgers data smoke test.')
class BurgersOfficialDataSmokeTests(unittest.TestCase):

    def test_verified_data_loads_with_network_disabled(self) -> None:
        data_root = Path(os.environ['BURGERS_DATA_ROOT']).expanduser()
        verify_burgers_dataset(data_root)
        set_seed(_bgpipe_CANONICAL_SEED)
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')), patch('requests.get', side_effect=AssertionError('network')):
            data = load_burgers(data_root)
            x_batch, y_batch = next(iter(data.train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_bgpipe_INPUT_STEPS, _bgpipe_RESOLUTION))
        self.assertEqual(tuple(y_batch.shape[1:]), (_bgpipe_ROLLOUT, _bgpipe_RESOLUTION))
        self.assertEqual(len(data.train_loader.dataset), _bgpipe_N_TRAIN)
        self.assertEqual(len(data.test_eval_loader.dataset), _bgpipe_N_TEST)

# Migrated from tests/test_cfd1d_pipeline.py
import contextlib




import subprocess






import h5py




from experiments import train_cfd1d as _c1pipe_train_module


from experiments.configs.cfd1d import BATCH_SIZE as _c1pipe_BATCH_SIZE, CANONICAL_SEED as _c1pipe_CANONICAL_SEED, DATASET_FILENAME as _c1pipe_DATASET_FILENAME, INPUT_STEPS as _c1pipe_INPUT_STEPS, N_TEST as _c1pipe_N_TEST, N_TRAIN as _c1pipe_N_TRAIN, RESOLUTION as _c1pipe_RESOLUTION, ROLLOUT as _c1pipe_ROLLOUT, SELECTED_FIELD

from experiments.train_cfd1d import CFD1DData, build_data_loaders as _c1pipe_build_data_loaders, fixed_split_indices as _c1pipe_fixed_split_indices, inspect_source_data as _c1pipe_inspect_source_data, load_cfd1d, rollout_cfd1d, run_training_loop as _c1pipe_run_training_loop, verify_cfd1d_dataset

class _c1pipe__StopAfterBuild(RuntimeError):
    pass

class _c1pipe__TrackingAdamW(torch.optim.AdamW):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step_learning_rates: list[float] = []

    def step(self, closure=None):
        self.training_step_learning_rates.append(float(self.param_groups[0]['lr']))
        return super().step(closure)

@unittest.skipUnless(os.environ.get('CFD1D_DATA_ROOT'), 'Set CFD1D_DATA_ROOT to run the verified offline CFD-1D data smoke test.')
class CFD1DOfficialDataSmokeTests(unittest.TestCase):

    def test_verified_vx_loads_with_network_disabled(self) -> None:
        data_root = Path(os.environ['CFD1D_DATA_ROOT']).expanduser()
        identity = verify_cfd1d_dataset(data_root)
        self.assertEqual(identity['files'][_c1pipe_DATASET_FILENAME]['sha256'], '86a2b8cf81f40191dbc40a7c2a9b268784979f2c1c269b59daa23c83885ebe8f')
        set_seed(_c1pipe_CANONICAL_SEED)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('network')))
            stack.enter_context(patch('requests.get', side_effect=AssertionError('network')))
            stack.enter_context(patch.object(_c1pipe_train_module, 'N_TRAIN', 2))
            stack.enter_context(patch.object(_c1pipe_train_module, 'N_TEST', 1))
            stack.enter_context(patch.object(_c1pipe_train_module, 'BATCH_SIZE', 2))
            data = load_cfd1d(data_root)
            x_batch, y_batch = next(iter(data.train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_c1pipe_INPUT_STEPS, _c1pipe_RESOLUTION))
        self.assertEqual(tuple(y_batch.shape[1:]), (_c1pipe_ROLLOUT, 1, _c1pipe_RESOLUTION))
        self.assertEqual(data.selected_field, 'Vx')

# Migrated from tests/test_cfd2d_pipeline.py














from experiments import train_cfd2d as _c2pipe_train_module


from experiments.configs.cfd2d import BATCH_SIZE as _c2pipe_BATCH_SIZE, CANONICAL_SEED as _c2pipe_CANONICAL_SEED, DATASET_FILENAME as _c2pipe_DATASET_FILENAME, INPUT_STEPS as _c2pipe_INPUT_STEPS, N_TEST as _c2pipe_N_TEST, N_TRAIN as _c2pipe_N_TRAIN, RESOLUTION as _c2pipe_RESOLUTION, ROLLOUT as _c2pipe_ROLLOUT

class _c2pipe__StopAfterBuild(RuntimeError):
    pass

class _c2pipe__TrackingAdamW(torch.optim.AdamW):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step_learning_rates: list[float] = []

    def step(self, closure=None):
        self.training_step_learning_rates.append(float(self.param_groups[0]['lr']))
        return super().step(closure)

@unittest.skipUnless(os.environ.get('CFD2D_DATA_ROOT'), 'Set CFD2D_DATA_ROOT to run the verified offline CFD-2D data smoke test.')
class CFD2DOfficialDataSmokeTests(unittest.TestCase):

    def test_verified_vx_loads_with_network_disabled(self) -> None:
        data_root = Path(os.environ['CFD2D_DATA_ROOT']).expanduser()
        identity = _c2pipe_train_module.verify_cfd2d_dataset(data_root)
        self.assertEqual(identity['files'][_c2pipe_DATASET_FILENAME]['sha256'], '8f21323cb7b61e80dd6d5ed93190bbba4b1a3e462b4be927d75a8fb0e286f3e4')
        set_seed(_c2pipe_CANONICAL_SEED)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('network')))
            stack.enter_context(patch('requests.get', side_effect=AssertionError('network')))
            stack.enter_context(patch.object(_c2pipe_train_module, 'N_TRAIN', 2))
            stack.enter_context(patch.object(_c2pipe_train_module, 'N_TEST', 1))
            stack.enter_context(patch.object(_c2pipe_train_module, 'BATCH_SIZE', 2))
            data = _c2pipe_train_module.load_cfd2d(data_root)
            x_batch, y_batch = next(iter(data.train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_c2pipe_INPUT_STEPS, *_c2pipe_RESOLUTION))
        self.assertEqual(tuple(y_batch.shape[1:]), (_c2pipe_ROLLOUT, 1, *_c2pipe_RESOLUTION))
        self.assertEqual(data.selected_field, 'Vx')

# Migrated from tests/test_darcy_models.py
import gc

import json






from experiments import train_darcy as train_darcy_module

from neuralop import LpLoss

from neuralop.training import AdamW

from neuralop.utils import count_model_params


from experiments.configs.darcy import AMFNO_CONFIG, BATCH_SIZE as _darcy_BATCH_SIZE, CANONICAL_SEED as _darcy_CANONICAL_SEED, EPOCHS, EVAL_INTERVAL, FNO_CONFIG, LEARNING_RATE, MODEL_CHOICES as _darcy_MODEL_CHOICES, N_TEST as _darcy_N_TEST, N_TRAIN as _darcy_N_TRAIN, RESOLUTION as _darcy_RESOLUTION, SCHEDULER_T_MAX, SIRENFNO_COMMON_CONFIG, SIRENFNO_FACTORIZATIONS, TEST_BATCH_SIZE as _darcy_TEST_BATCH_SIZE, TFNO_CP_CONFIG, UFNO_CONFIG, WEIGHT_DECAY, model_constructor_kwargs as _darcy_model_constructor_kwargs

from experiments.train_darcy import AUTHOR_LOCAL_IMPORT_PATHS, FNO, author_local_import_audit, build_model as _darcy_build_model, load_darcy, resolve_device as _darcy_resolve_device, rff_metadata, upstream_versions

@unittest.skipUnless(os.environ.get('DARCY_DATA_ROOT'), 'Set DARCY_DATA_ROOT to run the official-data model smoke suite.')
class DarcyModelSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = Path(os.environ['DARCY_DATA_ROOT']).expanduser()
        cls.device = _darcy_resolve_device(os.environ.get('DARCY_SMOKE_DEVICE', 'auto'))
        set_seed(_darcy_CANONICAL_SEED)
        cls.train_loader, _, cls.data_processor = load_darcy(cls.data_root)
        cls.data_processor = cls.data_processor.to(cls.device)
        batch = next(iter(cls.train_loader))
        cls.raw_sample = {key: value[:1].clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        cls.selected_models = tuple((name.strip() for name in os.environ.get('DARCY_SMOKE_MODELS', ','.join(_darcy_MODEL_CHOICES)).split(',') if name.strip()))
        unknown = sorted(set(cls.selected_models) - set(_darcy_MODEL_CHOICES))
        if unknown:
            raise ValueError(f'Unknown DARCY_SMOKE_MODELS entries: {unknown}')

    def test_forward_backward_and_optimizer_step(self) -> None:
        for model_name in self.selected_models:
            with self.subTest(model=model_name):
                model = optimizer = output = loss = processed = None
                try:
                    set_seed(_darcy_CANONICAL_SEED)
                    built = _darcy_build_model(model_name, self.device)
                    model = built.model
                    parameter_count = count_model_params(model)
                    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                    sample = {key: value.clone() if torch.is_tensor(value) else value for key, value in self.raw_sample.items()}
                    processed = self.data_processor.preprocess(sample)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(**processed)
                    output, processed = self.data_processor.postprocess(output, processed)
                    target = processed['y']
                    self.assertEqual(tuple(output.shape), tuple(target.shape))
                    self.assertEqual(tuple(output.shape[-2:]), (_darcy_RESOLUTION, _darcy_RESOLUTION))
                    self.assertTrue(torch.isfinite(output).all().item())
                    loss = LpLoss(d=2, p=2)(output, **processed)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                    optimizer.step()
                    result = {'model': model_name, 'model_class': model.__class__.__name__, 'parameter_count': parameter_count, 'factorization': built.factorization, 'rank': built.rank, 'input_shape': list(processed['x'].shape), 'output_shape': list(output.shape), 'relative_l2': float(loss.detach().cpu().item()), 'finite_output': True, 'finite_gradient': True, 'optimizer_step': True}
                    print('SMOKE_RESULT ' + json.dumps(result, sort_keys=True))
                finally:
                    del model, optimizer, output, loss, processed
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

# Migrated from tests/test_ns_offline_data.py





from contextlib import ExitStack




from experiments import train_ns2d as train_ns_module


from experiments.configs.ns2d import BATCH_SIZE as _nspipe_BATCH_SIZE, CANONICAL_SEED as _nspipe_CANONICAL_SEED, N_TEST as _nspipe_N_TEST, N_TRAIN as _nspipe_N_TRAIN, RESOLUTION as _nspipe_RESOLUTION, TEST_BATCH_SIZE as _nspipe_TEST_BATCH_SIZE

from neuralop.data.datasets import navier_stokes as navier_stokes_module

from neuralop.data.datasets import web_utils

def _write_small_ns_fixture(root: Path) -> None:
    values = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    torch.save({'x': values, 'y': values + 1}, root / 'nsforcing_train_128.pt')
    torch.save({'x': values + 2, 'y': values + 3}, root / 'nsforcing_test_128.pt')

def _blocked_network():
    stack = ExitStack()
    zenodo = stack.enter_context(patch.object(navier_stokes_module, 'download_from_zenodo_record', side_effect=AssertionError('training attempted a Zenodo download')))
    request = stack.enter_context(patch.object(web_utils.requests, 'get', side_effect=AssertionError('training attempted an HTTP request')))
    return (stack, zenodo, request)

def _qualified_type(value) -> str:
    return f'{type(value).__module__}.{type(value).__qualname__}'

def _loader_signature(loader) -> dict[str, object]:
    return {'dataset_length': len(loader.dataset), 'batch_size': loader.batch_size, 'num_workers': loader.num_workers, 'pin_memory': loader.pin_memory, 'persistent_workers': loader.persistent_workers, 'drop_last': loader.drop_last, 'sampler': _qualified_type(loader.sampler), 'batch_sampler': _qualified_type(loader.batch_sampler), 'timeout': loader.timeout, 'prefetch_factor': loader.prefetch_factor, 'generator_is_none': loader.generator is None, 'worker_init_fn_is_none': loader.worker_init_fn is None}

def _normalizer_signature(normalizer) -> dict[str, object] | None:
    if normalizer is None:
        return None
    return {'type': _qualified_type(normalizer), 'dim': list(normalizer.dim), 'eps': normalizer.eps, 'n_elements': normalizer.n_elements, 'mean': normalizer.mean.detach().clone(), 'std': normalizer.std.detach().clone()}

def _pipeline_snapshot(train_loader, test_loaders, data_processor):
    return {'train_loader': _loader_signature(train_loader), 'test_loaders': {resolution: _loader_signature(loader) for resolution, loader in test_loaders.items()}, 'train_batch': {key: value.detach().clone() for key, value in next(iter(train_loader)).items()}, 'test_batches': {resolution: {key: value.detach().clone() for key, value in next(iter(loader)).items()} for resolution, loader in test_loaders.items()}, 'processor_type': _qualified_type(data_processor), 'input_normalizer': _normalizer_signature(data_processor.in_normalizer), 'output_normalizer': _normalizer_signature(data_processor.out_normalizer)}

def _assert_nested_equal(test_case: unittest.TestCase, first, second) -> None:
    if torch.is_tensor(first):
        test_case.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test_case.assertEqual(set(first), set(second))
        for key in first:
            _assert_nested_equal(test_case, first[key], second[key])
    else:
        test_case.assertEqual(first, second)

@unittest.skipUnless(os.environ.get('NS_DATA_ROOT'), 'Set NS_DATA_ROOT to compare the official and offline NS128 pipelines.')
class NSOfficialPipelineParityTests(unittest.TestCase):

    def test_official_and_offline_pipelines_are_identical(self) -> None:
        data_root = Path(os.environ['NS_DATA_ROOT']).expanduser()
        train_ns_module.verify_ns_dataset(data_root)
        official_kwargs = {'n_train': _nspipe_N_TRAIN, 'batch_size': _nspipe_BATCH_SIZE, 'train_resolution': _nspipe_RESOLUTION, 'test_resolutions': [_nspipe_RESOLUTION], 'n_tests': [_nspipe_N_TEST], 'test_batch_sizes': [_nspipe_TEST_BATCH_SIZE], 'data_root': str(data_root)}
        with patch.object(navier_stokes_module, 'download_from_zenodo_record', return_value=None) as official_download, patch.object(web_utils.requests, 'get', side_effect=AssertionError('official parity test attempted HTTP')) as official_request:
            set_seed(_nspipe_CANONICAL_SEED)
            official = navier_stokes_module.load_navier_stokes_pt(**official_kwargs)
            official_snapshot = _pipeline_snapshot(*official)
        official_download.assert_called_once()
        official_request.assert_not_called()
        del official
        gc.collect()
        stack, offline_download, offline_request = _blocked_network()
        with stack:
            set_seed(_nspipe_CANONICAL_SEED)
            offline = train_ns_module.load_ns(data_root)
            offline_snapshot = _pipeline_snapshot(*offline)
        offline_download.assert_not_called()
        offline_request.assert_not_called()
        _assert_nested_equal(self, official_snapshot, offline_snapshot)

# Migrated from tests/test_train_ns.py











from experiments.configs.ns2d import CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _nsmodel_CANONICAL_SEED, MODEL_CHOICES as _nsmodel_MODEL_CHOICES, RESOLUTION as _nsmodel_RESOLUTION, model_constructor_kwargs as _nsmodel_model_constructor_kwargs

from experiments.train_ns2d import build_model as _nsmodel_build_model, load_ns, resolve_device as _nsmodel_resolve_device

CAFE_MODELS = tuple(CAFEPLUSFNO_FACTORIZATIONS)

def _g_buffers(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if name.endswith('.G')}

@unittest.skipUnless(os.environ.get('NS_DATA_ROOT'), 'Set NS_DATA_ROOT to run the official NS128 data smoke suite.')
class NSOfficialDataSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = _nsmodel_resolve_device(os.environ.get('NS_SMOKE_DEVICE', 'auto'))
        cls.data_root = Path(os.environ['NS_DATA_ROOT']).expanduser()
        set_seed(_nsmodel_CANONICAL_SEED)
        train_loader, _, data_processor = load_ns(cls.data_root)
        cls.data_processor = data_processor.to(cls.device)
        batch = next(iter(train_loader))
        cls.raw_sample = {key: value[:1].clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        cls.selected_models = tuple((name.strip() for name in os.environ.get('NS_SMOKE_MODELS', ','.join(_nsmodel_MODEL_CHOICES)).split(',') if name.strip()))

    def test_official_batch_all_selected_models(self) -> None:
        for model_name in self.selected_models:
            with self.subTest(model=model_name):
                set_seed(_nsmodel_CANONICAL_SEED)
                built = _nsmodel_build_model(model_name, self.device)
                model = built.model
                optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                sample = {key: value.clone() if torch.is_tensor(value) else value for key, value in self.raw_sample.items()}
                processed = self.data_processor.preprocess(sample)
                self.assertEqual(processed['x'].device.type, self.device.type)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(**processed)
                prediction, processed = self.data_processor.postprocess(prediction, processed)
                target = processed['y']
                self.assertEqual(tuple(prediction.shape), tuple(target.shape))
                self.assertEqual(tuple(prediction.shape[-2:]), (_nsmodel_RESOLUTION, _nsmodel_RESOLUTION))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = LpLoss(d=2, p=2)(prediction, **processed)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                optimizer.step()
                print('NS_OFFICIAL_SMOKE ' + json.dumps({'model': model_name, 'parameter_count': count_model_params(model), 'factorization': built.factorization, 'rank': built.rank, 'input_shape': list(processed['x'].shape), 'output_shape': list(prediction.shape), 'relative_l2': float(loss.detach().cpu().item())}, sort_keys=True))
                del model, optimizer, prediction, loss, processed, built
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Migrated from tests/test_reacdiff_pipeline.py














from experiments import train_reacdiff as _rdpipe_train_module


from experiments.configs.reacdiff1d import BATCH_SIZE as _rdpipe_BATCH_SIZE, CANONICAL_SEED as _rdpipe_CANONICAL_SEED, DATASET_FILENAME as _rdpipe_DATASET_FILENAME, INPUT_STEPS as _rdpipe_INPUT_STEPS, N_TEST as _rdpipe_N_TEST, N_TRAIN as _rdpipe_N_TRAIN, RESOLUTION as _rdpipe_RESOLUTION, ROLLOUT as _rdpipe_ROLLOUT

from experiments.train_reacdiff import ReacDiffData, build_data_loaders as _rdpipe_build_data_loaders, fixed_split_indices as _rdpipe_fixed_split_indices, inspect_source_data as _rdpipe_inspect_source_data, load_reacdiff, rollout_reacdiff, run_training_loop as _rdpipe_run_training_loop, verify_reacdiff_dataset

class _rdpipe__StopAfterBuild(RuntimeError):
    pass

class _rdpipe__TrackingAdamW(torch.optim.AdamW):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step_learning_rates: list[float] = []

    def step(self, closure=None):
        self.training_step_learning_rates.append(float(self.param_groups[0]['lr']))
        return super().step(closure)

class _RecordingRolloutModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.outputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.weight * x.sum(dim=1, keepdim=True)
        output.retain_grad()
        self.outputs.append(output)
        return output

@unittest.skipUnless(os.environ.get('REACDIFF_DATA_ROOT'), 'Set REACDIFF_DATA_ROOT to run the verified ReacDiff data test.')
class ReacDiffOfficialDataTests(unittest.TestCase):

    def test_verified_tensor_loads_offline_with_exact_shape(self) -> None:
        data_root = Path(os.environ['REACDIFF_DATA_ROOT']).expanduser()
        identity = verify_reacdiff_dataset(data_root)
        self.assertEqual(identity['files'][_rdpipe_DATASET_FILENAME]['sha256'], '0ccd649b1d5ecca8a5ae417a506dec5ef57ea365f7bce46afa190d34fe02dcc7')
        self.assertEqual(_rdpipe_inspect_source_data(data_root / _rdpipe_DATASET_FILENAME), (10000, 101, _rdpipe_RESOLUTION, '/tensor'))
        set_seed(_rdpipe_CANONICAL_SEED)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('network')))
            stack.enter_context(patch('requests.get', side_effect=AssertionError('network')))
            stack.enter_context(patch.object(_rdpipe_train_module, 'N_TRAIN', 2))
            stack.enter_context(patch.object(_rdpipe_train_module, 'N_TEST', 1))
            stack.enter_context(patch.object(_rdpipe_train_module, 'BATCH_SIZE', 2))
            data = load_reacdiff(data_root)
            x_batch, y_batch = next(iter(data.train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_rdpipe_INPUT_STEPS, _rdpipe_RESOLUTION))
        self.assertEqual(tuple(y_batch.shape[1:]), (_rdpipe_ROLLOUT, 1, _rdpipe_RESOLUTION))
        self.assertEqual(data.dataset_key, '/tensor')
