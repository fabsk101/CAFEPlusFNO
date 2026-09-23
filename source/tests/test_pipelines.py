"""Role-based tests consolidated from 6 legacy modules."""

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

class AirfoilOfflinePipelineTests(unittest.TestCase):

    def _write_fixture(self, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples, height, width = (6, 4, 3)
        x_values = np.arange(samples * height * width, dtype=np.float64).reshape(samples, height, width)
        y_values = x_values + 1000.0
        q_values = np.zeros((samples, 5, height, width), dtype=np.float64)
        for channel in range(5):
            q_values[:, channel] = x_values + channel * 10000.0
        np.save(root / DATASET_FILENAMES[0], x_values)
        np.save(root / DATASET_FILENAMES[1], y_values)
        np.save(root / DATASET_FILENAMES[2], q_values)
        return (x_values, y_values, q_values)

    def test_local_loader_extracts_exact_fields_and_split_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x_values, y_values, q_values = self._write_fixture(root)
            with patch.multiple(train_airfoil, RAW_SAMPLE_COUNT=6, RESOLUTION=(4, 3), N_TRAIN=4, N_VAL=0, N_TEST=2), patch('urllib.request.urlopen', side_effect=AssertionError('network access is forbidden')):
                set_seed(_afpipe_CANONICAL_SEED)
                data = train_airfoil.load_airfoil(root)
            train_input, train_target = data.train_loader.dataset.tensors
            test_input, test_target = data.test_loader.dataset.tensors
            np.testing.assert_array_equal(train_input.numpy()[..., 0], x_values[:4])
            np.testing.assert_array_equal(train_input.numpy()[..., 1], y_values[:4])
            np.testing.assert_array_equal(train_target.numpy()[..., 0], q_values[:4, 4])
            np.testing.assert_array_equal(test_input.numpy()[..., 0], x_values[4:6])
            np.testing.assert_array_equal(test_target.numpy()[..., 0], q_values[4:6, 4])
            self.assertEqual(train_input.dtype, torch.float32)
            self.assertEqual(test_target.dtype, torch.float32)
            self.assertIsInstance(data.train_loader.sampler, RandomSampler)
            self.assertIsInstance(data.test_loader.sampler, SequentialSampler)
            self.assertEqual(data.train_loader.batch_size, 8)
            self.assertFalse(data.train_loader.drop_last)
            self.assertFalse(data.train_loader.pin_memory)
            self.assertEqual(data.train_loader.num_workers, 0)

    def test_missing_data_never_falls_back_to_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen', side_effect=AssertionError('network access is forbidden')) as network:
            with self.assertRaises(FileNotFoundError):
                train_airfoil.load_airfoil(Path(directory))
        network.assert_not_called()

    def test_checksum_failure_never_falls_back_to_network(self) -> None:
        with patch.object(train_airfoil, 'verify_dataset', side_effect=RuntimeError('bad hash')), patch('urllib.request.urlopen', side_effect=AssertionError('network access is forbidden')) as network:
            with self.assertRaisesRegex(RuntimeError, 'failed checksum'):
                train_airfoil.verify_airfoil_dataset(Path('data'))
        network.assert_not_called()

    def test_scheduler_steps_after_each_batch_and_once_at_epoch_end(self) -> None:
        inputs = torch.randn(4, 4, 4, 2)
        targets = torch.randn(4, 4, 4, 1)
        train_loader, test_loader = train_airfoil.build_data_loaders(inputs, targets, inputs[:2], targets[:2])
        data = train_airfoil.AirfoilData(train_loader=train_loader, test_loader=test_loader, total_samples=6, input_shape=(4, 4, 2), target_shape=(4, 4, 1))
        model = torch.nn.Conv2d(2, 1, kernel_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
        original_step = scheduler.step
        calls: list[None] = []

        def counted_step(*args, **kwargs):
            calls.append(None)
            return original_step(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(train_airfoil, 'EPOCHS', 1), patch.object(scheduler, 'step', side_effect=counted_step):
            history = train_airfoil.run_training_loop(model=model, data=data, optimizer=optimizer, scheduler=scheduler, device=torch.device('cpu'), csv_path=Path(directory) / 'training_log.csv')
        self.assertEqual(len(history), 1)
        self.assertEqual(len(calls), len(train_loader) + 1)

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

class BurgersPipelineTests(unittest.TestCase):

    def test_training_module_has_no_acquisition_or_wandb_path(self) -> None:
        source = inspect.getsource(_bgpipe_train_module)
        for forbidden in ('ensure_data_available', '_download_file', 'urllib.request', 'requests.get', 'wandb'):
            self.assertNotIn(forbidden, source)

    def test_author_local_implementations_are_pinned(self) -> None:
        audited = _bgpipe_train_module.author_local_import_audit()
        for label, path in audited.items():
            if label == 'CAFEPlusFNO1D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO1D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_fixed_split_is_first_1000_then_next_200(self) -> None:
        train_indices, test_indices = _bgpipe_fixed_split_indices(10000)
        np.testing.assert_array_equal(train_indices, np.arange(0, _bgpipe_N_TRAIN))
        np.testing.assert_array_equal(test_indices, np.arange(_bgpipe_N_TRAIN, _bgpipe_N_TRAIN + _bgpipe_N_TEST))
        with self.assertRaises(ValueError):
            _bgpipe_fixed_split_indices(_bgpipe_N_TRAIN + _bgpipe_N_TEST - 1)

    def test_dataloader_options_match_pinned_source(self) -> None:
        train = torch.randn(64, _bgpipe_INPUT_STEPS + _bgpipe_ROLLOUT, 16)
        test = torch.randn(33, _bgpipe_INPUT_STEPS + _bgpipe_ROLLOUT, 16)
        train_loader, train_eval_loader, test_eval_loader = _bgpipe_build_data_loaders(train, test, 0.25, 1.5)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(train_eval_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_eval_loader.sampler, SequentialSampler)
        for loader in (train_loader, train_eval_loader, test_eval_loader):
            self.assertEqual(loader.batch_size, _bgpipe_BATCH_SIZE)
            self.assertEqual(loader.num_workers, 0)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.persistent_workers)
            self.assertIsNone(loader.generator)
            self.assertIsNone(loader.worker_init_fn)
            self.assertIsNone(loader.prefetch_factor)
        self.assertTrue(train_loader.drop_last)
        self.assertFalse(train_eval_loader.drop_last)
        self.assertFalse(test_eval_loader.drop_last)

    def test_loader_reuses_exact_pinned_preprocessing_functions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source = data_root / _bgpipe_DATASET_FILENAME
            source.touch()
            cache = data_root / '1D_Burgers_Sols_Nu0.001_rx1_rt1.hdf5'

            def fake_load(_path: str, indices: np.ndarray) -> torch.Tensor:
                return torch.zeros(len(indices), _bgpipe_INPUT_STEPS + _bgpipe_ROLLOUT, 8)
            attributes = {'source': str(source.resolve()), 'reduce_x': 1, 'reduce_t': 1}
            with patch.object(_bgpipe_train_module, '_preprocess_verified_dataset', return_value=cache) as preprocess, patch.object(_bgpipe_train_module, '_inspect_preprocessed_data', return_value=(10000, 201, _bgpipe_RESOLUTION, attributes)), patch.object(_bgpipe_train_module, 'load_subset_to_ram_1d_scalar', side_effect=fake_load) as load_subset, patch.object(_bgpipe_train_module, 'compute_mean_std_from_ram', return_value=(0.5, 2.0)) as statistics:
                loaded = load_burgers(data_root)
        preprocess.assert_called_once_with(source)
        self.assertEqual(load_subset.call_count, 2)
        np.testing.assert_array_equal(load_subset.call_args_list[0].args[1], np.arange(_bgpipe_N_TRAIN))
        np.testing.assert_array_equal(load_subset.call_args_list[1].args[1], np.arange(_bgpipe_N_TRAIN, _bgpipe_N_TRAIN + _bgpipe_N_TEST))
        statistics.assert_called_once()
        self.assertEqual(statistics.call_args.kwargs['max_traj_for_stats'], 200)
        self.assertEqual((loaded.mean, loaded.std), (0.5, 2.0))
        self.assertEqual(len(loaded.train_loader.dataset), _bgpipe_N_TRAIN)
        self.assertEqual(len(loaded.test_eval_loader.dataset), _bgpipe_N_TEST)

    def test_missing_data_never_attempts_network_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get, patch.object(_bgpipe_train_module, 'preprocess_decimated_hdf5_1d_scalar') as preprocess:
            with self.assertRaises(FileNotFoundError):
                load_burgers(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()
        preprocess.assert_not_called()

    def test_checksum_mismatch_never_attempts_network_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / _bgpipe_DATASET_FILENAME
            path.write_bytes(b'not the public dataset')
            with patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get:
                with self.assertRaisesRegex(RuntimeError, 'checksum verification'):
                    verify_burgers_dataset(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()

    def test_main_runtime_call_order_has_one_global_seed(self) -> None:
        order: list[str] = []
        arguments = Namespace(model='fno', seed=73, data_root=Path('data'), results_root=Path('results'), device='cpu', overwrite=False, allow_dirty_source=True)
        with patch.object(_bgpipe_train_module, 'parse_args', return_value=arguments), patch.object(_bgpipe_train_module, 'resolve_device', return_value=torch.device('cpu')), patch.object(_bgpipe_train_module, 'source_repository_state', side_effect=lambda **_kwargs: order.append('source') or {}), patch.object(_bgpipe_train_module, 'verify_burgers_dataset', side_effect=lambda _root: order.append('verify') or {}), patch.object(_bgpipe_train_module, 'set_seed', side_effect=lambda _seed: order.append('seed')) as seed, patch.object(_bgpipe_train_module, 'load_burgers', side_effect=lambda _root: order.append('load') or object()), patch.object(_bgpipe_train_module, 'build_model', side_effect=lambda *_args: (order.append('build'), (_ for _ in ()).throw(_bgpipe__StopAfterBuild()))[1]):
            with self.assertRaises(_bgpipe__StopAfterBuild):
                _bgpipe_train_module.main([])
        self.assertEqual(seed.call_count, 1)
        self.assertEqual(order, ['source', 'verify', 'seed', 'load', 'build'])

    def test_training_rollout_keeps_gradients_through_all_ten_steps(self) -> None:
        source = inspect.getsource(_bgpipe_run_training_loop)
        self.assertIn('pushforward_detach=PUSHFORWARD_DETACH', source)
        self.assertIn('rollout_loss_rel_l2(prediction, y_batch)', source)
        self.assertLess(source.index('scheduler.step()'), source.index('evaluate_rel_l2_metrics('))
        x = torch.randn(_bgpipe_BATCH_SIZE, _bgpipe_INPUT_STEPS, 16)
        y = torch.randn(_bgpipe_BATCH_SIZE, _bgpipe_ROLLOUT, 16)
        loaders = _bgpipe_build_data_loaders(torch.cat((x, y), dim=1), torch.cat((x, y), dim=1), 0.0, 1.0)
        data = BurgersData(train_loader=loaders[0], train_eval_loader=loaders[1], test_eval_loader=loaders[2], mean=0.0, std=1.0, total_trajectories=_bgpipe_BATCH_SIZE, time_steps=_bgpipe_INPUT_STEPS + _bgpipe_ROLLOUT, resolution=16, reduced_filename='fixture.hdf5')
        model = torch.nn.Conv1d(_bgpipe_INPUT_STEPS, 1, kernel_size=1)
        optimizer = _bgpipe__TrackingAdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(_bgpipe_train_module, 'EPOCHS', 1):
            csv_path = Path(directory) / 'training_log.csv'
            history = _bgpipe_run_training_loop(model=model, data=data, optimizer=optimizer, scheduler=scheduler, device=torch.device('cpu'), csv_path=csv_path)
            with csv_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(history), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]['epoch']), 1)
        logged_learning_rate = float(rows[0]['learning_rate'])
        self.assertTrue(optimizer.training_step_learning_rates)
        self.assertTrue(all((learning_rate == logged_learning_rate for learning_rate in optimizer.training_step_learning_rates)))
        self.assertEqual(logged_learning_rate, 0.001)
        self.assertEqual(float(optimizer.param_groups[0]['lr']), 0.0)
        self.assertTrue(all((parameter.grad is not None for parameter in model.parameters())))

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

class CFD1DPipelineTests(unittest.TestCase):

    def test_training_source_has_no_network_or_cache_acquisition(self) -> None:
        source = inspect.getsource(_c1pipe_train_module)
        load_source = inspect.getsource(load_cfd1d)
        for forbidden in ('ensure_data_available(', '_download_file(', 'urlopen(', 'requests.get', 'preprocess_decimated_hdf5('):
            self.assertNotIn(forbidden, load_source)

    def test_author_local_implementations_are_pinned(self) -> None:
        audited = _c1pipe_train_module.author_local_import_audit()
        for label, path in audited.items():
            if label == 'CAFEPlusFNO1D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO1D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_fixed_split_is_first_1800_then_next_200(self) -> None:
        train_indices, test_indices = _c1pipe_fixed_split_indices(10000)
        np.testing.assert_array_equal(train_indices, np.arange(0, _c1pipe_N_TRAIN))
        np.testing.assert_array_equal(test_indices, np.arange(_c1pipe_N_TRAIN, _c1pipe_N_TRAIN + _c1pipe_N_TEST))
        with self.assertRaises(ValueError):
            _c1pipe_fixed_split_indices(_c1pipe_N_TRAIN + _c1pipe_N_TEST - 1)

    def test_author_helper_selects_vx_from_public_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.hdf5'
            with h5py.File(path, 'w') as handle:
                for name in ('Vx', 'density', 'pressure'):
                    handle.create_dataset(name, shape=(8, 20, 16), dtype='f4')
            self.assertEqual(_c1pipe_inspect_source_data(path), (8, 20, 16))
        self.assertEqual(SELECTED_FIELD, 'Vx')

    def test_dataloader_options_and_channel_first_shapes_match_source(self) -> None:
        train = torch.randn(64, _c1pipe_INPUT_STEPS + _c1pipe_ROLLOUT, 16, 1)
        test = torch.randn(33, _c1pipe_INPUT_STEPS + _c1pipe_ROLLOUT, 16, 1)
        train_loader, train_eval_loader, test_eval_loader = _c1pipe_build_data_loaders(train, test)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(train_eval_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_eval_loader.sampler, SequentialSampler)
        for loader in (train_loader, train_eval_loader, test_eval_loader):
            self.assertEqual(loader.batch_size, _c1pipe_BATCH_SIZE)
            self.assertEqual(loader.num_workers, 0)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.persistent_workers)
            self.assertIsNone(loader.generator)
            self.assertIsNone(loader.worker_init_fn)
            self.assertIsNone(loader.prefetch_factor)
        self.assertTrue(train_loader.drop_last)
        self.assertFalse(train_eval_loader.drop_last)
        self.assertFalse(test_eval_loader.drop_last)
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_c1pipe_INPUT_STEPS, 16))
        self.assertEqual(tuple(y_batch.shape[1:]), (_c1pipe_ROLLOUT, 1, 16))

    def test_loader_reuses_pinned_ram_primitive_without_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source = data_root / _c1pipe_DATASET_FILENAME
            source.touch()

            def fake_load(_path: str, indices: np.ndarray) -> torch.Tensor:
                return torch.zeros(len(indices), 20, 16, 1)
            with patch.object(_c1pipe_train_module, 'N_TRAIN', 4), patch.object(_c1pipe_train_module, 'N_TEST', 2), patch.object(_c1pipe_train_module, 'EXPECTED_TIME_STEPS', 20), patch.object(_c1pipe_train_module, 'RESOLUTION', 16), patch.object(_c1pipe_train_module, 'inspect_source_data', return_value=(10, 20, 16)), patch.object(_c1pipe_train_module, 'load_subset_to_ram', side_effect=fake_load) as load_subset:
                loaded = load_cfd1d(data_root)
        self.assertEqual(load_subset.call_count, 2)
        np.testing.assert_array_equal(load_subset.call_args_list[0].args[1], np.arange(4))
        np.testing.assert_array_equal(load_subset.call_args_list[1].args[1], np.arange(4, 6))
        self.assertIsNone(loaded.train_loader.dataset.mean)
        self.assertIsNone(loaded.train_loader.dataset.std)
        self.assertEqual(loaded.selected_field, 'Vx')

    def test_missing_data_never_attempts_network_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get:
            with self.assertRaises(FileNotFoundError):
                load_cfd1d(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()

    def test_checksum_mismatch_never_attempts_network_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / _c1pipe_DATASET_FILENAME
            path.write_bytes(b'not the public dataset')
            with patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get:
                with self.assertRaisesRegex(RuntimeError, 'checksum verification'):
                    verify_cfd1d_dataset(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()

    def test_main_runtime_call_order_has_one_global_seed(self) -> None:
        order: list[str] = []
        arguments = Namespace(model='fno', seed=73, data_root=Path('data'), results_root=Path('results'), device='cpu', overwrite=False, allow_dirty_source=True)
        with patch.object(_c1pipe_train_module, 'parse_args', return_value=arguments), patch.object(_c1pipe_train_module, 'resolve_device', return_value=torch.device('cpu')), patch.object(_c1pipe_train_module, 'source_repository_state', side_effect=lambda **_kwargs: order.append('source') or {}), patch.object(_c1pipe_train_module, 'verify_cfd1d_dataset', side_effect=lambda _root: order.append('verify') or {}), patch.object(_c1pipe_train_module, 'set_seed', side_effect=lambda _seed: order.append('seed')) as seed, patch.object(_c1pipe_train_module, 'load_cfd1d', side_effect=lambda _root: order.append('load') or object()), patch.object(_c1pipe_train_module, 'build_model', side_effect=lambda *_args: (order.append('build'), (_ for _ in ()).throw(_c1pipe__StopAfterBuild()))[1]):
            with self.assertRaises(_c1pipe__StopAfterBuild):
                _c1pipe_train_module.main([])
        self.assertEqual(seed.call_count, 1)
        self.assertEqual(order, ['source', 'verify', 'seed', 'load', 'build'])

    def test_training_rollout_and_scheduler_order_match_pinned_source(self) -> None:
        source = inspect.getsource(_c1pipe_run_training_loop)
        self.assertIn('pushforward_detach=PUSHFORWARD_DETACH', source)
        self.assertIn('rollout_loss_lp(prediction, y_batch, lp_rel_train)', source)
        self.assertLess(source.index('scheduler.step()'), source.index('evaluate_corrected_relative_l2('))
        raw = torch.randn(_c1pipe_BATCH_SIZE, _c1pipe_INPUT_STEPS + _c1pipe_ROLLOUT, 16, 1)
        loaders = _c1pipe_build_data_loaders(raw, raw)
        data = CFD1DData(train_loader=loaders[0], train_eval_loader=loaders[1], test_eval_loader=loaders[2], total_trajectories=_c1pipe_BATCH_SIZE, time_steps=_c1pipe_INPUT_STEPS + _c1pipe_ROLLOUT, resolution=16, selected_field='Vx')
        model = torch.nn.Conv1d(_c1pipe_INPUT_STEPS, 1, kernel_size=1)
        optimizer = _c1pipe__TrackingAdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(_c1pipe_train_module, 'EPOCHS', 1):
            csv_path = Path(directory) / 'training_log.csv'
            history = _c1pipe_run_training_loop(model=model, data=data, optimizer=optimizer, scheduler=scheduler, device=torch.device('cpu'), csv_path=csv_path)
            with csv_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(history), 1)
        self.assertEqual(int(rows[0]['epoch']), 1)
        for metric_name in (
            'train_corrected_step_relative_l2',
            'train_corrected_trajectory_relative_l2',
            'test_corrected_step_relative_l2',
            'test_corrected_trajectory_relative_l2',
        ):
            self.assertTrue(np.isfinite(float(rows[0][metric_name])))
        self.assertEqual(int(rows[0]['train_evaluation_sample_count']), _c1pipe_BATCH_SIZE)
        self.assertEqual(int(rows[0]['test_evaluation_sample_count']), _c1pipe_BATCH_SIZE)
        self.assertEqual(int(rows[0]['evaluation_horizon']), _c1pipe_ROLLOUT)
        self.assertNotIn('train_step_relative_l2', rows[0])
        self.assertNotIn('train_full_relative_l2', rows[0])
        logged_learning_rate = float(rows[0]['learning_rate'])
        self.assertTrue(all((lr == logged_learning_rate for lr in optimizer.training_step_learning_rates)))
        self.assertEqual(logged_learning_rate, 0.001)
        self.assertEqual(float(optimizer.param_groups[0]['lr']), 0.0)
        self.assertTrue(all((parameter.grad is not None for parameter in model.parameters())))

    def test_rollout_is_scalar_channel_first_and_keeps_gradients(self) -> None:
        model = torch.nn.Conv1d(_c1pipe_INPUT_STEPS, 1, kernel_size=1)
        x = torch.randn(2, _c1pipe_INPUT_STEPS, 16)
        prediction = rollout_cfd1d(model, x, _c1pipe_ROLLOUT, pushforward_detach=False)
        self.assertEqual(tuple(prediction.shape), (2, _c1pipe_ROLLOUT, 1, 16))
        prediction.square().mean().backward()
        self.assertTrue(all((parameter.grad is not None for parameter in model.parameters())))

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

class CFD2DPipelineTests(unittest.TestCase):

    def test_training_loader_has_no_network_or_cache_acquisition(self) -> None:
        source = inspect.getsource(_c2pipe_train_module.load_cfd2d)
        for forbidden in ('ensure_data_available(', '_download_file(', 'urlopen(', 'requests.get', 'preprocess_decimated_hdf5('):
            self.assertNotIn(forbidden, source)

    def test_author_helper_selects_vx_and_split_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.hdf5'
            with h5py.File(path, 'w') as handle:
                for name in ('Vx', 'Vy', 'density', 'pressure'):
                    handle.create_dataset(name, shape=(8, 10, 16, 16), dtype='f4')
            self.assertEqual(_c2pipe_train_module.inspect_source_data(path), (8, 10, 16, 16))
        train_indices, test_indices = _c2pipe_train_module.fixed_split_indices(10000)
        np.testing.assert_array_equal(train_indices, np.arange(_c2pipe_N_TRAIN))
        np.testing.assert_array_equal(test_indices, np.arange(_c2pipe_N_TRAIN, _c2pipe_N_TRAIN + _c2pipe_N_TEST))

    def test_dataloader_options_and_channel_first_shapes_match_source(self) -> None:
        raw = torch.randn(64, _c2pipe_INPUT_STEPS + _c2pipe_ROLLOUT, 16, 16, 1)
        loaders = _c2pipe_train_module.build_data_loaders(raw, raw[:33])
        train_loader, train_eval_loader, test_eval_loader = loaders
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(train_eval_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_eval_loader.sampler, SequentialSampler)
        for loader in loaders:
            self.assertEqual(loader.batch_size, _c2pipe_BATCH_SIZE)
            self.assertEqual(loader.num_workers, 0)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.persistent_workers)
            self.assertIsNone(loader.generator)
            self.assertIsNone(loader.worker_init_fn)
            self.assertIsNone(loader.prefetch_factor)
        self.assertTrue(train_loader.drop_last)
        self.assertFalse(train_eval_loader.drop_last)
        self.assertFalse(test_eval_loader.drop_last)
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(tuple(x_batch.shape[1:]), (_c2pipe_INPUT_STEPS, 16, 16))
        self.assertEqual(tuple(y_batch.shape[1:]), (_c2pipe_ROLLOUT, 1, 16, 16))

    def test_pinned_ram_load_is_vx_only_and_unnormalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / _c2pipe_DATASET_FILENAME).touch()

            def fake_load(_path: str, indices: np.ndarray) -> torch.Tensor:
                return torch.zeros(len(indices), 10, 16, 16, 1)
            with patch.object(_c2pipe_train_module, 'N_TRAIN', 4), patch.object(_c2pipe_train_module, 'N_TEST', 2), patch.object(_c2pipe_train_module, 'EXPECTED_TIME_STEPS', 10), patch.object(_c2pipe_train_module, 'RESOLUTION', (16, 16)), patch.object(_c2pipe_train_module, 'inspect_source_data', return_value=(10, 10, 16, 16)), patch.object(_c2pipe_train_module, 'load_subset_to_ram', side_effect=fake_load) as load_subset:
                loaded = _c2pipe_train_module.load_cfd2d(root)
        self.assertEqual(load_subset.call_count, 2)
        np.testing.assert_array_equal(load_subset.call_args_list[0].args[1], np.arange(4))
        np.testing.assert_array_equal(load_subset.call_args_list[1].args[1], np.arange(4, 6))
        self.assertIsNone(loaded.train_loader.dataset.mean)
        self.assertIsNone(loaded.train_loader.dataset.std)
        self.assertEqual(loaded.selected_field, 'Vx')

    def test_missing_and_bad_data_never_use_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get:
            with self.assertRaises(FileNotFoundError):
                _c2pipe_train_module.load_cfd2d(Path(directory))
            (Path(directory) / _c2pipe_DATASET_FILENAME).write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'checksum verification'):
                _c2pipe_train_module.verify_cfd2d_dataset(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()

    def test_main_call_order_has_exactly_one_global_seed(self) -> None:
        order: list[str] = []
        arguments = Namespace(model='fno', seed=73, data_root=Path('data'), results_root=Path('results'), device='cpu', overwrite=False, allow_dirty_source=True)
        with patch.object(_c2pipe_train_module, 'parse_args', return_value=arguments), patch.object(_c2pipe_train_module, 'resolve_device', return_value=torch.device('cpu')), patch.object(_c2pipe_train_module, 'source_repository_state', side_effect=lambda **_kwargs: order.append('source') or {}), patch.object(_c2pipe_train_module, 'verify_cfd2d_dataset', side_effect=lambda _root: order.append('verify') or {}), patch.object(_c2pipe_train_module, 'set_seed', side_effect=lambda _seed: order.append('seed')) as seed, patch.object(_c2pipe_train_module, 'load_cfd2d', side_effect=lambda _root: order.append('load') or object()), patch.object(_c2pipe_train_module, 'build_model', side_effect=lambda *_args: (order.append('build'), (_ for _ in ()).throw(_c2pipe__StopAfterBuild()))[1]):
            with self.assertRaises(_c2pipe__StopAfterBuild):
                _c2pipe_train_module.main([])
        self.assertEqual(seed.call_count, 1)
        self.assertEqual(order, ['source', 'verify', 'seed', 'load', 'build'])

    def test_rollout_gradients_scheduler_order_and_lr_logging(self) -> None:
        source = inspect.getsource(_c2pipe_train_module.run_training_loop)
        self.assertIn('require_lp=True', source)
        self.assertLess(source.index('scheduler.step()'), source.index('evaluate_corrected_relative_l2('))
        raw = torch.randn(_c2pipe_BATCH_SIZE, _c2pipe_INPUT_STEPS + _c2pipe_ROLLOUT, 8, 8, 1)
        loaders = _c2pipe_train_module.build_data_loaders(raw, raw)
        data = _c2pipe_train_module.CFD2DData(train_loader=loaders[0], train_eval_loader=loaders[1], test_eval_loader=loaders[2], total_trajectories=_c2pipe_BATCH_SIZE, time_steps=_c2pipe_INPUT_STEPS + _c2pipe_ROLLOUT, spatial_shape=(8, 8), selected_field='Vx')
        model = torch.nn.Conv2d(_c2pipe_INPUT_STEPS, 1, kernel_size=1)
        optimizer = _c2pipe__TrackingAdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(_c2pipe_train_module, 'EPOCHS', 1):
            csv_path = Path(directory) / 'training_log.csv'
            history = _c2pipe_train_module.run_training_loop(model=model, data=data, optimizer=optimizer, scheduler=scheduler, device=torch.device('cpu'), csv_path=csv_path)
            with csv_path.open(newline='', encoding='utf-8') as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(len(history), 1)
        for metric_name in (
            'train_corrected_step_relative_l2',
            'train_corrected_trajectory_relative_l2',
            'test_corrected_step_relative_l2',
            'test_corrected_trajectory_relative_l2',
        ):
            self.assertTrue(np.isfinite(float(row[metric_name])))
        self.assertEqual(int(row['train_evaluation_sample_count']), _c2pipe_BATCH_SIZE)
        self.assertEqual(int(row['test_evaluation_sample_count']), _c2pipe_BATCH_SIZE)
        self.assertEqual(int(row['evaluation_horizon']), _c2pipe_ROLLOUT)
        self.assertNotIn('train_step_relative_l2', row)
        self.assertNotIn('train_full_relative_l2', row)
        self.assertEqual(float(row['learning_rate']), 0.001)
        self.assertTrue(all((lr == 0.001 for lr in optimizer.training_step_learning_rates)))
        self.assertEqual(float(optimizer.param_groups[0]['lr']), 0.0)
        self.assertTrue(all((parameter.grad is not None for parameter in model.parameters())))

    def test_rollout_shape_is_scalar_channel_first_and_connected(self) -> None:
        model = torch.nn.Conv2d(_c2pipe_INPUT_STEPS, 1, kernel_size=1)
        x = torch.randn(2, _c2pipe_INPUT_STEPS, 8, 8)
        prediction = _c2pipe_train_module.rollout_cfd2d(model, x, _c2pipe_ROLLOUT, pushforward_detach=False)
        self.assertEqual(tuple(prediction.shape), (2, _c2pipe_ROLLOUT, 1, 8, 8))
        prediction.square().mean().backward()
        self.assertTrue(all((parameter.grad is not None for parameter in model.parameters())))

# Migrated from tests/test_ns_offline_data.py
import gc





from contextlib import ExitStack




from experiments import train_ns2d as train_ns_module


from experiments.configs.ns2d import BATCH_SIZE as _nspipe_BATCH_SIZE, CANONICAL_SEED as _nspipe_CANONICAL_SEED, N_TEST as _nspipe_N_TEST, N_TRAIN as _nspipe_N_TRAIN, RESOLUTION as _nspipe_RESOLUTION, TEST_BATCH_SIZE

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

class NSOfflineLoaderTests(unittest.TestCase):

    def test_local_files_load_with_all_network_calls_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_small_ns_fixture(root)
            stack, zenodo, request = _blocked_network()
            with stack:
                train_loader, test_loaders, data_processor = train_ns_module.load_ns(root)
            self.assertEqual(len(train_loader.dataset), 4)
            self.assertEqual(len(test_loaders[128].dataset), 4)
            self.assertEqual(_qualified_type(data_processor).split('.')[-1], 'DefaultDataProcessor')
            zenodo.assert_not_called()
            request.assert_not_called()

    def test_missing_data_fails_closed_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack, zenodo, request = _blocked_network()
            with stack, self.assertRaisesRegex(RuntimeError, 'python -B data/download_data.py --dataset ns128 --data-root data'):
                train_ns_module.verify_ns_dataset(Path(directory))
            zenodo.assert_not_called()
            request.assert_not_called()

    def test_checksum_mismatch_fails_closed_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nsforcing_train_128.pt').write_bytes(b'invalid')
            (root / 'nsforcing_test_128.pt').write_bytes(b'invalid')
            stack, zenodo, request = _blocked_network()
            with stack, self.assertRaisesRegex(RuntimeError, 'failed checksum verification'):
                train_ns_module.verify_ns_dataset(root)
            zenodo.assert_not_called()
            request.assert_not_called()

    def test_main_uses_single_seed_between_verification_and_loader(self) -> None:
        events: list[str] = []

        class StopAfterModelOrderCheck(RuntimeError):
            pass

        class Processor:

            def to(self, _device):
                events.append('processor')
                return self
        args = Namespace(model='fno', seed=73, data_root=Path('data'), results_root=Path('results') / 'unused-ns-test', device='cpu', overwrite=False, allow_dirty_source=True)

        def record(name, result=None):

            def action(*_args, **_kwargs):
                events.append(name)
                return result
            return action

        def stop_at_model(*_args, **_kwargs):
            events.append('model')
            raise StopAfterModelOrderCheck
        with patch.object(train_ns_module, 'parse_args', return_value=args), patch.object(train_ns_module, 'resolve_device', return_value=torch.device('cpu')), patch.object(train_ns_module, 'source_repository_state', return_value={'source_repository_dirty': False}), patch.object(train_ns_module, 'verify_ns_dataset', side_effect=record('verify', {})), patch.object(train_ns_module, 'set_seed', side_effect=record('seed')) as seeded, patch.object(train_ns_module, 'load_ns', side_effect=record('loader', (object(), {}, Processor()))), patch.object(train_ns_module, 'build_model', side_effect=stop_at_model):
            with self.assertRaises(StopAfterModelOrderCheck):
                train_ns_module.main()
        self.assertEqual(events, ['verify', 'seed', 'loader', 'processor', 'model'])
        seeded.assert_called_once_with(73)

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

class ReacDiffPipelineTests(unittest.TestCase):

    def test_training_source_has_no_acquisition_or_network_path(self) -> None:
        module_source = inspect.getsource(_rdpipe_train_module)
        load_source = inspect.getsource(load_reacdiff)
        for forbidden in ('ensure_data_available(', '_download_file(', 'urlopen(', 'requests.get', 'preprocess_decimated_hdf5('):
            self.assertNotIn(forbidden, module_source)
            self.assertNotIn(forbidden, load_source)

    def test_fixed_split_is_first_1000_then_next_200(self) -> None:
        train_indices, test_indices = _rdpipe_fixed_split_indices(10000)
        np.testing.assert_array_equal(train_indices, np.arange(0, _rdpipe_N_TRAIN))
        np.testing.assert_array_equal(test_indices, np.arange(_rdpipe_N_TRAIN, _rdpipe_N_TRAIN + _rdpipe_N_TEST))
        with self.assertRaises(ValueError):
            _rdpipe_fixed_split_indices(_rdpipe_N_TRAIN + _rdpipe_N_TEST - 1)

    def test_pinned_helper_uniquely_selects_tensor_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.hdf5'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('tensor', shape=(8, 101, 16), dtype='f4')
                handle.create_dataset('t-coordinate', shape=(101,), dtype='f4')
            self.assertEqual(_rdpipe_inspect_source_data(path), (8, 101, 16, '/tensor'))

    def test_dataloader_options_and_shapes_match_released_source(self) -> None:
        train = torch.randn(64, _rdpipe_INPUT_STEPS + _rdpipe_ROLLOUT, 16, 1)
        test = torch.randn(33, _rdpipe_INPUT_STEPS + _rdpipe_ROLLOUT, 16, 1)
        loaders = _rdpipe_build_data_loaders(train, test)
        self.assertIsInstance(loaders[0].sampler, RandomSampler)
        self.assertIsInstance(loaders[1].sampler, SequentialSampler)
        self.assertIsInstance(loaders[2].sampler, SequentialSampler)
        for loader in loaders:
            self.assertEqual(loader.batch_size, _rdpipe_BATCH_SIZE)
            self.assertEqual(loader.num_workers, 0)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.persistent_workers)
            self.assertIsNone(loader.generator)
            self.assertIsNone(loader.worker_init_fn)
            self.assertIsNone(loader.prefetch_factor)
        self.assertTrue(loaders[0].drop_last)
        self.assertFalse(loaders[1].drop_last)
        self.assertFalse(loaders[2].drop_last)
        x_batch, y_batch = next(iter(loaders[0]))
        self.assertEqual(tuple(x_batch.shape[1:]), (_rdpipe_INPUT_STEPS, 16))
        self.assertEqual(tuple(y_batch.shape[1:]), (_rdpipe_ROLLOUT, 1, 16))
        self.assertIsNone(loaders[0].dataset.mean)
        self.assertIsNone(loaders[0].dataset.std)

    def test_missing_and_bad_data_never_attempt_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen') as urlopen, patch('requests.get') as requests_get:
            with self.assertRaises(FileNotFoundError):
                load_reacdiff(Path(directory))
            path = Path(directory) / _rdpipe_DATASET_FILENAME
            path.write_bytes(b'not the public dataset')
            with self.assertRaisesRegex(RuntimeError, 'checksum verification'):
                verify_reacdiff_dataset(Path(directory))
        urlopen.assert_not_called()
        requests_get.assert_not_called()

    def test_main_call_order_has_exactly_one_global_seed(self) -> None:
        order: list[str] = []
        arguments = Namespace(model='fno', seed=73, data_root=Path('data'), results_root=Path('results'), device='cpu', overwrite=False, allow_dirty_source=True)
        with patch.object(_rdpipe_train_module, 'parse_args', return_value=arguments), patch.object(_rdpipe_train_module, 'resolve_device', return_value=torch.device('cpu')), patch.object(_rdpipe_train_module, 'source_repository_state', side_effect=lambda **_kwargs: order.append('source') or {}), patch.object(_rdpipe_train_module, 'verify_reacdiff_dataset', side_effect=lambda _root: order.append('verify') or {}), patch.object(_rdpipe_train_module, 'set_seed', side_effect=lambda _seed: order.append('seed')) as seed, patch.object(_rdpipe_train_module, 'load_reacdiff', side_effect=lambda _root: order.append('load') or object()), patch.object(_rdpipe_train_module, 'build_model', side_effect=lambda *_args: (order.append('build'), (_ for _ in ()).throw(_rdpipe__StopAfterBuild()))[1]):
            with self.assertRaises(_rdpipe__StopAfterBuild):
                _rdpipe_train_module.main([])
        self.assertEqual(seed.call_count, 1)
        self.assertEqual(order, ['source', 'verify', 'seed', 'load', 'build'])

    def test_full_ten_step_rollout_keeps_every_step_in_graph(self) -> None:
        model = _RecordingRolloutModel()
        x = torch.randn(2, _rdpipe_INPUT_STEPS, 16)
        prediction = rollout_reacdiff(model, x, _rdpipe_ROLLOUT, pushforward_detach=False)
        self.assertEqual(tuple(prediction.shape), (2, _rdpipe_ROLLOUT, 1, 16))
        prediction[:, -1].square().mean().backward()
        self.assertEqual(len(model.outputs), _rdpipe_ROLLOUT)
        self.assertTrue(all((output.grad is not None for output in model.outputs)))
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(torch.isfinite(model.weight.grad).item())

    def test_scheduler_steps_exactly_once_after_each_train_loop(self) -> None:
        source = inspect.getsource(_rdpipe_run_training_loop)
        self.assertEqual(source.count('scheduler.step()'), 1)
        self.assertLess(source.index('optimizer.step()'), source.index('scheduler.step()'))
        self.assertLess(source.index('scheduler.step()'), source.index('evaluate_corrected_relative_l2('))
        raw = torch.randn(_rdpipe_BATCH_SIZE, _rdpipe_INPUT_STEPS + _rdpipe_ROLLOUT, 8, 1)
        loaders = _rdpipe_build_data_loaders(raw, raw)
        data = ReacDiffData(train_loader=loaders[0], train_eval_loader=loaders[1], test_eval_loader=loaders[2], total_trajectories=_rdpipe_BATCH_SIZE, time_steps=_rdpipe_INPUT_STEPS + _rdpipe_ROLLOUT, resolution=8, dataset_key='/tensor')
        model = torch.nn.Conv1d(_rdpipe_INPUT_STEPS, 1, kernel_size=1)
        optimizer = _rdpipe__TrackingAdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(_rdpipe_train_module, 'EPOCHS', 1):
            csv_path = Path(directory) / 'training_log.csv'
            history = _rdpipe_run_training_loop(model=model, data=data, optimizer=optimizer, scheduler=scheduler, device=torch.device('cpu'), csv_path=csv_path)
            with csv_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(history), 1)
        for metric_name in (
            'train_corrected_step_relative_l2',
            'train_corrected_trajectory_relative_l2',
            'test_corrected_step_relative_l2',
            'test_corrected_trajectory_relative_l2',
        ):
            self.assertTrue(np.isfinite(float(rows[0][metric_name])))
        self.assertEqual(int(rows[0]['train_evaluation_sample_count']), _rdpipe_BATCH_SIZE)
        self.assertEqual(int(rows[0]['test_evaluation_sample_count']), _rdpipe_BATCH_SIZE)
        self.assertEqual(int(rows[0]['evaluation_horizon']), _rdpipe_ROLLOUT)
        self.assertNotIn('train_step_relative_l2', rows[0])
        self.assertNotIn('train_full_relative_l2', rows[0])
        self.assertEqual(len(optimizer.training_step_learning_rates), 1)
        self.assertEqual(float(rows[0]['learning_rate']), 0.001)
        self.assertEqual(float(optimizer.param_groups[0]['lr']), 0.0)

