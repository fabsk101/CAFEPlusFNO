"""Role-based tests consolidated from 2 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_airfoil_cuda_smoke.py
import gc

import json

import os

import unittest

from pathlib import Path

from unittest.mock import patch

import torch

from experiments.configs.airfoil import BATCH_SIZE as _afcuda_BATCH_SIZE, EXPECTED_CAFE_PARAMETER_COUNTS, EXPECTED_PINNED_PARAMETER_COUNTS, LEARNING_RATE as _afcuda_LEARNING_RATE, MODEL_CHOICES as _afcuda_MODEL_CHOICES, PAPER_SEEDS, RESOLUTION, WEIGHT_DECAY as _afcuda_WEIGHT_DECAY

from experiments.common.seed import set_seed

from experiments import train_airfoil

@unittest.skipUnless(os.environ.get('AIRFOIL_DATA_ROOT') and torch.cuda.is_available(), 'Set AIRFOIL_DATA_ROOT and use a CUDA runtime for the Airfoil full-batch gate.')
class AirfoilCUDAFullBatchSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(os.environ['AIRFOIL_DATA_ROOT']).expanduser()
        with patch('urllib.request.urlopen', side_effect=AssertionError('network access is forbidden')) as network:
            train_airfoil.verify_airfoil_dataset(root)
            set_seed(PAPER_SEEDS[0])
            data = train_airfoil.load_airfoil(root)
            cls.input_batch, cls.target_batch = next(iter(data.train_loader))
        network.assert_not_called()
        if cls.input_batch.shape != (_afcuda_BATCH_SIZE, *RESOLUTION, 2):
            raise AssertionError(f'Unexpected Airfoil input batch: {cls.input_batch.shape}')
        if cls.target_batch.shape != (_afcuda_BATCH_SIZE, *RESOLUTION, 1):
            raise AssertionError(f'Unexpected Airfoil target batch: {cls.target_batch.shape}')

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.input_batch
        del cls.target_batch
        gc.collect()
        torch.cuda.empty_cache()

    def test_all_eleven_models_one_full_batch_optimizer_step(self) -> None:
        device = torch.device('cuda')
        expected_counts = {**EXPECTED_PINNED_PARAMETER_COUNTS, **EXPECTED_CAFE_PARAMETER_COUNTS}
        for model_name in _afcuda_MODEL_CHOICES:
            inputs = targets = built = model = optimizer = None
            prediction = loss = loss_function = gradients = None
            try:
                with self.subTest(model=model_name):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    inputs = train_airfoil._channel_first(self.input_batch, device)
                    targets = train_airfoil._channel_first(self.target_batch, device)
                    built = train_airfoil.build_model(model_name, device)
                    model = built.model
                    parameter_count = train_airfoil.count_model_params(model)
                    train_airfoil.validate_model_parameter_count(model_name, parameter_count)
                    self.assertEqual(parameter_count, expected_counts[model_name])
                    optimizer = torch.optim.AdamW(model.parameters(), lr=_afcuda_LEARNING_RATE, weight_decay=_afcuda_WEIGHT_DECAY)
                    loss_function = train_airfoil.LpLoss(d=2, p=2, reduction='sum')
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(inputs)
                    self.assertEqual(tuple(prediction.shape), (_afcuda_BATCH_SIZE, 1, *RESOLUTION))
                    self.assertTrue(torch.isfinite(prediction).all().item())
                    loss = loss_function(prediction, targets)
                    self.assertTrue(torch.isfinite(loss).all().item())
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    self.assertTrue(all((torch.isfinite(parameter).all().item() for parameter in model.parameters() if parameter.requires_grad)))
                    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 2 ** 20
                    print('AIRFOIL_CUDA_SMOKE ' + json.dumps({'model': model_name, 'parameter_count': parameter_count, 'output_shape': list(prediction.shape), 'loss_finite': True, 'gradients_finite': True, 'optimizer_step': True, 'peak_cuda_memory_mb': round(peak_memory_mb, 2), 'status': 'PASS'}, sort_keys=True))
            finally:
                del inputs
                del targets
                del prediction
                del loss
                del loss_function
                del optimizer
                del model
                del built
                del gradients
                gc.collect()
                torch.cuda.empty_cache()

# Migrated from tests/test_reacdiff_cuda_smoke.py
import contextlib







from experiments import train_reacdiff as train_module


from experiments.configs.reacdiff1d import BATCH_SIZE as _rdcuda_BATCH_SIZE, CANONICAL_SEED, LEARNING_RATE as _rdcuda_LEARNING_RATE, MODEL_CHOICES as _rdcuda_MODEL_CHOICES, ROLLOUT, WEIGHT_DECAY as _rdcuda_WEIGHT_DECAY

from experiments.train_reacdiff import build_model, load_reacdiff, rollout_reacdiff, validate_model_parameter_count, verify_reacdiff_dataset

@unittest.skipUnless(os.environ.get('RUN_REACDIFF_CUDA_SMOKE') == '1' and os.environ.get('REACDIFF_DATA_ROOT'), 'Set RUN_REACDIFF_CUDA_SMOKE=1 and REACDIFF_DATA_ROOT to opt in.')
class ReacDiffOfficialBatchCUDASmokeTests(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable.')
    def test_all_twelve_models_one_full_rollout_optimizer_step(self) -> None:
        data_root = Path(os.environ['REACDIFF_DATA_ROOT']).expanduser()
        verify_reacdiff_dataset(data_root)
        set_seed(CANONICAL_SEED)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('network')))
            stack.enter_context(patch('requests.get', side_effect=AssertionError('network')))
            stack.enter_context(patch.object(train_module, 'N_TRAIN', _rdcuda_BATCH_SIZE))
            stack.enter_context(patch.object(train_module, 'N_TEST', 1))
            data = load_reacdiff(data_root)
        x_batch, y_batch = next(iter(data.train_loader))
        self.assertEqual(x_batch.shape[0], _rdcuda_BATCH_SIZE)
        device = torch.device('cuda')
        for model_name in _rdcuda_MODEL_CHOICES:
            with self.subTest(model=model_name):
                torch.cuda.reset_peak_memory_stats()
                built = build_model(model_name, device)
                model = built.model
                count = train_module.count_model_params(model)
                validate_model_parameter_count(model_name, count)
                optimizer = torch.optim.AdamW(model.parameters(), lr=_rdcuda_LEARNING_RATE, weight_decay=_rdcuda_WEIGHT_DECAY)
                x = x_batch.to(device, non_blocking=True).float()
                y = y_batch.to(device, non_blocking=True).float()
                prediction = rollout_reacdiff(model, x, ROLLOUT, pushforward_detach=False)
                self.assertEqual(tuple(prediction.shape), tuple(y.shape))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss_fn = train_module.LpLoss(d=1, p=2, reduction='mean')
                loss = train_module.rollout_loss_lp(prediction, y, loss_fn)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all((torch.isfinite(gradient).all().item() for gradient in gradients)))
                optimizer.step()
                self.assertTrue(all((torch.isfinite(parameter).all().item() for parameter in model.parameters())))
                print(f'REACDIFF_CUDA_SMOKE {model_name} params={count} peak_bytes={torch.cuda.max_memory_allocated()}')
                del x, y, prediction, loss, optimizer, model, built, gradients
                gc.collect()
                torch.cuda.empty_cache()
