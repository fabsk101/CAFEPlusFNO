from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import torch

from ablations.checkpoints import restore_model_checkpoint
from ablations.contracts import ABLATION_CONDITIONS, INPUT32_CONDITION, conditions_for_variant
from ablations.models import build_model_from_kwargs
from ablations.tests.test_models import tiny_config


class CheckpointMatrixTests(unittest.TestCase):
    def test_all_42_supported_architectures_save_strict_reload_and_continue(self):
        from experiments.common.checkpoints import prepare_weights_only_checkpoint
        checked = 0
        for dim in (1, 2):
            shape = (2, 1, 7) if dim == 1 else (2, 1, 4, 5)
            dataset = 'burgers1d' if dim == 1 else 'darcy'
            for factorization in ('dense', 'cp', 'tt', 'tucker'):
                for condition in conditions_for_variant(factorization):
                    with self.subTest(dim=dim, factorization=factorization, condition=condition):
                        torch.manual_seed(42)
                        config = tiny_config(dim, factorization=factorization)
                        if condition == INPUT32_CONDITION:
                            config.update(rff_basis=10, cheb_basis=16, cafe_branch_dim=32)
                        built = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition=condition)
                        x, target = torch.randn(shape), torch.randn(shape)
                        optimizer = torch.optim.AdamW(built.model.parameters(), lr=0.001)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
                        output = built.model(x)
                        self.assertTrue(torch.isfinite(output).all())
                        torch.nn.functional.mse_loss(output, target).backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in built.model.parameters()))
                        optimizer.step()
                        scheduler.step()
                        payload = prepare_weights_only_checkpoint({
                            'base_configuration': config, 'model_configuration': built.configuration,
                            'ablation_variant': condition, 'condition': condition, 'dataset': dataset,
                            'model_variant': factorization, 'seed': 42,
                            'model_state_dict': built.model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 'epoch': 1, 'result_kind': 'diagnostic',
                        })
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / 'checkpoint.pt'
                            torch.save(payload, path)
                            restored, saved = restore_model_checkpoint(path, expected_condition=condition,
                                expected_dataset=dataset, expected_model_variant=factorization, expected_seed=42)
                            with self.assertRaisesRegex(ValueError, 'mismatch'):
                                restore_model_checkpoint(path, expected_condition=('full' if condition != 'full' else 'learnable_sigma'),
                                    expected_dataset=dataset, expected_model_variant=factorization)
                        candidate = restored.model
                        with torch.no_grad():
                            torch.testing.assert_close(built.model(x), candidate(x), rtol=0, atol=0)
                        other_optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001)
                        other_optimizer.load_state_dict(saved['optimizer_state_dict'])
                        other_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(other_optimizer, T_max=500)
                        other_scheduler.load_state_dict(saved['scheduler_state_dict'])
                        for model, opt, sch in ((built.model, optimizer, scheduler), (candidate, other_optimizer, other_scheduler)):
                            opt.zero_grad(set_to_none=True)
                            torch.nn.functional.mse_loss(model(x), target).backward()
                            opt.step()
                            sch.step()
                        for name, value in built.model.state_dict().items():
                            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
                        checked += 1
        self.assertEqual(checked, 42)

    def test_learnable_sigma_preserves_initial_state_rng_and_gaussian_basis(self):
        for dim in (1, 2):
            for factorization in ('dense', 'cp', 'tt', 'tucker'):
                with self.subTest(dim=dim, factorization=factorization):
                    config = tiny_config(dim, factorization=factorization)
                    torch.manual_seed(73)
                    full = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition='full').model
                    state = torch.get_rng_state().clone()
                    torch.manual_seed(73)
                    learned = build_model_from_kwargs(spatial_dim=dim, configuration=config, condition='learnable_sigma').model
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    for name, value in full.state_dict().items():
                        torch.testing.assert_close(value, learned.state_dict()[name], rtol=0, atol=0)
                    delta = set(dict(learned.named_parameters())) - set(dict(full.named_parameters()))
                    self.assertTrue(delta and all(name.endswith('log_sigma') for name in delta))
                    gaussian = {n: b.clone() for n, b in learned.named_buffers() if n.endswith('.G')}
                    x = torch.randn((1, 1, 7) if dim == 1 else (1, 1, 4, 5))
                    torch.testing.assert_close(full(x), learned(x), rtol=0, atol=0)
                    opt = torch.optim.AdamW(learned.parameters(), lr=0.001)
                    learned(x).square().mean().backward()
                    for name in delta:
                        gradient = dict(learned.named_parameters())[name].grad
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                    opt.step()
                    for name, value in gaussian.items():
                        torch.testing.assert_close(value, dict(learned.named_buffers())[name], rtol=0, atol=0)
