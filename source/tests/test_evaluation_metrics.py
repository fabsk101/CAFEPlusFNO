"""Regression tests for corrected paper relative-L2 evaluation."""

from __future__ import annotations

import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.evaluation import (
    DEFAULT_RELATIVE_L2_EPSILON,
    EVALUATION_PROTOCOL_VERSION,
    corrected_relative_l2,
    corrected_relative_l2_metadata,
    corrected_relative_l2_ratios,
    evaluate_corrected_relative_l2,
)


def _provided_prediction_rollout(
    _model: torch.nn.Module,
    prediction: torch.Tensor,
    _steps: int,
    pushforward_detach: bool = True,
) -> torch.Tensor:
    del pushforward_detach
    return prediction


def _evaluate_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
):
    loader = DataLoader(
        TensorDataset(prediction, target),
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_corrected_relative_l2(
        torch.nn.Identity(),
        loader,
        horizon=target.shape[1],
        rollout_fn=_provided_prediction_rollout,
        device="cpu",
        time_dim=1,
        channel_dim=2,
    )


class CorrectedRelativeL2DefinitionTests(unittest.TestCase):
    def test_two_hundred_samples_are_independent_of_batch_partition(self) -> None:
        target = torch.ones(200, 3, 1, 7)
        prediction = torch.zeros_like(target)
        for batch_size in (1, 8, 16, 32, 200):
            with self.subTest(batch_size=batch_size):
                result = _evaluate_tensors(prediction, target, batch_size)
                self.assertAlmostEqual(result.corrected_step_relative_l2, 1.0)
                self.assertAlmostEqual(
                    result.corrected_trajectory_relative_l2, 1.0
                )
                self.assertEqual(result.sample_count, 200)
                self.assertEqual(result.horizon, 3)

    def test_small_last_batch_has_equal_per_sample_weight(self) -> None:
        target = torch.ones(10, 2, 1, 4)
        prediction = target.clone()
        prediction[-2:] = 11.0
        split = _evaluate_tensors(prediction, target, batch_size=8)
        single = _evaluate_tensors(prediction, target, batch_size=10)
        # Eight ratios are zero and two are ten: arithmetic sample mean = 2.
        self.assertAlmostEqual(split.corrected_step_relative_l2, 2.0)
        self.assertAlmostEqual(split.corrected_trajectory_relative_l2, 2.0)
        self.assertAlmostEqual(
            split.corrected_step_relative_l2,
            single.corrected_step_relative_l2,
        )
        self.assertAlmostEqual(
            split.corrected_trajectory_relative_l2,
            single.corrected_trajectory_relative_l2,
        )

    def test_matches_independent_flattened_reference(self) -> None:
        prediction = torch.tensor(
            [
                [[[0.0, 2.0], [1.0, 4.0]], [[2.0, 0.0], [3.0, 1.0]]],
                [[[2.0, 1.0], [0.0, 2.0]], [[1.0, 3.0], [4.0, 0.0]]],
            ]
        )
        target = torch.tensor(
            [
                [[[1.0, 2.0], [1.0, 2.0]], [[2.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
            ]
        )
        expected_steps: list[float] = []
        expected_trajectories: list[float] = []
        for sample in range(target.shape[0]):
            for step in range(target.shape[1]):
                difference = (prediction[sample, step] - target[sample, step]).flatten()
                reference = target[sample, step].flatten()
                numerator = math.sqrt(float(sum(value * value for value in difference)))
                denominator = max(
                    math.sqrt(float(sum(value * value for value in reference))),
                    DEFAULT_RELATIVE_L2_EPSILON,
                )
                expected_steps.append(numerator / denominator)
            difference = (prediction[sample] - target[sample]).flatten()
            reference = target[sample].flatten()
            numerator = math.sqrt(float(sum(value * value for value in difference)))
            denominator = max(
                math.sqrt(float(sum(value * value for value in reference))),
                DEFAULT_RELATIVE_L2_EPSILON,
            )
            expected_trajectories.append(numerator / denominator)

        actual_step, actual_trajectory = corrected_relative_l2(
            prediction,
            target,
            time_dim=1,
            channel_dim=2,
        )
        self.assertAlmostEqual(
            actual_step, sum(expected_steps) / len(expected_steps), places=6
        )
        self.assertAlmostEqual(
            actual_trajectory,
            sum(expected_trajectories) / len(expected_trajectories),
            places=6,
        )

    def test_step_and_trajectory_norms_are_distinct(self) -> None:
        target = torch.tensor([[[[1.0]], [[10.0]]]])
        prediction = torch.tensor([[[[0.0]], [[10.0]]]])
        step, trajectory = corrected_relative_l2(
            prediction,
            target,
            time_dim=1,
            channel_dim=2,
        )
        self.assertAlmostEqual(step, 0.5)
        self.assertAlmostEqual(trajectory, 1.0 / math.sqrt(101.0), places=7)

    def test_1d_2d_multichannel_batch_one_and_axis_conversion(self) -> None:
        generators = (
            (torch.Size((1, 3, 2, 5)), 1, 2),
            (torch.Size((2, 4, 3, 5, 6)), 1, 2),
            # Explicit noncanonical [B,C,H,W,T] layout.
            (torch.Size((2, 3, 5, 6, 4)), 4, 1),
        )
        for shape, time_dim, channel_dim in generators:
            with self.subTest(shape=tuple(shape)):
                torch.manual_seed(123)
                target = torch.rand(shape) + 0.5
                prediction = target + torch.randn(shape) * 0.1
                steps, trajectories = corrected_relative_l2_ratios(
                    prediction,
                    target,
                    time_dim=time_dim,
                    channel_dim=channel_dim,
                )
                canonical_target = torch.movedim(
                    target, (time_dim, channel_dim), (1, 2)
                )
                self.assertEqual(
                    tuple(steps.shape),
                    tuple(canonical_target.shape[:2]),
                )
                self.assertEqual(tuple(trajectories.shape), (shape[0],))
                self.assertTrue(torch.isfinite(steps).all())
                self.assertTrue(torch.isfinite(trajectories).all())

        torch.manual_seed(456)
        canonical_target = torch.rand(2, 4, 3, 5, 6) + 0.5
        canonical_prediction = canonical_target + torch.randn_like(canonical_target)
        canonical = corrected_relative_l2_ratios(
            canonical_prediction,
            canonical_target,
            time_dim=1,
            channel_dim=2,
        )
        # Same values encoded explicitly as [B,C,H,W,T].
        noncanonical_target = canonical_target.permute(0, 2, 3, 4, 1)
        noncanonical_prediction = canonical_prediction.permute(0, 2, 3, 4, 1)
        converted = corrected_relative_l2_ratios(
            noncanonical_prediction,
            noncanonical_target,
            time_dim=4,
            channel_dim=1,
        )
        torch.testing.assert_close(converted[0], canonical[0])
        torch.testing.assert_close(converted[1], canonical[1])

    def test_zero_target_policy_and_nonfinite_rejection(self) -> None:
        target = torch.zeros(1, 1, 1, 2)
        prediction = torch.tensor([[[[3.0, 4.0]]]])
        step, trajectory = corrected_relative_l2(
            prediction,
            target,
            time_dim=1,
            channel_dim=2,
            epsilon=0.01,
        )
        self.assertAlmostEqual(step, 500.0)
        self.assertAlmostEqual(trajectory, 500.0)
        zero_step, zero_trajectory = corrected_relative_l2(
            target,
            target,
            time_dim=1,
            channel_dim=2,
        )
        self.assertEqual((zero_step, zero_trajectory), (0.0, 0.0))

        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad_value):
                bad_prediction = prediction.clone()
                bad_prediction[0, 0, 0, 0] = bad_value
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    corrected_relative_l2(
                        bad_prediction,
                        target,
                        time_dim=1,
                        channel_dim=2,
                    )
                bad_target = target.clone()
                bad_target[0, 0, 0, 0] = bad_value
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    corrected_relative_l2(
                        prediction,
                        bad_target,
                        time_dim=1,
                        channel_dim=2,
                    )
        with self.assertRaises(ValueError):
            corrected_relative_l2(
                prediction,
                target,
                time_dim=1,
                channel_dim=2,
                epsilon=0.0,
            )

    def test_protocol_has_a_nonlegacy_version(self) -> None:
        self.assertEqual(EVALUATION_PROTOCOL_VERSION, "corrected_relative_l2_v1")
        self.assertNotIn("legacy", EVALUATION_PROTOCOL_VERSION)
        metadata = corrected_relative_l2_metadata(
            sample_count={"test": 200},
            horizon=10,
            evaluation_space="physical_source_values_no_normalization",
        )
        self.assertEqual(metadata["evaluation_sample_count"], {"test": 200})
        self.assertEqual(metadata["evaluation_horizon"], 10)
        self.assertEqual(metadata["evaluation_tensor_layout"], "B,T,C,*spatial")
        self.assertEqual(
            metadata["evaluation_norm_axes"],
            {
                "corrected_step_relative_l2": "C,*spatial",
                "corrected_trajectory_relative_l2": "T,C,*spatial",
            },
        )


class _DropoutRollout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.projection = torch.nn.Conv1d(2, 1, kernel_size=1)
        self.register_buffer("calibration", torch.tensor([2.0]))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.dropout(values))


def _model_rollout(
    model: torch.nn.Module,
    values: torch.Tensor,
    steps: int,
    pushforward_detach: bool = True,
) -> torch.Tensor:
    predictions = [model(values).unsqueeze(1) for _ in range(steps)]
    result = torch.cat(predictions, dim=1)
    return result.detach() if pushforward_detach else result


class EvaluationStateIsolationTests(unittest.TestCase):
    @staticmethod
    def _training_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        values: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[float, list[torch.Tensor]]:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(values) - target).square().mean()
        loss.backward()
        gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        return float(loss.detach().item()), gradients

    def test_evaluation_preserves_training_update_mode_rng_and_buffers(self) -> None:
        torch.manual_seed(91)
        baseline = _DropoutRollout()
        evaluated = _DropoutRollout()
        evaluated.load_state_dict(baseline.state_dict(), strict=True)
        baseline.train()
        evaluated.train()
        baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
        evaluated_optimizer = torch.optim.AdamW(evaluated.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            evaluated_optimizer, T_max=3
        )
        scheduler_before = scheduler.state_dict().copy()
        inputs = torch.randn(6, 2, 5)
        training_target = torch.randn(6, 1, 5)
        evaluation_target = torch.randn(6, 2, 1, 5)
        evaluation_loader = DataLoader(
            TensorDataset(inputs, evaluation_target), batch_size=4, shuffle=False
        )
        parameters_before = {
            name: value.detach().clone()
            for name, value in evaluated.named_parameters()
        }
        buffers_before = {
            name: value.detach().clone() for name, value in evaluated.named_buffers()
        }

        torch.manual_seed(707)
        baseline_loss, baseline_gradients = self._training_step(
            baseline, baseline_optimizer, inputs, training_target
        )
        torch.manual_seed(707)
        evaluate_corrected_relative_l2(
            evaluated,
            evaluation_loader,
            horizon=2,
            rollout_fn=_model_rollout,
            device="cpu",
            time_dim=1,
            channel_dim=2,
        )
        self.assertTrue(evaluated.training)
        self.assertEqual(scheduler.state_dict(), scheduler_before)
        for name, parameter in evaluated.named_parameters():
            torch.testing.assert_close(parameter, parameters_before[name])
            self.assertIsNone(parameter.grad)
        for name, buffer in evaluated.named_buffers():
            torch.testing.assert_close(buffer, buffers_before[name])

        evaluated_loss, evaluated_gradients = self._training_step(
            evaluated, evaluated_optimizer, inputs, training_target
        )
        self.assertEqual(evaluated_loss, baseline_loss)
        for actual, expected in zip(evaluated_gradients, baseline_gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(evaluated.parameters(), baseline.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
