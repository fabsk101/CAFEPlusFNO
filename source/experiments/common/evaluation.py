"""Corrected, batch-partition-independent paper evaluation metrics.

The three rollout benchmarks using this module produce tensors with explicit
``[batch, time, physical_channel, *spatial]`` semantics.  Training continues
to use the loss functions from the pinned SirenFNO implementation; the
helpers here are evaluation-only and deliberately do not reuse ``LpLoss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable

import torch


EVALUATION_PROTOCOL_VERSION = "corrected_relative_l2_v1"
DEFAULT_RELATIVE_L2_EPSILON = 1e-12
CANONICAL_TRAJECTORY_LAYOUT = "B,T,C,*spatial"


def corrected_relative_l2_metadata(
    *,
    sample_count: dict[str, int],
    horizon: int,
    evaluation_space: str,
    epsilon: float = DEFAULT_RELATIVE_L2_EPSILON,
) -> dict[str, object]:
    """Build the shared serializable metadata contract for corrected metrics."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not sample_count or any(value <= 0 for value in sample_count.values()):
        raise ValueError("Every evaluation sample count must be positive.")
    if not evaluation_space:
        raise ValueError("evaluation_space must be explicit.")
    if not isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")
    return {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_implementation": "experiments/common/evaluation.py",
        "evaluation_space": evaluation_space,
        "evaluation_tensor_layout": CANONICAL_TRAJECTORY_LAYOUT,
        "evaluation_epsilon": float(epsilon),
        "evaluation_zero_reference_policy": "max(target_l2, epsilon)",
        "evaluation_norm_axes": {
            "corrected_step_relative_l2": "C,*spatial",
            "corrected_trajectory_relative_l2": "T,C,*spatial",
        },
        "evaluation_aggregation": {
            "corrected_step_relative_l2": "arithmetic_mean_over_samples_and_time",
            "corrected_trajectory_relative_l2": "arithmetic_mean_over_samples",
        },
        "evaluation_horizon": int(horizon),
        "evaluation_sample_count": {
            str(split): int(count) for split, count in sample_count.items()
        },
        "evaluation_metrics": [
            "corrected_step_relative_l2",
            "corrected_trajectory_relative_l2",
        ],
    }


@dataclass(frozen=True)
class CorrectedRelativeL2:
    """A complete result from one loader evaluation."""

    corrected_step_relative_l2: float
    corrected_trajectory_relative_l2: float
    sample_count: int
    horizon: int
    epsilon: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and strictly positive.")
        if not isfinite(self.corrected_step_relative_l2):
            raise ValueError("Corrected step relative L2 is not finite.")
        if not isfinite(self.corrected_trajectory_relative_l2):
            raise ValueError("Corrected trajectory relative L2 is not finite.")


def _canonicalize_trajectory(
    tensor: torch.Tensor,
    *,
    time_dim: int,
    channel_dim: int,
    label: str,
) -> torch.Tensor:
    """Move declared semantic axes to ``[B,T,C,*spatial]`` positions."""

    if not torch.is_tensor(tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    if tensor.ndim < 3:
        raise ValueError(
            f"{label} must contain batch, time, and channel axes; "
            f"received shape {tuple(tensor.shape)}."
        )
    ndim = tensor.ndim
    resolved_time = time_dim % ndim
    resolved_channel = channel_dim % ndim
    if resolved_time == 0 or resolved_channel == 0:
        raise ValueError("The batch axis is fixed at dimension 0.")
    if resolved_time == resolved_channel:
        raise ValueError("time_dim and channel_dim must identify different axes.")
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise TypeError(f"{label} must have floating or complex dtype.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains NaN or infinity.")
    return torch.movedim(
        tensor,
        (resolved_time, resolved_channel),
        (1, 2),
    )


def corrected_relative_l2_ratios(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    time_dim: int,
    channel_dim: int,
    epsilon: float = DEFAULT_RELATIVE_L2_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-step ``[B,T]`` and per-trajectory ``[B]`` ratios.

    For each sample/time pair the step norm jointly reduces the physical
    channel axis and every spatial axis.  The trajectory norm additionally
    reduces time.  A zero reference therefore uses ``max(norm, epsilon)``;
    two identical zero tensors score zero, while a nonzero prediction against
    a zero target is scaled by ``epsilon``.
    """

    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must both be torch.Tensor values.")
    if not isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            "prediction and target must have identical shapes; received "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )

    canonical_prediction = _canonicalize_trajectory(
        prediction,
        time_dim=time_dim,
        channel_dim=channel_dim,
        label="prediction",
    )
    canonical_target = _canonicalize_trajectory(
        target,
        time_dim=time_dim,
        channel_dim=channel_dim,
        label="target",
    )
    batch_size, horizon = canonical_prediction.shape[:2]
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("Batch and time dimensions must be nonempty.")

    step_difference = (canonical_prediction - canonical_target).reshape(
        batch_size, horizon, -1
    )
    step_reference = canonical_target.reshape(batch_size, horizon, -1)
    step_numerator = torch.linalg.vector_norm(step_difference, ord=2, dim=-1)
    step_denominator = torch.linalg.vector_norm(
        step_reference, ord=2, dim=-1
    ).clamp_min(epsilon)
    step_ratios = step_numerator / step_denominator

    trajectory_difference = step_difference.reshape(batch_size, -1)
    trajectory_reference = step_reference.reshape(batch_size, -1)
    trajectory_numerator = torch.linalg.vector_norm(
        trajectory_difference, ord=2, dim=-1
    )
    trajectory_denominator = torch.linalg.vector_norm(
        trajectory_reference, ord=2, dim=-1
    ).clamp_min(epsilon)
    trajectory_ratios = trajectory_numerator / trajectory_denominator

    if not bool(torch.isfinite(step_ratios).all().item()):
        raise ValueError("Corrected step relative L2 produced NaN or infinity.")
    if not bool(torch.isfinite(trajectory_ratios).all().item()):
        raise ValueError(
            "Corrected trajectory relative L2 produced NaN or infinity."
        )
    return step_ratios, trajectory_ratios


def corrected_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    time_dim: int,
    channel_dim: int,
    epsilon: float = DEFAULT_RELATIVE_L2_EPSILON,
) -> tuple[float, float]:
    """Compute corrected arithmetic means for one in-memory batch."""

    step_ratios, trajectory_ratios = corrected_relative_l2_ratios(
        prediction,
        target,
        time_dim=time_dim,
        channel_dim=channel_dim,
        epsilon=epsilon,
    )
    return (
        float(step_ratios.to(dtype=torch.float64).mean().item()),
        float(trajectory_ratios.to(dtype=torch.float64).mean().item()),
    )


def _snapshot_module_modes(model: torch.nn.Module) -> tuple[tuple[torch.nn.Module, bool], ...]:
    return tuple((module, module.training) for module in model.modules())


def _restore_module_modes(
    states: tuple[tuple[torch.nn.Module, bool], ...],
) -> None:
    # Direct assignment preserves mixed child-module modes exactly. Calling
    # train() recursively here would overwrite those child-specific states.
    for module, was_training in states:
        module.training = was_training


def evaluate_corrected_relative_l2(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    horizon: int,
    rollout_fn: Callable[
        [torch.nn.Module, torch.Tensor, int, bool], torch.Tensor
    ],
    device: torch.device | str,
    time_dim: int,
    channel_dim: int,
    epsilon: float = DEFAULT_RELATIVE_L2_EPSILON,
) -> CorrectedRelativeL2:
    """Evaluate a rollout loader without changing training state or RNG.

    Ratios are accumulated per sample (and per sample/time for the step
    metric), so incomplete final batches receive exactly the same weight as
    all other samples.  Loader iteration and even a stochastic custom rollout
    cannot perturb the caller's CPU/CUDA RNG streams.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    resolved_device = torch.device(device)
    mode_states = _snapshot_module_modes(model)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    total_step_ratio = 0.0
    total_trajectory_ratio = 0.0
    sample_count = 0
    step_count = 0

    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise TypeError(
                        "Evaluation loader must yield (input, target) pairs."
                    )
                input_batch, target_batch = batch
                input_batch = input_batch.to(
                    resolved_device, non_blocking=True
                ).float()
                target_batch = target_batch.to(
                    resolved_device, non_blocking=True
                ).float()
                if input_batch.shape[0] != target_batch.shape[0]:
                    raise ValueError(
                        "Evaluation input and target batch sizes differ."
                    )
                prediction = rollout_fn(
                    model,
                    input_batch,
                    horizon,
                    pushforward_detach=True,
                )
                step_ratios, trajectory_ratios = corrected_relative_l2_ratios(
                    prediction,
                    target_batch,
                    time_dim=time_dim,
                    channel_dim=channel_dim,
                    epsilon=epsilon,
                )
                if step_ratios.shape[1] != horizon:
                    raise ValueError(
                        "Prediction/target time dimension does not match the "
                        f"configured horizon {horizon}."
                    )
                batch_samples = int(trajectory_ratios.numel())
                total_step_ratio += float(
                    step_ratios.detach().to("cpu", dtype=torch.float64).sum().item()
                )
                total_trajectory_ratio += float(
                    trajectory_ratios.detach()
                    .to("cpu", dtype=torch.float64)
                    .sum()
                    .item()
                )
                sample_count += batch_samples
                step_count += int(step_ratios.numel())
    finally:
        _restore_module_modes(mode_states)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if sample_count == 0:
        raise ValueError("Evaluation loader produced no samples.")
    expected_step_count = sample_count * horizon
    if step_count != expected_step_count:
        raise RuntimeError(
            "Evaluation accumulated an inconsistent number of sample/time pairs."
        )
    return CorrectedRelativeL2(
        corrected_step_relative_l2=total_step_ratio / step_count,
        corrected_trajectory_relative_l2=(
            total_trajectory_ratio / sample_count
        ),
        sample_count=sample_count,
        horizon=horizon,
        epsilon=float(epsilon),
    )
