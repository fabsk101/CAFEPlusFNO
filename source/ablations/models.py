"""Ablation-aware CAFE encoders layered onto the repository's FNO code.

The Full and Learnable-sigma conditions instantiate the original model class
directly. Feature and branch ablations replace only CAFE mode encoders and the
affected kernel-head input projection. Spectral mixing, factor reconstruction,
lifting, residual/FFN blocks, projection, and model forward methods remain the
repository implementations.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import Tensor, nn

from .contracts import (
    ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS, INPUT32_CONDITION,
    validate_condition_variant,
)


class AblationModeEncoder(nn.Module):
    """CAFE encoder supporting feature removal and a true branch-free path."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int,
        branch_dim: int | None,
        sigma_init: float,
        learnable_sigma: bool,
        phase_scale: float,
        use_fourier: bool,
        use_chebyshev: bool,
        use_linear_branches: bool,
        gaussian: Tensor | None,
        normalized_coordinates: Callable[..., Tensor],
        chebyshev_embedding: Callable[[Tensor, int], Tensor],
    ) -> None:
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError("spatial_dim must be 1 or 2")
        if not use_fourier and not use_chebyshev:
            raise ValueError("At least one CAFE feature family must remain")
        if rff_basis <= 0 or cheb_basis <= 0 or num_branches <= 0:
            raise ValueError("source basis and branch counts must be positive")
        self.spatial_dim = spatial_dim
        self.source_rff_basis = int(rff_basis)
        self.source_cheb_basis = int(cheb_basis)
        self.rff_basis = int(rff_basis) if use_fourier else 0
        self.cheb_basis = int(cheb_basis) if use_chebyshev else 0
        self.use_fourier = bool(use_fourier)
        self.use_chebyshev = bool(use_chebyshev)
        self.use_linear_branches = bool(use_linear_branches)
        self.num_branches = int(num_branches) if use_linear_branches else 0
        self.phase_scale = float(phase_scale)
        self.embedding_dim = 2 * self.rff_basis + spatial_dim * self.cheb_basis
        self.branch_dim = (
            int(branch_dim) if use_linear_branches and branch_dim is not None
            else self.embedding_dim
        )
        self._normalized_coordinates = normalized_coordinates
        self._chebyshev_embedding = chebyshev_embedding

        if use_fourier:
            if gaussian is None or tuple(gaussian.shape) != (spatial_dim, rff_basis):
                raise ValueError("The preserved Gaussian basis has the wrong shape")
            self.register_buffer("G", gaussian.detach().clone(), persistent=True)
        else:
            self.register_buffer(
                "G", torch.empty(spatial_dim, 0, dtype=torch.float32), persistent=True
            )

        initial_log_sigma = torch.tensor(math.log(float(sigma_init)))
        if learnable_sigma:
            self.log_sigma = nn.Parameter(initial_log_sigma)
        else:
            self.register_buffer("log_sigma", initial_log_sigma, persistent=True)
        self.learnable_sigma = bool(learnable_sigma)

        self.branches = nn.ModuleList()
        if use_linear_branches:
            self.branches.extend(
                nn.Linear(self.embedding_dim, self.branch_dim, bias=True)
                for _ in range(num_branches)
            )
        self._fixed_feature_cache: dict[tuple[Any, ...], tuple[Tensor, Tensor | None]] = {}

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    @property
    def reference(self) -> Tensor:
        if self.use_linear_branches:
            return self.branches[0].weight
        return self.log_sigma

    def clear_feature_cache(self) -> None:
        self._fixed_feature_cache.clear()

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_feature_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn):  # type: ignore[override]
        self.clear_feature_cache()
        return super()._apply(fn)

    def _features_from_coordinates(self, coordinates: Tensor) -> Tensor:
        components: list[Tensor] = []
        if self.use_chebyshev:
            components.append(
                torch.cat(
                    [
                        self._chebyshev_embedding(coordinates[:, axis], self.cheb_basis)
                        for axis in range(self.spatial_dim)
                    ],
                    dim=-1,
                )
            )
        if self.use_fourier:
            bandwidth = self.sigma.to(coordinates) * self.G.to(coordinates)
            phase = self.phase_scale * (coordinates @ bandwidth)
            components.append(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1))
        embedding = torch.cat(components, dim=-1)
        if embedding.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"expected embedding dim {self.embedding_dim}, got {embedding.shape[-1]}"
            )
        return embedding

    def _encode(self, embedding: Tensor) -> Tensor:
        reference = self.reference
        embedding = embedding.to(device=reference.device, dtype=reference.dtype)
        if not self.use_linear_branches:
            # Identity replaces the entire branch/Hadamard stage. This keeps the
            # kernel path alive instead of multiplying it by a zero branch.
            return embedding
        product = self.branches[0](embedding)
        for branch in self.branches[1:]:
            product = product * branch(embedding)
        return product

    def build_embedding(
        self,
        spatial_shape: int | Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        shape = (
            (int(spatial_shape),)
            if isinstance(spatial_shape, int)
            else tuple(int(value) for value in spatial_shape)
        )
        if len(shape) != self.spatial_dim or any(value <= 0 for value in shape):
            raise ValueError(f"invalid spatial shape {shape}")
        key = (shape, device.type, device.index, dtype)
        cached = None if self.learnable_sigma else self._fixed_feature_cache.get(key)
        if cached is None:
            arg: int | Sequence[int] = shape[0] if self.spatial_dim == 1 else shape
            coordinates = self._normalized_coordinates(arg, device=device, dtype=dtype)
            coordinates = coordinates.reshape(-1, self.spatial_dim)
            embedding = self._features_from_coordinates(coordinates)
            if not self.learnable_sigma:
                self._fixed_feature_cache[key] = (coordinates, embedding)
        else:
            embedding = cached[1]
            assert embedding is not None
        return embedding

    def forward(self, spatial_shape: int | Sequence[int]) -> Tensor:
        reference = self.reference
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        embedding = self.build_embedding(
            spatial_shape, device=reference.device, dtype=feature_dtype
        )
        return self._encode(embedding)

    def forward_coordinates(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[-1] != self.spatial_dim:
            raise ValueError(
                f"coordinates must have shape [N, {self.spatial_dim}]"
            )
        reference = self.reference
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        coordinates = coordinates.to(device=reference.device, dtype=feature_dtype)
        return self._encode(self._features_from_coordinates(coordinates))

    def extra_repr(self) -> str:
        return (
            f"spatial_dim={self.spatial_dim}, rff_basis={self.rff_basis}, "
            f"cheb_basis={self.cheb_basis}, embedding_dim={self.embedding_dim}, "
            f"branch_dim={self.branch_dim}, linear_branches={self.use_linear_branches}, "
            f"learnable_sigma={self.learnable_sigma}"
        )


class AblationScalarFactorBlock(nn.Module):
    """Original scalar factor MLP driven by an ablation-aware 1-D encoder."""

    def __init__(
        self,
        *,
        output_dim: int,
        encoder: AblationModeEncoder,
        mlp: nn.Module,
        midpoint_coordinates: Callable[..., Tensor],
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.encoder = encoder
        self.mlp = mlp
        self._midpoint_coordinates = midpoint_coordinates

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    @property
    def branch_dim(self) -> int:
        return self.encoder.branch_dim

    @property
    def sigma(self) -> Tensor:
        return self.encoder.sigma

    def forward(self, axis_length: int) -> Tensor:
        if axis_length <= 0:
            raise ValueError("axis_length must be positive")
        reference = self.encoder.reference
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        coordinates = self._midpoint_coordinates(
            axis_length, device=reference.device, dtype=feature_dtype
        )
        output = self.mlp(self.encoder.forward_coordinates(coordinates))
        expected = (axis_length, self.output_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"factor block returned {tuple(output.shape)}, expected {expected}")
        return output


@dataclass(frozen=True)
class AblationBuild:
    model: nn.Module
    configuration: dict[str, Any]
    factorization: str
    rank: int
    condition: str


def _replace_mlp_input(mlp: nn.Module, input_dim: int) -> None:
    if int(mlp.input_dim) == input_dim:
        return
    old = mlp.linear_in
    replacement = nn.Linear(
        input_dim,
        old.out_features,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    nn.init.xavier_uniform_(replacement.weight)
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)
    mlp.linear_in = replacement
    mlp.input_dim = int(input_dim)


def _make_encoder(
    source: nn.Module,
    *,
    spatial_dim: int,
    condition: str,
    branch_dim: int | None,
    one_d: Any,
    two_d: Any,
) -> AblationModeEncoder:
    spec = ABLATION_CONDITIONS[condition]
    helpers = one_d if spatial_dim == 1 else two_d
    gaussian = source.G if spec.use_fourier else None
    rff_basis = int(source.rff_basis)
    cheb_basis = int(source.cheb_basis)
    if condition == INPUT32_CONDITION:
        # Preserve the base draw and coordinate/bandwidth conventions. Do not
        # draw another G or change the parent config shared by other conditions.
        rff_basis, cheb_basis = 8, 16 // spatial_dim
        if int(source.rff_basis) < rff_basis or int(source.cheb_basis) < cheb_basis:
            raise ValueError("INPUT32_SOURCE_BASIS_TOO_SMALL")
        if gaussian is None or tuple(gaussian.shape) != (spatial_dim, int(source.rff_basis)):
            raise ValueError("INPUT32_SOURCE_G_SHAPE_INVALID")
        gaussian = gaussian[:, :rff_basis]
    encoder = AblationModeEncoder(
        spatial_dim=spatial_dim,
        rff_basis=rff_basis,
        cheb_basis=cheb_basis,
        num_branches=int(source.num_branches),
        branch_dim=branch_dim,
        sigma_init=float(source.sigma.detach().cpu().item()),
        learnable_sigma=spec.learnable_sigma,
        phase_scale=float(source.phase_scale),
        use_fourier=spec.use_fourier,
        use_chebyshev=spec.use_chebyshev,
        use_linear_branches=spec.use_linear_branches,
        gaussian=gaussian,
        normalized_coordinates=helpers.normalized_rfft_coordinates,
        chebyshev_embedding=helpers.chebyshev_embedding,
    )
    if condition == INPUT32_CONDITION:
        encoder.source_rff_basis = int(source.rff_basis)
        encoder.source_cheb_basis = int(source.cheb_basis)
        if encoder.embedding_dim != 32 or encoder.branch_dim != 32 or len(encoder.branches):
            raise RuntimeError("INPUT32_ENCODER_CONTRACT_INVALID")
    return encoder


def _rewrite_generator(
    generator: nn.Module,
    *,
    spatial_dim: int,
    condition: str,
    dense_branch_was_implicit: bool,
    one_d: Any,
    two_d: Any,
) -> None:
    spec = ABLATION_CONDITIONS[condition]
    validate_condition_variant(condition, str(generator.factorization))
    if generator.factorization == "dense":
        source = generator.encoder
        active_embedding = (
            (2 * int(source.rff_basis) if spec.use_fourier else 0)
            + (spatial_dim * int(source.cheb_basis) if spec.use_chebyshev else 0)
        )
        if condition == INPUT32_CONDITION:
            active_embedding = 32
        requested_branch_dim = (
            active_embedding
            if (not spec.use_linear_branches or dense_branch_was_implicit)
            else int(source.branch_dim)
        )
        encoder = _make_encoder(
            source,
            spatial_dim=spatial_dim,
            condition=condition,
            branch_dim=requested_branch_dim,
            one_d=one_d,
            two_d=two_d,
        )
        generator.encoder = encoder
        _replace_mlp_input(generator.head.joint_mlp, encoder.branch_dim)
        return

    replacements: dict[str, nn.Module] = {}
    for role, source_block in generator.factor_blocks.items():
        source_encoder = source_block.encoder
        requested_branch_dim = (
            None if not spec.use_linear_branches else int(source_encoder.branch_dim)
        )
        encoder = _make_encoder(
            source_encoder,
            spatial_dim=1,
            condition=condition,
            branch_dim=requested_branch_dim,
            one_d=one_d,
            two_d=two_d,
        )
        mlp = source_block.mlp
        _replace_mlp_input(mlp, encoder.branch_dim)
        replacements[role] = AblationScalarFactorBlock(
            output_dim=int(source_block.output_dim),
            encoder=encoder,
            mlp=mlp,
            midpoint_coordinates=one_d.midpoint_axis_coordinates,
        )
    generator.factor_blocks = nn.ModuleDict(replacements)


def build_model_from_kwargs(
    *,
    spatial_dim: int,
    configuration: dict[str, Any],
    condition: str,
    device: torch.device | str = "cpu",
) -> AblationBuild:
    """Build one model without consulting train-module globals or factories."""

    if condition not in ABLATION_CONDITIONS:
        raise KeyError(f"Unknown ablation condition: {condition}")
    if spatial_dim not in (1, 2):
        raise ValueError("spatial_dim must be 1 or 2")
    # No factorized model, output directory, or new random draw is constructed
    # for an unsupported Dense-only request.
    validate_condition_variant(
        condition, str(configuration.get("factorization", "dense")).strip().lower()
    )
    one_d = importlib.import_module("models.Cafe_Plus_FNO1D")
    two_d = importlib.import_module("models.Cafe_Plus_FNO2D")
    model_class = one_d.CAFEPlusFNO1D if spatial_dim == 1 else two_d.CAFEPlusFNO2D
    kwargs = dict(configuration)
    kwargs["learnable_sigma"] = condition == "learnable_sigma"
    model = model_class(**kwargs)

    if condition not in {"full", "learnable_sigma"}:
        implicit_branch = configuration.get("cafe_branch_dim") is None
        for layer in model.operator_layers:
            _rewrite_generator(
                layer.spectral.kernel_generator,
                spatial_dim=spatial_dim,
                condition=condition,
                dense_branch_was_implicit=implicit_branch,
                one_d=one_d,
                two_d=two_d,
            )
        first_generator = model.operator_layers[0].spectral.kernel_generator
        model.embedding_dim = int(first_generator.embedding_dim)
        model.branch_dim = int(first_generator.branch_dim)
        if model.factorization == "dense":
            model.rff_basis = int(first_generator.encoder.rff_basis)
            model.cheb_basis = int(first_generator.encoder.cheb_basis)
        else:
            model.factor_embedding_dim = int(first_generator.embedding_dim)

    model.ablation_condition = condition
    model.ablation_spec = ABLATION_CONDITIONS[condition].__dict__.copy()
    model.to(device)
    resolved = dict(configuration)
    resolved["learnable_sigma"] = condition == "learnable_sigma"
    resolved["ablation_condition"] = condition
    resolved["active_fourier_features"] = ABLATION_CONDITIONS[condition].use_fourier
    resolved["active_chebyshev_features"] = ABLATION_CONDITIONS[condition].use_chebyshev
    resolved["active_linear_branches"] = ABLATION_CONDITIONS[condition].use_linear_branches
    resolved["active_hadamard_product"] = ABLATION_CONDITIONS[condition].use_hadamard_product
    if condition == INPUT32_CONDITION:
        encoder = model.operator_layers[0].spectral.kernel_generator.encoder
        resolved["direct_input32_basis"] = {
            "version": "direct_input32_v1",
            "source_rff_basis": int(encoder.source_rff_basis),
            "source_cheb_basis_per_axis": int(encoder.source_cheb_basis),
            "rff_basis": 8,
            "cheb_basis_per_axis": 16 // spatial_dim,
            "rff_feature_dim": 16,
            "chebyshev_feature_dim": 16,
            "embedding_dim": 32,
            "kernel_mlp_input_dim": 32,
            "gaussian_selection": "base_G_columns_0_through_7_no_redraw",
            "feature_order": "chebyshev_axis_order_then_cos_then_sin",
        }
    return AblationBuild(
        model=model,
        configuration=resolved,
        factorization=str(model.factorization),
        rank=int(model.rank),
        condition=condition,
    )


def build_ablation_model(
    dataset: str,
    model_variant: str,
    condition: str,
    *,
    device: torch.device | str = "cpu",
) -> AblationBuild:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}")
    if model_variant not in MODEL_VARIANTS:
        raise KeyError(f"Unknown model variant: {model_variant}")
    from .configs import model_configuration
    configuration = model_configuration(dataset, model_variant)
    return build_model_from_kwargs(
        spatial_dim=DATASETS[dataset].spatial_dim,
        configuration=configuration,
        condition=condition,
        device=device,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def describe_model(model: nn.Module) -> dict[str, Any]:
    generators = [layer.spectral.kernel_generator for layer in model.operator_layers]
    first = generators[0]
    encoders: list[nn.Module] = []
    for generator in generators:
        if generator.factorization == "dense":
            encoders.append(generator.encoder)
        else:
            encoders.extend(block.encoder for block in generator.factor_blocks.values())
    linear_branch_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".branches." in name
    )
    sigma_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("log_sigma")
    )
    gaussian_buffer_values = sum(
        buffer.numel()
        for name, buffer in model.named_buffers()
        if name.endswith(".G") or name == "G"
    )
    return {
        "parameter_count": count_parameters(model),
        "rff_feature_dim_per_encoder": 2 * int(encoders[0].rff_basis),
        "chebyshev_feature_dim_per_encoder": int(encoders[0].embedding_dim) - 2 * int(encoders[0].rff_basis),
        "linear_branch_count_per_encoder": int(encoders[0].num_branches),
        "hadamard_combination": bool(encoders[0].num_branches),
        "kernel_mlp_input_dim": int(first.branch_dim),
        "encoder_count": len(encoders),
        "embedding_dim": int(first.embedding_dim),
        "branch_output_dim": int(first.branch_dim),
        "linear_branch_parameters": linear_branch_parameters,
        "sigma_trainable_parameter_count": sigma_parameter_count,
        "gaussian_basis_buffer_values": gaussian_buffer_values,
        "gaussian_basis_trainable": any(
            name.endswith(".G") or name == "G" for name, _ in model.named_parameters()
        ),
        "factorization": str(model.factorization),
        "rank": int(model.rank),
    }


@contextlib.contextmanager
def preserved_rng_state() -> Iterator[None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
