"""One-dimensional CAFE+FNO with dense and tensor-factorized kernels.

``CAFE+FNO`` uses a CAFE+ mode encoder to parameterize a full-frequency
Fourier kernel.  This module provides four kernel parameterizations:

``dense``
    The canonical CAFE+FNO kernel.  One joint MLP emits the real and imaginary
    parts of every ``C_out x C_in`` channel-mixing matrix.
``cp`` / ``tt`` / ``tucker``
    SirenFNO-parity functional tensor decompositions.  Independent scalar-axis
    compact CAFE+ blocks generate input-channel, output-channel, and frequency
    factors for the real and imaginary kernels.  Their embedding, branch, and
    hidden widths are factor-specific rather than inherited from dense. TT is ordered
    ``C_in -> frequency -> C_out``; Tucker additionally learns independent
    real/imaginary cores.

With a sufficiently small rank, the factorized variants can reduce the number
of learned kernel-generator parameters; a large rank can instead increase it.
They deliberately reconstruct the full dense complex kernel before spectral
multiplication, so this implementation does *not* claim reduced FFT-layer
FLOPs or peak kernel memory.  On factorized paths, every tensor axis is sampled
independently with scalar midpoint coordinates, matching SirenFNO's functional
factor construction.

The default ``factorization='dense'`` path preserves the module hierarchy and
state-dict keys of the canonical implementation, including
``lifting``, ``operator_layers.*.spectral.kernel_generator.encoder``,
``operator_layers.*.spectral.kernel_generator.head.joint_mlp``, ``ffn``, and
``projection``.

Only PyTorch and the Python standard library are required.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__version__ = "0.5.1"

MODEL_NAME = "CAFE+FNO"
MODEL_REGISTRY_KEY = "cafe_plus_fno"

InputLayout = Literal["auto", "channels_first", "channels_last"]
OutputLayout = Literal["match_input", "channels_first", "channels_last"]
Factorization = Literal["dense", "cp", "tt", "tucker"]
PaddingLike = Union[int, Sequence[int]]

_FACTORIZATIONS = frozenset({"dense", "cp", "tt", "tucker"})


def _canonical_kernel_mlp_type(name: str) -> str:
    """Normalize the legacy API token used by the unchanged dense head."""

    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "joint": "joint",
        "one": "joint",
        "single": "joint",
        "single_mlp": "joint",
        "one_mlp": "joint",
        "shared": "joint",
    }
    if normalized not in aliases:
        raise ValueError(
            "use kernel_mlp_type='joint'; dense uses the canonical joint head "
            "and factorized variants use independent per-factor MLPs"
        )
    return aliases[normalized]


def _canonical_factorization(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("factorization must be a string")
    normalized = name.strip().lower()
    if normalized not in _FACTORIZATIONS:
        choices = ", ".join(sorted(_FACTORIZATIONS))
        raise ValueError(f"factorization must be one of: {choices}")
    return normalized


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _make_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError("activation must be one of: gelu, relu, silu")


def _normalize_padding(padding: PaddingLike) -> Tuple[int]:
    if isinstance(padding, bool):
        raise TypeError("padding must be a non-negative integer or length-1 sequence")
    if isinstance(padding, Integral):
        values = (int(padding),)
    else:
        try:
            values = tuple(int(value) for value in padding)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "padding must be a non-negative integer or length-1 sequence"
            ) from error
    if len(values) != 1:
        raise ValueError(f"1-D padding must contain one value, got {values}")
    if values[0] < 0:
        raise ValueError("padding must be non-negative")
    return values


def chebyshev_embedding(x: Tensor, n_basis: int) -> Tensor:
    """Evaluate ``[T_0(x), ..., T_{n_basis-1}(x)]`` by recurrence."""

    n_basis = _positive_int("n_basis", n_basis)
    x = x.clamp(-1.0, 1.0)
    t0 = torch.ones_like(x)
    if n_basis == 1:
        return t0.unsqueeze(-1)

    t1 = x
    values = [t0, t1]
    for _ in range(2, n_basis):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def normalized_rfft_coordinates(
    full_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return native one-sided rFFT coordinates normalized to ``[0, 1]``."""

    full_length = _positive_int("full_length", full_length)
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a real floating-point dtype")
    return (
        2.0
        * torch.fft.rfftfreq(full_length, d=1.0, device=device, dtype=dtype)
    ).unsqueeze(-1)


def midpoint_axis_coordinates(
    axis_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return SirenFNO scalar-axis samples ``((i + .5) / N) * 2 - 1``."""

    axis_length = _positive_int("axis_length", axis_length)
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a real floating-point dtype")
    coordinate = torch.arange(axis_length, device=device, dtype=dtype) + 0.5
    return (2.0 * coordinate / float(axis_length) - 1.0).unsqueeze(-1)


def enforce_rfft1_kernel_hermitian(kernel_half: Tensor, full_length: int) -> Tensor:
    """Make the DC and, when present, Nyquist kernel entries real-valued."""

    if not torch.is_complex(kernel_half):
        raise TypeError("kernel_half must be complex")
    if kernel_half.ndim != 3:
        raise ValueError("kernel_half must have shape [C_out, C_in, L_f]")
    full_length = _positive_int("full_length", full_length)
    expected_half_length = full_length // 2 + 1
    if kernel_half.shape[-1] != expected_half_length:
        raise ValueError(
            f"expected L_f={expected_half_length} for full_length={full_length}, "
            f"got {kernel_half.shape[-1]}"
        )

    real = kernel_half.real
    imag = kernel_half.imag.clone()
    imag[..., 0] = 0.0
    if full_length % 2 == 0:
        imag[..., -1] = 0.0
    return torch.complex(real, imag)


class CAFEPlusModeEncoder(nn.Module):
    """Encode 1-D Fourier-mode coordinates with CAFE+.

    The raw embedding concatenates ``cheb_basis`` Chebyshev values and
    ``2 * rff_basis`` random Fourier features.  Independent linear branches
    project the embedding to ``branch_dim`` and their outputs are multiplied
    elementwise.  The fixed persistent Gaussian basis is sampled once from the
    global PyTorch RNG during construction.
    """

    def __init__(
        self,
        *,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int = 2,
        branch_dim: Optional[int] = None,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        phase_scale: float = math.pi,
    ) -> None:
        super().__init__()
        self.spatial_dim = 1
        self.rff_basis = _positive_int("rff_basis", rff_basis)
        self.cheb_basis = _positive_int("cheb_basis", cheb_basis)
        self.num_branches = _positive_int("num_branches", num_branches)
        if not isinstance(sigma_init, (int, float)) or isinstance(sigma_init, bool):
            raise TypeError("sigma_init must be a positive real number")
        if not math.isfinite(float(sigma_init)) or float(sigma_init) <= 0.0:
            raise ValueError("sigma_init must be a finite positive number")

        self.embedding_dim = 2 * self.rff_basis + self.cheb_basis
        self.branch_dim = (
            self.embedding_dim
            if branch_dim is None
            else _positive_int("branch_dim", branch_dim)
        )
        self.phase_scale = float(phase_scale)
        if not math.isfinite(self.phase_scale):
            raise ValueError("phase_scale must be finite")
        gaussian = torch.randn(
            1,
            self.rff_basis,
            dtype=torch.float32,
        )
        self.register_buffer("G", gaussian, persistent=True)

        initial_log_sigma = torch.tensor(math.log(float(sigma_init)))
        if learnable_sigma:
            self.log_sigma = nn.Parameter(initial_log_sigma)
        else:
            self.register_buffer("log_sigma", initial_log_sigma, persistent=True)
        self.learnable_sigma = bool(learnable_sigma)

        self.branches = nn.ModuleList(
            [
                nn.Linear(self.embedding_dim, self.branch_dim, bias=True)
                for _ in range(self.num_branches)
            ]
        )
        self._fixed_feature_cache: Dict[
            Tuple[int, str, Optional[int], torch.dtype], Tuple[Tensor, Tensor]
        ] = {}

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    def clear_feature_cache(self) -> None:
        self._fixed_feature_cache.clear()

    def _apply(self, fn):  # type: ignore[override]
        self.clear_feature_cache()
        return super()._apply(fn)

    def _fixed_features(
        self,
        full_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        full_length = _positive_int("full_length", full_length)
        key = (full_length, device.type, device.index, dtype)
        cached = self._fixed_feature_cache.get(key)
        if cached is not None:
            return cached

        coordinates = normalized_rfft_coordinates(
            full_length, device=device, dtype=dtype
        )
        chebyshev = chebyshev_embedding(
            coordinates[:, 0], self.cheb_basis
        )
        self._fixed_feature_cache[key] = (coordinates, chebyshev)
        return coordinates, chebyshev

    def build_embedding(
        self,
        full_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        coordinates, chebyshev = self._fixed_features(
            full_length, device=device, dtype=dtype
        )
        bandwidth = self.sigma.to(device=device, dtype=dtype) * self.G.to(
            device=device, dtype=dtype
        )
        phase = self.phase_scale * (coordinates @ bandwidth)
        rff = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        embedding = torch.cat((chebyshev, rff), dim=-1)
        if embedding.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"expected embedding dim {self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )
        return embedding

    def forward(self, full_length: int) -> Tensor:
        reference = self.branches[0].weight
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        embedding = self.build_embedding(
            full_length,
            device=reference.device,
            dtype=feature_dtype,
        )
        embedding = embedding.to(
            device=self.branches[0].weight.device,
            dtype=self.branches[0].weight.dtype,
        )
        product = self.branches[0](embedding)
        for branch in self.branches[1:]:
            product = product * branch(embedding)
        return product

    def forward_coordinates(self, coordinates: Tensor) -> Tensor:
        """Encode explicit scalar coordinates without changing the dense path."""

        if coordinates.ndim != 2 or coordinates.shape[-1] != 1:
            raise ValueError("coordinates must have shape [N, 1]")
        reference = self.branches[0].weight
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        coordinates = coordinates.to(device=reference.device, dtype=feature_dtype)
        chebyshev = chebyshev_embedding(coordinates[:, 0], self.cheb_basis)
        bandwidth = self.sigma.to(
            device=reference.device, dtype=feature_dtype
        ) * self.G.to(device=reference.device, dtype=feature_dtype)
        phase = self.phase_scale * (coordinates @ bandwidth)
        rff = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        embedding = torch.cat((chebyshev, rff), dim=-1).to(dtype=reference.dtype)
        product = self.branches[0](embedding)
        for branch in self.branches[1:]:
            product = product * branch(embedding)
        return product

    def extra_repr(self) -> str:
        return (
            f"rff_basis={self.rff_basis}, cheb_basis={self.cheb_basis}, "
            f"embedding_dim={self.embedding_dim}, branch_dim={self.branch_dim}, "
            f"num_branches={self.num_branches}, "
            f"learnable_sigma={self.learnable_sigma}"
        )


class KernelMLP(nn.Module):
    """One-hidden-layer MLP used by the joint complex kernel head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        output_init_std: float = 1e-3,
        output_init_mode: str = "normal",
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.output_dim = _positive_int("output_dim", output_dim)
        self.output_init_mode = str(output_init_mode).strip().lower()
        if self.output_init_mode not in {"normal", "xavier"}:
            raise ValueError("output_init_mode must be 'normal' or 'xavier'")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout must be a real number in [0, 1)")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            not isinstance(output_init_std, (int, float))
            or isinstance(output_init_std, bool)
        ):
            raise TypeError("output_init_std must be a positive real number")
        if (
            not math.isfinite(float(output_init_std))
            or float(output_init_std) <= 0.0
        ):
            raise ValueError("output_init_std must be a finite positive number")

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activation = _make_activation(activation)
        self.dropout = (
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.zeros_(self.linear_in.bias)
        if self.output_init_mode == "normal":
            nn.init.normal_(
                self.linear_out.weight, mean=0.0, std=float(output_init_std)
            )
        else:
            nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"KernelMLP expected input dim {self.input_dim}, got {x.shape[-1]}"
            )
        return self.linear_out(self.dropout(self.activation(self.linear_in(x))))


class JointRealImagKernelHead(nn.Module):
    """Emit real and imaginary values from one joint MLP."""

    num_mlps: int = 1

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        matrix_dim: int,
        activation: str,
        dropout: float,
        output_init_std: float,
    ) -> None:
        super().__init__()
        self.matrix_dim = _positive_int("matrix_dim", matrix_dim)
        self.joint_mlp = KernelMLP(
            input_dim,
            hidden_dim,
            2 * self.matrix_dim,
            activation=activation,
            dropout=dropout,
            output_init_std=output_init_std,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.joint_mlp(x)
        return output.split(self.matrix_dim, dim=-1)


class CAFEPlusScalarFactorBlock(nn.Module):
    """Generate one ``[axis_length, out_dim]`` functional tensor factor.

    Each compact block owns its Gaussian basis, bandwidth, multiplicative
    CAFE+ branches, and Xavier-output MLP. Coordinates follow the exact
    midpoint sampling convention used by official SirenFNO scalar-axis blocks.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int,
        branch_dim: Optional[int],
        hidden_dim: int,
        activation: str,
        dropout: float,
        sigma_init: float,
        learnable_sigma: bool,
        output_init_std: float,
        output_init_mode: str = "xavier",
    ) -> None:
        super().__init__()
        self.output_dim = _positive_int("output_dim", output_dim)
        self.output_init_mode = str(output_init_mode).strip().lower()
        if self.output_init_mode not in {"normal", "xavier"}:
            raise ValueError("output_init_mode must be 'normal' or 'xavier'")
        self.encoder = CAFEPlusModeEncoder(
            rff_basis=rff_basis,
            cheb_basis=cheb_basis,
            num_branches=num_branches,
            branch_dim=branch_dim,
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma,
        )
        self.mlp = KernelMLP(
            self.encoder.branch_dim,
            int(hidden_dim),
            self.output_dim,
            activation=activation,
            dropout=float(dropout),
            output_init_std=float(output_init_std),
            output_init_mode=output_init_mode,
        )

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
        axis_length = _positive_int("axis_length", axis_length)
        reference = self.encoder.branches[0].weight
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        coordinates = midpoint_axis_coordinates(
            axis_length, device=reference.device, dtype=feature_dtype
        )
        output = self.mlp(self.encoder.forward_coordinates(coordinates))
        expected = (axis_length, self.output_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"factor block returned {tuple(output.shape)}, expected {expected}"
            )
        return output


def _cp_reconstruct_1d(
    input_factor: Tensor, output_factor: Tensor, frequency_factor: Tensor
) -> Tensor:
    """Official SirenFNO 1-D CP contraction in ``[C_in,C_out,L_f]`` order."""

    return torch.einsum(
        "ir,jr,lr->ijl", input_factor, output_factor, frequency_factor
    )


def _tt_reconstruct_1d(
    input_factor: Tensor, output_factor: Tensor, frequency_raw: Tensor
) -> Tensor:
    """Official TT order ``C_in -> frequency -> C_out``."""

    rank = input_factor.shape[-1]
    frequency_core = frequency_raw.reshape(-1, rank, rank).permute(1, 0, 2)
    partial = torch.einsum("ir,rlq->ilq", input_factor, frequency_core)
    return torch.einsum("ilq,qj->ijl", partial, output_factor.t())


def _tucker_reconstruct_1d(
    core: Tensor,
    input_factor: Tensor,
    output_factor: Tensor,
    frequency_factor: Tensor,
) -> Tensor:
    """Official SirenFNO 1-D Tucker contraction."""

    return torch.einsum(
        "abx,ia,jb,lx->ijl",
        core,
        input_factor,
        output_factor,
        frequency_factor,
    )


class CAFEPlusKernelGenerator1D(nn.Module):
    """Generate a full complex ``[C_out, C_in, L_f]`` Fourier kernel.

    Factorization changes only the parameterization. Every factorized branch
    reconstructs the complete raw kernel, calibrates one persistent base scale
    against the projected kernel RMS, and applies one learnable log gain. The
    dense path keeps its original Hermitian projection and spectral multiply.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int = 2,
        branch_dim: Optional[int] = None,
        hidden_dim: int = 64,
        kernel_mlp_type: str = "joint",
        activation: str = "gelu",
        dropout: float = 0.0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        enforce_hermitian: bool = True,
        output_init_std: float = 1e-3,
        factorization: Factorization = "dense",
        rank: int = 16,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
    ) -> None:
        super().__init__()
        self.spatial_dim = 1
        self.in_channels = _positive_int("in_channels", in_channels)
        self.out_channels = _positive_int("out_channels", out_channels)
        self.matrix_dim = self.out_channels * self.in_channels
        self.kernel_mlp_type = _canonical_kernel_mlp_type(kernel_mlp_type)
        self.enforce_hermitian = bool(enforce_hermitian)
        self.factorization = _canonical_factorization(factorization)
        self.rank = _positive_int("rank", rank)

        if self.factorization == "dense":
            # Attribute names and construction order are intentionally unchanged
            # for strict state-dict and numerical compatibility.
            self.encoder = CAFEPlusModeEncoder(
                rff_basis=rff_basis,
                cheb_basis=cheb_basis,
                num_branches=num_branches,
                branch_dim=branch_dim,
                sigma_init=sigma_init,
                learnable_sigma=learnable_sigma,
            )
            self.head = JointRealImagKernelHead(
                input_dim=self.encoder.branch_dim,
                hidden_dim=int(hidden_dim),
                matrix_dim=self.matrix_dim,
                activation=activation,
                dropout=float(dropout),
                output_init_std=float(output_init_std),
            )
        else:
            self.factor_rff_basis = _positive_int(
                "factor_rff_basis", factor_rff_basis
            )
            self.factor_cheb_basis = _positive_int(
                "factor_cheb_basis", factor_cheb_basis
            )
            self.factor_branch_dim = _positive_int(
                "factor_branch_dim", factor_branch_dim
            )
            self.factor_hidden_dim = _positive_int(
                "factor_hidden_dim", factor_hidden_dim
            )
            self.factor_output_init_mode = str(
                factor_output_init_mode
            ).strip().lower()
            if self.factor_output_init_mode != "xavier":
                raise ValueError("factor_output_init_mode must be 'xavier'")
            if (
                not isinstance(factor_kernel_target_rms, (int, float))
                or isinstance(factor_kernel_target_rms, bool)
                or not math.isfinite(float(factor_kernel_target_rms))
                or float(factor_kernel_target_rms) <= 0.0
            ):
                raise ValueError(
                    "factor_kernel_target_rms must be finite and positive"
                )
            self.factor_kernel_target_rms = float(factor_kernel_target_rms)
            frequency_dim = self.rank * self.rank if self.factorization == "tt" else self.rank
            role_specs = (
                ("input_real", self.in_channels, self.rank),
                ("input_imag", self.in_channels, self.rank),
                ("output_real", self.out_channels, self.rank),
                ("output_imag", self.out_channels, self.rank),
                ("frequency_real", None, frequency_dim),
                ("frequency_imag", None, frequency_dim),
            )
            self.factor_blocks = nn.ModuleDict(
                {
                    role: CAFEPlusScalarFactorBlock(
                        output_dim=output_dim,
                        rff_basis=self.factor_rff_basis,
                        cheb_basis=self.factor_cheb_basis,
                        num_branches=num_branches,
                        branch_dim=self.factor_branch_dim,
                        hidden_dim=self.factor_hidden_dim,
                        activation=activation,
                        dropout=float(dropout),
                        sigma_init=sigma_init,
                        learnable_sigma=learnable_sigma,
                        output_init_std=float(output_init_std),
                        output_init_mode=self.factor_output_init_mode,
                    )
                    for role, _, output_dim in role_specs
                }
            )
            if self.factorization == "tucker":
                core_std = 1.0 / math.sqrt(self.rank)
                core_shape = (self.rank, self.rank, self.rank)
                self.core_real = nn.Parameter(torch.randn(core_shape) * core_std)
                self.core_imag = nn.Parameter(torch.randn(core_shape) * core_std)
            self.register_buffer(
                "factor_kernel_base_scale",
                torch.zeros((), dtype=torch.float32),
                persistent=True,
            )
            self.factor_log_gain = nn.Parameter(torch.zeros(()))
            self._factor_kernel_calibrated = False

    @property
    def embedding_dim(self) -> int:
        if self.factorization == "dense":
            return self.encoder.embedding_dim
        return self.factor_blocks["input_real"].embedding_dim

    @property
    def branch_dim(self) -> int:
        if self.factorization == "dense":
            return self.encoder.branch_dim
        return self.factor_blocks["input_real"].branch_dim

    @property
    def sigma(self) -> Tensor:
        if self.factorization == "dense":
            return self.encoder.sigma
        return self.factor_blocks["input_real"].sigma

    @property
    def num_kernel_mlps(self) -> int:
        return 1 if self.factorization == "dense" else len(self.factor_blocks)

    @property
    def head_output_dim(self) -> Union[int, Dict[str, int]]:
        if self.factorization == "dense":
            return self.head.joint_mlp.output_dim
        return {
            role: block.output_dim for role, block in self.factor_blocks.items()
        }

    @property
    def rff_rng_metadata(self) -> Dict[str, str]:
        return {
            "rff_rng": "torch_global",
            "rff_seed_source": "global_experiment_seed",
        }

    def _reconstruct_dense(
        self, mode_real: Tensor, mode_imag: Tensor
    ) -> Tuple[Tensor, Tensor]:
        mode_count = mode_real.shape[0]
        expected = (mode_count, self.matrix_dim)
        if tuple(mode_real.shape) != expected or tuple(mode_imag.shape) != expected:
            raise RuntimeError(
                f"dense kernel head must return {expected}; got "
                f"{tuple(mode_real.shape)} and {tuple(mode_imag.shape)}"
            )
        real = mode_real.view(
            mode_count, self.out_channels, self.in_channels
        ).permute(1, 2, 0).contiguous()
        imag = mode_imag.view(
            mode_count, self.out_channels, self.in_channels
        ).permute(1, 2, 0).contiguous()
        return real, imag

    def factor_tensors(self, spatial_shape: Sequence[int]) -> Dict[str, Tensor]:
        """Generate raw functional factors for parity tests and inspection."""

        if self.factorization == "dense":
            raise RuntimeError("dense kernels do not expose factor tensors")
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("spatial_shape must contain one positive length")
        frequency_length = shape[0] // 2 + 1
        axis_lengths = {
            "input_real": self.in_channels,
            "input_imag": self.in_channels,
            "output_real": self.out_channels,
            "output_imag": self.out_channels,
            "frequency_real": frequency_length,
            "frequency_imag": frequency_length,
        }
        return {
            role: block(axis_lengths[role])
            for role, block in self.factor_blocks.items()
        }

    def _reconstruct_factorized_siren_layout(
        self, factors: Dict[str, Tensor], component: str
    ) -> Tensor:
        input_factor = factors[f"input_{component}"]
        output_factor = factors[f"output_{component}"]
        frequency_factor = factors[f"frequency_{component}"]
        if self.factorization == "cp":
            return _cp_reconstruct_1d(
                input_factor, output_factor, frequency_factor
            )
        if self.factorization == "tt":
            return _tt_reconstruct_1d(
                input_factor, output_factor, frequency_factor
            )
        if self.factorization == "tucker":
            core = self.core_real if component == "real" else self.core_imag
            return _tucker_reconstruct_1d(
                core, input_factor, output_factor, frequency_factor
            )
        raise RuntimeError("factorized reconstruction requires cp, tt, or tucker")

    def _raw_factorized_components(
        self, spatial_shape: Sequence[int]
    ) -> Tuple[Tensor, Tensor]:
        factors = self.factor_tensors(spatial_shape)
        real = self._reconstruct_factorized_siren_layout(factors, "real")
        imag = self._reconstruct_factorized_siren_layout(factors, "imag")
        return (
            real.permute(1, 0, 2).contiguous(),
            imag.permute(1, 0, 2).contiguous(),
        )

    @torch.no_grad()
    def _calibrate_factor_kernel(self, spatial_shape: Sequence[int]) -> None:
        """Set one persistent scale from the projected initial raw kernel."""

        if self._factor_kernel_calibrated:
            return
        was_training = self.training
        try:
            self.eval()
            real, imag = self._raw_factorized_components(spatial_shape)
            raw = torch.complex(real.float(), imag.float())
            if self.enforce_hermitian:
                raw = enforce_rfft1_kernel_hermitian(raw, int(spatial_shape[0]))
            raw_rms = raw.abs().square().mean().sqrt()
            if not torch.isfinite(raw_rms) or float(raw_rms.item()) <= 0.0:
                raise RuntimeError("factorized kernel calibration RMS is invalid")
            scale = self.factor_kernel_target_rms / (raw_rms + 1e-12)
            self.factor_kernel_base_scale.copy_(
                scale.to(self.factor_kernel_base_scale)
            )
        finally:
            self.train(was_training)
        self._factor_kernel_calibrated = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        calibration_key = prefix + "factor_kernel_base_scale"
        calibrated = False
        if self.factorization != "dense" and calibration_key in state_dict:
            loaded_scale = state_dict[calibration_key]
            if not torch.is_tensor(loaded_scale) or loaded_scale.numel() != 1:
                error_msgs.append(
                    f'{calibration_key} must contain one finite non-negative scalar'
                )
            else:
                try:
                    scale_value = float(
                        loaded_scale.detach().to(device="cpu").item()
                    )
                except (TypeError, ValueError, RuntimeError):
                    error_msgs.append(
                        f'{calibration_key} must contain one finite non-negative scalar'
                    )
                else:
                    if not math.isfinite(scale_value) or scale_value < 0.0:
                        error_msgs.append(
                            f'{calibration_key} must contain one finite '
                            "non-negative scalar"
                        )
                    else:
                        calibrated = scale_value > 0.0
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.factorization != "dense":
            self._factor_kernel_calibrated = calibrated

    def forward(self, spatial_shape: Sequence[int]) -> Tensor:
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("spatial_shape must contain one positive length")

        if self.factorization == "dense":
            # Keep the canonical dense operation order exactly unchanged.
            cafe_features = self.encoder(shape[0])
            mode_real, mode_imag = self.head(cafe_features)
            real, imag = self._reconstruct_dense(mode_real, mode_imag)
        else:
            if not self._factor_kernel_calibrated:
                self._calibrate_factor_kernel(shape)
            real, imag = self._raw_factorized_components(shape)
            scale = (
                self.factor_kernel_base_scale.to(real)
                * self.factor_log_gain.exp()
            )
            real = scale * real
            imag = scale * imag

        expected_modes = shape[0] // 2 + 1
        expected_kernel_shape = (
            self.out_channels,
            self.in_channels,
            expected_modes,
        )
        if tuple(real.shape) != expected_kernel_shape:
            raise RuntimeError(
                f"reconstructed kernel has shape {tuple(real.shape)}, "
                f"expected {expected_kernel_shape}"
            )

        if real.dtype in (torch.float16, torch.bfloat16):
            real = real.float()
            imag = imag.float()
        kernel = torch.complex(real, imag)
        if self.enforce_hermitian:
            kernel = enforce_rfft1_kernel_hermitian(kernel, shape[0])
        return kernel

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"factorization='{self.factorization}', rank={self.rank}, "
            f"full_kernel_materialization=True"
        )


class CAFEPlusSpectralConv1D(nn.Module):
    """Full-frequency 1-D CAFE+-parameterized spectral convolution."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int = 2,
        branch_dim: Optional[int] = None,
        hidden_dim: int = 64,
        kernel_mlp_type: str = "joint",
        activation: str = "gelu",
        dropout: float = 0.0,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        enforce_hermitian: bool = True,
        output_init_std: float = 1e-3,
        factorization: Factorization = "dense",
        rank: int = 16,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
    ) -> None:
        super().__init__()
        self.spatial_dim = 1
        self.in_channels = _positive_int("in_channels", in_channels)
        self.out_channels = _positive_int("out_channels", out_channels)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.output_dropout = float(dropout)

        self.kernel_generator = CAFEPlusKernelGenerator1D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            rff_basis=rff_basis,
            cheb_basis=cheb_basis,
            num_branches=num_branches,
            branch_dim=branch_dim,
            hidden_dim=hidden_dim,
            kernel_mlp_type=kernel_mlp_type,
            activation=activation,
            dropout=dropout,
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma,
            enforce_hermitian=enforce_hermitian,
            output_init_std=output_init_std,
            factorization=factorization,
            rank=rank,
            factor_rff_basis=factor_rff_basis,
            factor_cheb_basis=factor_cheb_basis,
            factor_branch_dim=factor_branch_dim,
            factor_hidden_dim=factor_hidden_dim,
            factor_output_init_mode=factor_output_init_mode,
            factor_kernel_target_rms=factor_kernel_target_rms,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels, 1))

    @property
    def sigma(self) -> Tensor:
        return self.kernel_generator.sigma

    @property
    def embedding_dim(self) -> int:
        return self.kernel_generator.embedding_dim

    @property
    def branch_dim(self) -> int:
        return self.kernel_generator.branch_dim

    @property
    def num_kernel_mlps(self) -> int:
        return self.kernel_generator.num_kernel_mlps

    @property
    def factorization(self) -> str:
        return self.kernel_generator.factorization

    @property
    def rank(self) -> int:
        return self.kernel_generator.rank

    def forward(self, x: Tensor, *, return_kernel: bool = False):
        if x.ndim != 3:
            raise ValueError(
                f"expected a rank-3 channel-first tensor, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[-1] <= 0:
            raise ValueError("the spatial length must be positive")
        if not torch.is_floating_point(x):
            raise TypeError("spectral convolution input must be real floating point")

        original_dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spatial_shape = (int(work.shape[-1]),)
        transform_dims = (-1,)
        x_ft = torch.fft.rfftn(work, dim=transform_dims)
        kernel = self.kernel_generator(spatial_shape).to(
            device=x_ft.device, dtype=x_ft.dtype
        )
        out_ft = torch.einsum("bik,oik->bok", x_ft, kernel)
        y = torch.fft.irfftn(
            out_ft, s=spatial_shape, dim=transform_dims
        )

        if self.output_dropout > 0.0 and self.training:
            y = F.dropout(y, p=self.output_dropout)
        y = y + self.bias.to(device=y.device, dtype=y.dtype)
        y = y.to(original_dtype)
        if return_kernel:
            return y, kernel
        return y


class PointwiseMLP1D(nn.Module):
    """Two-layer pointwise MLP implemented by 1x1 convolutions."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        hidden_channels = _positive_int("hidden_channels", hidden_channels)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.linear1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.activation = _make_activation(activation)
        self.dropout = (
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.linear2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class _CAFEPlusFNOOperatorLayer1D(nn.Module):
    """One residual 1-D CAFE+FNO operator layer."""

    def __init__(
        self,
        *,
        channels: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int,
        branch_dim: Optional[int],
        kernel_hidden_dim: int,
        kernel_mlp_type: str,
        kernel_activation: str,
        sigma_init: float,
        learnable_sigma: bool,
        enforce_hermitian: bool,
        kernel_output_init_std: float,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        factorization: Factorization = "dense",
        rank: int = 16,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
    ) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        ffn_expansion = _positive_int("ffn_expansion", ffn_expansion)
        self.spectral = CAFEPlusSpectralConv1D(
            in_channels=channels,
            out_channels=channels,
            rff_basis=rff_basis,
            cheb_basis=cheb_basis,
            num_branches=num_branches,
            branch_dim=branch_dim,
            hidden_dim=kernel_hidden_dim,
            kernel_mlp_type=kernel_mlp_type,
            activation=kernel_activation,
            dropout=dropout,
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma,
            enforce_hermitian=enforce_hermitian,
            output_init_std=kernel_output_init_std,
            factorization=factorization,
            rank=rank,
            factor_rff_basis=factor_rff_basis,
            factor_cheb_basis=factor_cheb_basis,
            factor_branch_dim=factor_branch_dim,
            factor_hidden_dim=factor_hidden_dim,
            factor_output_init_mode=factor_output_init_mode,
            factor_kernel_target_rms=factor_kernel_target_rms,
        )
        self.ffn = PointwiseMLP1D(
            in_channels=channels,
            out_channels=channels,
            hidden_channels=ffn_expansion * channels,
            dropout=dropout,
            activation="gelu",
        )

    @property
    def sigma(self) -> Tensor:
        return self.spectral.sigma

    @property
    def embedding_dim(self) -> int:
        return self.spectral.embedding_dim

    @property
    def branch_dim(self) -> int:
        return self.spectral.branch_dim

    @property
    def num_kernel_mlps(self) -> int:
        return self.spectral.num_kernel_mlps

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.spectral(x))


class CAFEPlusFNO1D(nn.Module):
    """One-dimensional CAFE+FNO with optional kernel factorization.

    Parameters
    ----------
    factorization:
        ``'dense'`` (default), ``'cp'``, ``'tt'``, or ``'tucker'``.
        Factorized variants use independent midpoint-sampled scalar-axis CAFE+
        blocks for input channels, output channels, and rFFT frequency.  The
        full kernel is reconstructed before multiplication.
    rank:
        Positive integer factorization rank.  TT follows
        ``C_in -> frequency -> C_out`` and Tucker uses multilinear rank
        ``(R, R, R)``.  It is retained as metadata but ignored by ``dense``.

    Notes
    -----
    Inputs may be ``[B, C_in, L]`` or ``[B, L, C_in]``.  With
    ``input_layout='auto'``, ambiguous shapes require an explicit layout.
    ``H``/``L`` are accepted only for legacy constructor compatibility because
    the FFT length is inferred on every forward pass.  Legacy ``m`` and ``n``
    map to ``rff_basis`` and ``cheb_basis`` respectively.
    """

    def __init__(
        self,
        width: int = 32,
        input_dim: int = 1,
        output_dim: int = 1,
        padding: PaddingLike = 0,
        mlp_dropout: float = 0.0,
        add_grid: bool = True,
        num_layers: int = 4,
        ffn_expansion: int = 4,
        rff_basis: int = 32,
        cheb_basis: int = 32,
        cafe_branches: int = 2,
        cafe_branch_dim: Optional[int] = None,
        kernel_hidden_dim: int = 64,
        kernel_mlp_type: str = "joint",
        kernel_activation: str = "gelu",
        kernel_output_init_std: float = 1e-3,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        enforce_hermitian: bool = True,
        input_layout: InputLayout = "auto",
        output_layout: OutputLayout = "match_input",
        factorization: Factorization = "dense",
        rank: int = 16,
        variant: Optional[str] = None,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
        # Legacy aliases; resolution is inferred at every forward pass.
        H: Optional[int] = None,
        L: Optional[int] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
    ) -> None:
        super().__init__()
        del H, L
        if m is not None:
            rff_basis = _positive_int("m", m)
        if n is not None:
            cheb_basis = _positive_int("n", n)

        self.spatial_dim = 1
        self.width = _positive_int("width", width)
        self.input_dim = _positive_int("input_dim", input_dim)
        self.output_dim = _positive_int("output_dim", output_dim)
        self.padding = _normalize_padding(padding)
        if not isinstance(mlp_dropout, (int, float)) or isinstance(
            mlp_dropout, bool
        ):
            raise TypeError("mlp_dropout must be a real number in [0, 1)")
        if not 0.0 <= float(mlp_dropout) < 1.0:
            raise ValueError("mlp_dropout must be in [0, 1)")
        self.mlp_dropout = float(mlp_dropout)
        self.add_grid = bool(add_grid)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.ffn_expansion = _positive_int("ffn_expansion", ffn_expansion)
        self.rff_basis = _positive_int("rff_basis", rff_basis)
        self.cheb_basis = _positive_int("cheb_basis", cheb_basis)
        self.cafe_branches = _positive_int("cafe_branches", cafe_branches)
        self.cafe_branch_dim = cafe_branch_dim
        self.kernel_hidden_dim = _positive_int(
            "kernel_hidden_dim", kernel_hidden_dim
        )
        self.kernel_mlp_type = _canonical_kernel_mlp_type(kernel_mlp_type)
        self.kernel_activation = str(kernel_activation).strip().lower()
        _make_activation(self.kernel_activation)
        if (
            not isinstance(kernel_output_init_std, (int, float))
            or isinstance(kernel_output_init_std, bool)
        ):
            raise TypeError("kernel_output_init_std must be positive")
        if (
            not math.isfinite(float(kernel_output_init_std))
            or float(kernel_output_init_std) <= 0.0
        ):
            raise ValueError("kernel_output_init_std must be finite and positive")
        self.kernel_output_init_std = float(kernel_output_init_std)
        if not isinstance(sigma_init, (int, float)) or isinstance(sigma_init, bool):
            raise TypeError("sigma_init must be positive")
        if not math.isfinite(float(sigma_init)) or float(sigma_init) <= 0.0:
            raise ValueError("sigma_init must be finite and positive")
        self.sigma_init = float(sigma_init)
        self.learnable_sigma = bool(learnable_sigma)
        self.enforce_hermitian = bool(enforce_hermitian)
        self.factorization = _canonical_factorization(factorization)
        self.rank = _positive_int("rank", rank)
        expected_variant = self.factorization
        if variant is not None and variant != expected_variant:
            raise ValueError(
                f"variant is derived from factorization and must be "
                f"'{expected_variant}', got {variant!r}"
            )
        self.variant = expected_variant

        if self.factorization == "dense":
            self.factor_rff_basis = None
            self.factor_cheb_basis = None
            self.factor_embedding_dim = None
            self.factor_branch_dim = None
            self.factor_hidden_dim = None
            self.factor_output_init_mode = None
            self.factor_kernel_target_rms = None
        else:
            self.factor_rff_basis = _positive_int(
                "factor_rff_basis", factor_rff_basis
            )
            self.factor_cheb_basis = _positive_int(
                "factor_cheb_basis", factor_cheb_basis
            )
            self.factor_embedding_dim = (
                2 * self.factor_rff_basis + self.factor_cheb_basis
            )
            self.factor_branch_dim = _positive_int(
                "factor_branch_dim", factor_branch_dim
            )
            self.factor_hidden_dim = _positive_int(
                "factor_hidden_dim", factor_hidden_dim
            )
            self.factor_output_init_mode = str(
                factor_output_init_mode
            ).strip().lower()
            if self.factor_output_init_mode != "xavier":
                raise ValueError("factor_output_init_mode must be 'xavier'")
            if (
                not isinstance(factor_kernel_target_rms, (int, float))
                or isinstance(factor_kernel_target_rms, bool)
                or not math.isfinite(float(factor_kernel_target_rms))
                or float(factor_kernel_target_rms) <= 0.0
            ):
                raise ValueError(
                    "factor_kernel_target_rms must be finite and positive"
                )
            self.factor_kernel_target_rms = float(factor_kernel_target_rms)

        if input_layout not in {"auto", "channels_first", "channels_last"}:
            raise ValueError("unsupported input_layout")
        if output_layout not in {
            "match_input",
            "channels_first",
            "channels_last",
        }:
            raise ValueError("unsupported output_layout")
        self.input_layout = input_layout
        self.output_layout = output_layout

        if self.factorization == "dense":
            self.embedding_dim = 2 * self.rff_basis + self.cheb_basis
            self.branch_dim = (
                self.embedding_dim
                if cafe_branch_dim is None
                else _positive_int("cafe_branch_dim", cafe_branch_dim)
            )
        else:
            self.embedding_dim = int(self.factor_embedding_dim)
            self.branch_dim = int(self.factor_branch_dim)

        lifting_channels = self.input_dim + (1 if self.add_grid else 0)
        self.lifting = nn.Conv1d(lifting_channels, self.width, kernel_size=1)
        self.operator_layers = nn.ModuleList(
            [
                _CAFEPlusFNOOperatorLayer1D(
                    channels=self.width,
                    rff_basis=self.rff_basis,
                    cheb_basis=self.cheb_basis,
                    num_branches=self.cafe_branches,
                    branch_dim=self.branch_dim,
                    kernel_hidden_dim=self.kernel_hidden_dim,
                    kernel_mlp_type=self.kernel_mlp_type,
                    kernel_activation=self.kernel_activation,
                    sigma_init=self.sigma_init,
                    learnable_sigma=self.learnable_sigma,
                    enforce_hermitian=self.enforce_hermitian,
                    kernel_output_init_std=self.kernel_output_init_std,
                    ffn_expansion=self.ffn_expansion,
                    dropout=self.mlp_dropout,
                    factorization=self.factorization,
                    rank=self.rank,
                    factor_rff_basis=factor_rff_basis,
                    factor_cheb_basis=factor_cheb_basis,
                    factor_branch_dim=factor_branch_dim,
                    factor_hidden_dim=factor_hidden_dim,
                    factor_output_init_mode=factor_output_init_mode,
                    factor_kernel_target_rms=factor_kernel_target_rms,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.projection = PointwiseMLP1D(
            in_channels=self.width,
            out_channels=self.output_dim,
            hidden_channels=self.ffn_expansion * self.width,
            dropout=0.0,
            activation="gelu",
        )

    def _infer_input_layout(self, x: Tensor) -> str:
        if self.input_layout != "auto":
            return self.input_layout
        channel_first = x.shape[1] == self.input_dim
        channel_last = x.shape[-1] == self.input_dim
        if channel_first and not channel_last:
            return "channels_first"
        if channel_last and not channel_first:
            return "channels_last"
        if channel_first and channel_last:
            raise ValueError(
                f"ambiguous channel axis for input shape {tuple(x.shape)}; "
                "set input_layout explicitly"
            )
        raise ValueError(
            f"cannot infer channel axis for input shape {tuple(x.shape)} and "
            f"input_dim={self.input_dim}; set input_layout explicitly"
        )

    def _to_channel_first(self, x: Tensor) -> Tuple[Tensor, str]:
        if x.ndim != 3:
            raise ValueError(f"expected a rank-3 tensor, got {tuple(x.shape)}")
        layout = self._infer_input_layout(x)
        if layout == "channels_last":
            return x.permute(0, 2, 1).contiguous(), layout
        return x, layout

    def _format_output(self, y: Tensor, input_layout: str) -> Tensor:
        layout = self.output_layout
        if layout == "match_input":
            layout = input_layout
        if layout == "channels_last":
            return y.permute(0, 2, 1).contiguous()
        return y

    @staticmethod
    def _make_grid(
        batch: int,
        full_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return (
            torch.linspace(0.0, 1.0, full_length, device=device, dtype=dtype)
            .view(1, 1, full_length)
            .expand(batch, 1, full_length)
        )

    @staticmethod
    def _prepare_grid(grid: Tensor, x: Tensor) -> Tensor:
        if grid.ndim != 3:
            raise ValueError("grid must be rank 3")
        channel_first = grid.shape[1] == 1 and grid.shape[2] == x.shape[2]
        channel_last = grid.shape[-1] == 1 and grid.shape[1] == x.shape[2]
        if channel_first and channel_last:
            raise ValueError(
                f"ambiguous coordinate axis for grid shape {tuple(grid.shape)}"
            )
        if channel_first:
            grid_cf = grid
        elif channel_last:
            grid_cf = grid.permute(0, 2, 1).contiguous()
        else:
            raise ValueError("grid must contain one coordinate channel")
        if grid_cf.shape[0] != x.shape[0]:
            raise ValueError(
                "grid and input must have the same batch size; "
                f"got {grid_cf.shape[0]} and {x.shape[0]}"
            )
        if grid_cf.shape[2] != x.shape[2]:
            raise ValueError(
                "grid and input must have the same spatial length; "
                f"got {grid_cf.shape[2]} and {x.shape[2]}"
            )
        return grid_cf.to(device=x.device, dtype=x.dtype)

    def _pad(self, x: Tensor) -> Tensor:
        return F.pad(x, (0, self.padding[0])) if self.padding[0] else x

    def _crop(self, x: Tensor) -> Tensor:
        return x[..., : -self.padding[0]] if self.padding[0] else x

    def _sigma_encoders_by_role(self) -> Dict[str, list[nn.Module]]:
        """Return every independent sigma-owning encoder grouped by role."""

        generators = [
            layer.spectral.kernel_generator for layer in self.operator_layers
        ]
        if self.factorization == "dense":
            return {"dense": [generator.encoder for generator in generators]}
        roles = tuple(generators[0].factor_blocks.keys())
        return {
            role: [generator.factor_blocks[role].encoder for generator in generators]
            for role in roles
        }

    def sigma_values(
        self, *, detach: bool = True
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Return sigma diagnostics without changing their device.

        Dense models return one sigma per operator layer as ``Tensor[L]``.
        Factorized models return a dictionary containing one ``Tensor[L]`` per
        independent functional-factor role.  ``detach=False`` preserves the
        autograd graph when sigma is learnable.
        """

        grouped_encoders = self._sigma_encoders_by_role()
        values = {
            role: torch.stack([encoder.sigma for encoder in encoders])
            for role, encoders in grouped_encoders.items()
        }
        if detach:
            values = {role: value.detach() for role, value in values.items()}
        if self.factorization == "dense":
            return values["dense"]
        return values

    def sigma_regularization(self) -> Tensor:
        """Mean squared log-sigma across every independent bandwidth."""

        values = [
            encoder.log_sigma.square()
            for encoders in self._sigma_encoders_by_role().values()
            for encoder in encoders
        ]
        return torch.stack(values).mean()

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        parameters: Iterable[nn.Parameter] = self.parameters()
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def count_kernel_parameters(self, *, trainable_only: bool = True) -> int:
        """Count parameters used by all Fourier-kernel generators only."""

        parameters: Iterable[nn.Parameter] = (
            parameter
            for layer in self.operator_layers
            for parameter in layer.spectral.kernel_generator.parameters()
        )
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def get_config(self) -> Dict[str, object]:
        """Return public-constructor arguments required to rebuild this model."""

        config: Dict[str, object] = {
            "width": self.width,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "padding": self.padding,
            "mlp_dropout": self.mlp_dropout,
            "add_grid": self.add_grid,
            "num_layers": self.num_layers,
            "ffn_expansion": self.ffn_expansion,
            "rff_basis": self.rff_basis,
            "cheb_basis": self.cheb_basis,
            "cafe_branches": self.cafe_branches,
            "cafe_branch_dim": self.cafe_branch_dim,
            "kernel_hidden_dim": self.kernel_hidden_dim,
            "kernel_mlp_type": self.kernel_mlp_type,
            "kernel_activation": self.kernel_activation,
            "kernel_output_init_std": self.kernel_output_init_std,
            "sigma_init": self.sigma_init,
            "learnable_sigma": self.learnable_sigma,
            "enforce_hermitian": self.enforce_hermitian,
            "input_layout": self.input_layout,
            "output_layout": self.output_layout,
            "factorization": self.factorization,
            "rank": self.rank,
        }
        if self.factorization != "dense":
            config.update(
                {
                    "factor_rff_basis": self.factor_rff_basis,
                    "factor_cheb_basis": self.factor_cheb_basis,
                    "factor_branch_dim": self.factor_branch_dim,
                    "factor_hidden_dim": self.factor_hidden_dim,
                    "factor_output_init_mode": self.factor_output_init_mode,
                    "factor_kernel_target_rms": self.factor_kernel_target_rms,
                }
            )
        return config

    def checkpoint_dict(self) -> Dict[str, object]:
        """Build a portable checkpoint with constructor config and metadata."""

        return {
            "checkpoint_format_version": 2,
            "model_name": MODEL_NAME,
            "model_key": MODEL_REGISTRY_KEY,
            "implementation_version": __version__,
            "spatial_dim": 1,
            "variant": self.variant,
            "factorization": self.factorization,
            "rank": self.rank,
            "config": self.get_config(),
            "metadata": self.architecture_summary(),
            "state_dict": self.state_dict(),
        }

    def architecture_summary(self) -> Dict[str, object]:
        generator = self.operator_layers[0].spectral.kernel_generator
        return {
            "model": self.__class__.__name__,
            "display_name": MODEL_NAME,
            "registry_key": MODEL_REGISTRY_KEY,
            "implementation_version": __version__,
            "variant": self.variant,
            "factorization": self.factorization,
            "rank": self.rank,
            "rank_is_active": self.factorization != "dense",
            "parameter_factorized": self.factorization != "dense",
            "full_kernel_materialization": True,
            "fft_flop_reduction_claimed": False,
            "peak_kernel_memory_reduction_claimed": False,
            "factorization_source": (
                "SirenFNO functional tensor decomposition"
                if self.factorization != "dense"
                else "canonical CAFE+FNO dense head"
            ),
            "factor_generator": "CAFE+",
            "decomposition_axes": ("C_in", "C_out", "frequency"),
            "full_kernel_materialized_before_spectral_mixing": True,
            "tensorization": (
                "canonical dense joint mode head"
                if self.factorization == "dense"
                else "C_in x frequency x C_out; independent scalar-axis factors"
            ),
            "tt_mode_order": ("C_in", "frequency", "C_out"),
            "factor_axis_coordinate_convention": (
                "((0.5 + arange(N)) / N) * 2 - 1"
                if self.factorization != "dense"
                else None
            ),
            "real_imag_parameterization": (
                "one joint dense MLP"
                if self.factorization == "dense"
                else "independent CAFE+ blocks and parameters"
            ),
            **generator.rff_rng_metadata,
            "config": self.get_config(),
            "spatial_dim": 1,
            "kernel_mlp_type": self.kernel_mlp_type,
            "kernel_mlps_per_layer": generator.num_kernel_mlps,
            "kernel_head_output_dim_per_layer": generator.head_output_dim,
            "operator_layers": self.num_layers,
            "width": self.width,
            "rff_basis_m": (
                self.rff_basis
                if self.factorization == "dense"
                else self.factor_rff_basis
            ),
            "cheb_basis_n": (
                self.cheb_basis
                if self.factorization == "dense"
                else self.factor_cheb_basis
            ),
            "embedding_dim_2m_plus_n": self.embedding_dim,
            "linear_branch_output_dim": self.branch_dim,
            "cafe_branches": self.cafe_branches,
            "kernel_hidden_dim": self.kernel_hidden_dim,
            "kernel_activation": self.kernel_activation,
            "kernel_output_init_std": self.kernel_output_init_std,
            "factor_rff_basis": self.factor_rff_basis,
            "factor_cheb_basis": self.factor_cheb_basis,
            "factor_embedding_dim": self.factor_embedding_dim,
            "factor_branch_dim": self.factor_branch_dim,
            "factor_hidden_dim": self.factor_hidden_dim,
            "factor_output_init_mode": self.factor_output_init_mode,
            "factor_kernel_target_rms": self.factor_kernel_target_rms,
            "learnable_sigma": self.learnable_sigma,
            "sigma_init": self.sigma_init,
            "enforce_hermitian": self.enforce_hermitian,
            "parameters": self.count_parameters(),
            "kernel_generator_trainable_parameters": (
                self.count_kernel_parameters(trainable_only=True)
            ),
            "kernel_generator_total_parameters": (
                self.count_kernel_parameters(trainable_only=False)
            ),
        }

    @torch.no_grad()
    def generate_kernels(
        self,
        spatial_shape: Sequence[int],
        *,
        deterministic: bool = True,
    ) -> Tuple[Tensor, ...]:
        """Generate layer kernels for an unpadded input spatial shape."""

        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("spatial_shape must contain one positive length")
        fft_shape = (shape[0] + self.padding[0],)
        generators = tuple(
            layer.spectral.kernel_generator for layer in self.operator_layers
        )
        training_states = tuple(generator.training for generator in generators)
        try:
            if deterministic:
                for generator in generators:
                    generator.eval()
            return tuple(generator(fft_shape) for generator in generators)
        finally:
            if deterministic:
                for generator, was_training in zip(generators, training_states):
                    generator.train(was_training)

    def forward(
        self,
        x: Optional[Tensor] = None,
        grid: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if x is None:
            x = kwargs.pop("x", None)
        if kwargs:
            raise TypeError(f"unexpected forward arguments: {sorted(kwargs)}")
        if x is None:
            raise TypeError("forward expected argument 'x'")
        if not torch.is_floating_point(x):
            raise TypeError("x must be a real floating-point tensor")

        x, input_layout = self._to_channel_first(x)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} input channels, got {x.shape[1]}"
            )
        full_length = int(x.shape[-1])
        if full_length <= 0:
            raise ValueError("input spatial length must be positive")

        if self.add_grid:
            grid_cf = (
                self._make_grid(
                    x.shape[0],
                    full_length,
                    device=x.device,
                    dtype=x.dtype,
                )
                if grid is None
                else self._prepare_grid(grid, x)
            )
            x = torch.cat((x, grid_cf), dim=1)
        elif grid is not None:
            raise ValueError("grid was provided but add_grid=False")

        x = self._pad(self.lifting(x))
        for index, layer in enumerate(self.operator_layers):
            x = layer(x)
            if index < self.num_layers - 1:
                x = F.gelu(x)
        x = self._crop(x)
        y = self.projection(x)
        return self._format_output(y, input_layout)


def build_cafe_plus_fno_1d(**kwargs) -> CAFEPlusFNO1D:
    """Build a :class:`CAFEPlusFNO1D` from explicit keyword arguments."""

    return CAFEPlusFNO1D(**kwargs)


__all__ = [
    "MODEL_NAME",
    "MODEL_REGISTRY_KEY",
    "CAFEPlusFNO1D",
    "build_cafe_plus_fno_1d",
]
