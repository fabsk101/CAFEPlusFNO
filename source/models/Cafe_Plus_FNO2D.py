"""Two-dimensional CAFE+FNO with dense and tensor-factorized kernels.

The public :class:`CAFEPlusFNO2D` model keeps the canonical dense CAFE+FNO
path unchanged.  Its CP, TT, and Tucker variants exactly follow the official
SirenFNO functional tensorization: independent scalar-axis CAFE+ blocks
generate factors for ``C_in``, ``C_out``, ``k_x``, and ``k_y`` separately for
real and imaginary kernels. These factor blocks use compact factor-specific
embedding, branch, and hidden widths. TT is ordered
``C_in -> k_x -> k_y -> C_out``.

All variants reconstruct the full complex kernel before spectral mixing and
then apply the same rFFT-boundary Hermitian projection.  At sufficiently small
rank, the factorized variants can reduce the number of kernel-generator
parameters; a large rank can instead increase it.  This implementation does
not claim reduced spectral-contraction FLOPs or peak kernel memory relative to
the dense variant.

Only PyTorch and the Python standard library are required.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .Cafe_Plus_FNO1D import CAFEPlusScalarFactorBlock

__version__ = "0.5.1"

MODEL_NAME = "CAFE+FNO"
MODEL_REGISTRY_KEY = "cafe_plus_fno"
JOINT_MODE_TENSORIZATION = (
    "C_out x joint_rFFT_mode(H*W_f) x C_in; real/imag independently"
)
FUNCTIONAL_AXIS_TENSORIZATION = (
    "C_in x kx x ky x C_out; independent scalar-axis real/imag CAFE+ blocks"
)

InputLayout = Literal["auto", "channels_first", "channels_last"]
OutputLayout = Literal["match_input", "channels_first", "channels_last"]
Factorization = Literal["dense", "cp", "tt", "tucker"]
PaddingLike = Union[int, Sequence[int]]


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
        raise TypeError(
            f"factorization must be a string, got {type(name).__name__}"
        )
    normalized = name.strip().lower()
    supported = {"dense", "cp", "tt", "tucker"}
    if normalized not in supported:
        raise ValueError(
            "factorization must be one of: dense, cp, tt, tucker; "
            f"got {name!r}"
        )
    return normalized


def _validate_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank must be a positive integer, got {rank!r}")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    return rank


def _make_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError("activation must be one of: gelu, relu, silu")


def _product(values: Sequence[int]) -> int:
    return int(reduce(mul, values, 1))


def _normalize_padding(padding: PaddingLike) -> Tuple[int, int]:
    if isinstance(padding, int):
        values = (int(padding), int(padding))
    else:
        if isinstance(padding, (str, bytes)):
            raise TypeError("padding must be an int or a two-element sequence")
        values = tuple(int(value) for value in padding)
    if len(values) != 2:
        raise ValueError(f"padding must contain two values, got {values}")
    if any(value < 0 for value in values):
        raise ValueError("padding values must be non-negative")
    return values


def chebyshev_embedding(x: Tensor, n_basis: int) -> Tensor:
    """Evaluate ``[T_0(x), ..., T_{n_basis-1}(x)]`` by recurrence."""

    if n_basis <= 0:
        raise ValueError("n_basis must be positive")
    x = x.clamp(-1.0, 1.0)
    t0 = torch.ones_like(x)
    if n_basis == 1:
        return t0.unsqueeze(-1)
    values = [t0, x]
    for _ in range(2, n_basis):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def normalized_rfft_coordinates(
    spatial_shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return native normalized ``(k_x, k_y)`` coordinates for ``rfft2``.

    The first axis follows signed ``fftfreq`` ordering in approximately
    ``[-1, 1)``.  The one-sided final axis lies in ``[0, 1]``.
    """

    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D spatial shape, got {shape}")
    if any(value <= 0 for value in shape):
        raise ValueError("all spatial dimensions must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a real floating-point dtype")

    kx = 2.0 * torch.fft.fftfreq(
        shape[0], d=1.0, device=device, dtype=dtype
    )
    ky = 2.0 * torch.fft.rfftfreq(
        shape[1], d=1.0, device=device, dtype=dtype
    )
    mesh = torch.meshgrid(kx, ky, indexing="ij")
    return torch.stack(mesh, dim=-1)


def _negative_frequency_indices(n: int, device: torch.device) -> Tensor:
    return torch.remainder(-torch.arange(n, device=device), n).long()


def enforce_rfft2_kernel_hermitian(
    kernel_half: Tensor, full_width: int
) -> Tensor:
    """Project a ``[C_out,C_in,H,W_f]`` kernel onto rFFT boundaries."""

    if not torch.is_complex(kernel_half):
        raise TypeError("kernel_half must be complex")
    if kernel_half.ndim != 4:
        raise ValueError("kernel_half must have shape [C_out, C_in, H, W_f]")
    if full_width <= 0:
        raise ValueError("full_width must be positive")
    expected_half_width = full_width // 2 + 1
    if kernel_half.shape[-1] != expected_half_width:
        raise ValueError(
            f"expected W_f={expected_half_width} for full_width={full_width}, "
            f"got {kernel_half.shape[-1]}"
        )

    out = kernel_half.clone()
    negative_x = _negative_frequency_indices(out.shape[-2], out.device)
    columns = [0]
    if full_width % 2 == 0:
        columns.append(full_width // 2)
    for column in columns:
        if column >= out.shape[-1]:
            continue
        value = out[..., column]
        partner = value.index_select(-1, negative_x).conj()
        out[..., column] = 0.5 * (value + partner)
    return out


class CAFEPlusModeEncoder2D(nn.Module):
    """Joint CAFE+ encoder whose fixed ``G`` uses the global PyTorch RNG."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        rff_basis: int,
        cheb_basis: int,
        num_branches: int = 2,
        branch_dim: Optional[int] = None,
        sigma_init: float = 1.0,
        learnable_sigma: bool = True,
        phase_scale: float = math.pi,
    ) -> None:
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("CAFEPlusModeEncoder2D accepts spatial_dim=2 only")
        if rff_basis <= 0 or cheb_basis <= 0:
            raise ValueError("rff_basis and cheb_basis must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if sigma_init <= 0.0:
            raise ValueError("sigma_init must be positive")

        self.spatial_dim = 2
        self.rff_basis = int(rff_basis)
        self.cheb_basis = int(cheb_basis)
        self.num_branches = int(num_branches)
        self.embedding_dim = 2 * self.rff_basis + 2 * self.cheb_basis
        self.branch_dim = (
            self.embedding_dim if branch_dim is None else int(branch_dim)
        )
        if self.branch_dim <= 0:
            raise ValueError("branch_dim must be positive")
        self.phase_scale = float(phase_scale)
        gaussian = torch.randn(
            2,
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
            Tuple[Tuple[int, ...], str, Optional[int], torch.dtype],
            Tuple[Tensor, Tensor],
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
        spatial_shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 2:
            raise ValueError(f"expected two spatial dimensions, got {shape}")
        key = (shape, device.type, device.index, dtype)
        cached = self._fixed_feature_cache.get(key)
        if cached is not None:
            return cached

        coordinates = normalized_rfft_coordinates(
            shape, device=device, dtype=dtype
        )
        flat_coordinates = coordinates.reshape(-1, 2)
        chebyshev = torch.cat(
            [
                chebyshev_embedding(flat_coordinates[:, axis], self.cheb_basis)
                for axis in range(2)
            ],
            dim=-1,
        )
        self._fixed_feature_cache[key] = (flat_coordinates, chebyshev)
        return flat_coordinates, chebyshev

    def build_embedding(
        self,
        spatial_shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        coordinates, chebyshev = self._fixed_features(
            spatial_shape, device=device, dtype=dtype
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

    def forward(self, spatial_shape: Sequence[int]) -> Tensor:
        reference = self.branches[0].weight
        feature_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        embedding = self.build_embedding(
            spatial_shape,
            device=reference.device,
            dtype=feature_dtype,
        ).to(device=reference.device, dtype=reference.dtype)
        product = self.branches[0](embedding)
        for branch in self.branches[1:]:
            product = product * branch(embedding)
        return product

    def extra_repr(self) -> str:
        return (
            f"spatial_dim=2, rff_basis={self.rff_basis}, "
            f"cheb_basis={self.cheb_basis}, embedding_dim={self.embedding_dim}, "
            f"branch_dim={self.branch_dim}, num_branches={self.num_branches}, "
            f"learnable_sigma={self.learnable_sigma}"
        )


class KernelMLP(nn.Module):
    """One-hidden-layer MLP used by every joint real/imag kernel head."""

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
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("MLP dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if output_init_std <= 0.0:
            raise ValueError("output_init_std must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.output_init_mode = str(output_init_mode).strip().lower()
        if self.output_init_mode not in {"normal", "xavier"}:
            raise ValueError("output_init_mode must be 'normal' or 'xavier'")
        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.zeros_(self.linear_in.bias)
        if self.output_init_mode == "normal":
            nn.init.normal_(self.linear_out.weight, mean=0.0, std=output_init_std)
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
    """Dense joint head retained byte-for-byte in parameter structure."""

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
        self.matrix_dim = int(matrix_dim)
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


def _cp_reconstruct_2d(
    input_factor: Tensor,
    output_factor: Tensor,
    kx_factor: Tensor,
    ky_factor: Tensor,
) -> Tensor:
    """Official SirenFNO 2-D CP contraction in ``[C_in,C_out,H,W_f]``."""

    return torch.einsum(
        "ir,jr,hr,wr->ijhw",
        input_factor,
        output_factor,
        kx_factor,
        ky_factor,
    )


def _tt_reconstruct_2d(
    input_factor: Tensor,
    output_factor: Tensor,
    kx_raw: Tensor,
    ky_raw: Tensor,
) -> Tensor:
    """Official TT order ``C_in -> k_x -> k_y -> C_out``."""

    rank = input_factor.shape[-1]
    kx_core = kx_raw.reshape(-1, rank, rank).permute(1, 0, 2)
    ky_core = ky_raw.reshape(-1, rank, rank).permute(1, 0, 2)
    partial = torch.einsum("ir,rhq->ihq", input_factor, kx_core)
    partial = torch.einsum("ihq,qwr->ihwr", partial, ky_core)
    return torch.einsum("ihwr,rj->ijhw", partial, output_factor.t())


def _tucker_reconstruct_2d(
    core: Tensor,
    input_factor: Tensor,
    output_factor: Tensor,
    kx_factor: Tensor,
    ky_factor: Tensor,
) -> Tensor:
    """Official SirenFNO 2-D Tucker contraction."""

    return torch.einsum(
        "abxy,ia,jb,hx,wy->ijhw",
        core,
        input_factor,
        output_factor,
        kx_factor,
        ky_factor,
    )


class CAFEPlusKernelGenerator2D(nn.Module):
    """Generate the full complex kernel used by the unchanged spectral mixer.

    Dense retains the canonical joint two-dimensional CAFE+ path. CP, TT, and
    Tucker use compact independent generators to reconstruct the SirenFNO-layout tensor
    ``[C_in,C_out,H,W_f]`` from eight independent scalar-axis CAFE+ blocks, then
    transpose only the channel axes for the existing CAFE spectral contraction.
    One persistent projected-RMS base scale and one learnable log gain are then
    applied to the reconstructed kernel.
    """

    def __init__(
        self,
        *,
        spatial_dim: int,
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
        factorization: str = "dense",
        rank: int = 16,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
    ) -> None:
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("CAFEPlusKernelGenerator2D accepts spatial_dim=2 only")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel counts must be positive")

        self.spatial_dim = 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.matrix_dim = self.out_channels * self.in_channels
        self.kernel_mlp_type = _canonical_kernel_mlp_type(kernel_mlp_type)
        self.enforce_hermitian = bool(enforce_hermitian)
        self.factorization = _canonical_factorization(factorization)
        self.rank = _validate_rank(rank)

        if self.factorization == "dense":
            # Preserve original names, construction order, and dense numerics.
            self.encoder = CAFEPlusModeEncoder2D(
                spatial_dim=2,
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
            self.factor_rff_basis = _validate_rank(factor_rff_basis)
            self.factor_cheb_basis = _validate_rank(factor_cheb_basis)
            self.factor_branch_dim = _validate_rank(factor_branch_dim)
            self.factor_hidden_dim = _validate_rank(factor_hidden_dim)
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
            spatial_output_dim = (
                self.rank * self.rank if self.factorization == "tt" else self.rank
            )
            role_specs = (
                ("input_real", self.in_channels, self.rank),
                ("input_imag", self.in_channels, self.rank),
                ("output_real", self.out_channels, self.rank),
                ("output_imag", self.out_channels, self.rank),
                ("kx_real", None, spatial_output_dim),
                ("kx_imag", None, spatial_output_dim),
                ("ky_real", None, spatial_output_dim),
                ("ky_imag", None, spatial_output_dim),
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
                core_shape = (self.rank, self.rank, self.rank, self.rank)
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

    def factor_tensors(self, spatial_shape: Sequence[int]) -> Dict[str, Tensor]:
        """Generate raw functional factors for parity tests and inspection."""

        if self.factorization == "dense":
            raise RuntimeError("dense kernels do not expose factor tensors")
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(f"expected two positive spatial sizes, got {shape}")
        half_width = shape[1] // 2 + 1
        axis_lengths = {
            "input_real": self.in_channels,
            "input_imag": self.in_channels,
            "output_real": self.out_channels,
            "output_imag": self.out_channels,
            "kx_real": shape[0],
            "kx_imag": shape[0],
            "ky_real": half_width,
            "ky_imag": half_width,
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
        kx_factor = factors[f"kx_{component}"]
        ky_factor = factors[f"ky_{component}"]
        if self.factorization == "cp":
            return _cp_reconstruct_2d(
                input_factor, output_factor, kx_factor, ky_factor
            )
        if self.factorization == "tt":
            return _tt_reconstruct_2d(
                input_factor, output_factor, kx_factor, ky_factor
            )
        if self.factorization == "tucker":
            core = self.core_real if component == "real" else self.core_imag
            return _tucker_reconstruct_2d(
                core, input_factor, output_factor, kx_factor, ky_factor
            )
        raise RuntimeError("factorized reconstruction requires cp, tt, or tucker")

    def _raw_factorized_components(
        self, spatial_shape: Sequence[int]
    ) -> Tuple[Tensor, Tensor]:
        factors = self.factor_tensors(spatial_shape)
        real = self._reconstruct_factorized_siren_layout(factors, "real")
        imag = self._reconstruct_factorized_siren_layout(factors, "imag")
        return (
            real.permute(1, 0, 2, 3).contiguous(),
            imag.permute(1, 0, 2, 3).contiguous(),
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
                raw = enforce_rfft2_kernel_hermitian(raw, int(spatial_shape[1]))
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
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(f"expected two positive spatial sizes, got {shape}")

        mode_shape = (shape[0], shape[1] // 2 + 1)
        expected_modes = _product(mode_shape)
        if self.factorization == "dense":
            # Canonical dense numerical path: operation order is unchanged.
            cafe_features = self.encoder(shape)
            real, imag = self.head(cafe_features)
            expected = (expected_modes, self.matrix_dim)
            if tuple(real.shape) != expected:
                raise RuntimeError(
                    f"kernel head returned {tuple(real.shape)}, expected {expected}"
                )
            view_shape = (*mode_shape, self.out_channels, self.in_channels)
            real = real.view(view_shape).permute(2, 3, 0, 1).contiguous()
            imag = imag.view(view_shape).permute(2, 3, 0, 1).contiguous()
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

        expected_kernel_shape = (
            self.out_channels,
            self.in_channels,
            *mode_shape,
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
            kernel = enforce_rfft2_kernel_hermitian(kernel, shape[1])
        return kernel


class CAFEPlusSpectralConv2D(nn.Module):
    """Full-frequency CAFE+-parameterized 2-D spectral convolution."""

    def __init__(
        self,
        *,
        spatial_dim: int,
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
        factorization: str = "dense",
        rank: int = 16,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
    ) -> None:
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("CAFEPlusSpectralConv2D accepts spatial_dim=2 only")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.spatial_dim = 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.output_dropout = float(dropout)
        self.factorization = _canonical_factorization(factorization)
        self.rank = _validate_rank(rank)

        self.kernel_generator = CAFEPlusKernelGenerator2D(
            spatial_dim=2,
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
            factorization=self.factorization,
            rank=self.rank,
            factor_rff_basis=factor_rff_basis,
            factor_cheb_basis=factor_cheb_basis,
            factor_branch_dim=factor_branch_dim,
            factor_hidden_dim=factor_hidden_dim,
            factor_output_init_mode=factor_output_init_mode,
            factor_kernel_target_rms=factor_kernel_target_rms,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1))

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

    @staticmethod
    def _complex_mix(x_ft: Tensor, kernel: Tensor) -> Tensor:
        return torch.einsum("bihw,oihw->bohw", x_ft, kernel)

    def forward(self, x: Tensor, *, return_kernel: bool = False):
        if x.ndim != 4:
            raise ValueError(
                f"expected a rank-4 channel-first tensor, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if not torch.is_floating_point(x):
            raise TypeError("spectral input must be real floating point")

        original_dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spatial_shape = tuple(int(value) for value in work.shape[-2:])
        x_ft = torch.fft.rfftn(work, dim=(-2, -1))
        kernel = self.kernel_generator(spatial_shape).to(
            device=x_ft.device, dtype=x_ft.dtype
        )
        out_ft = self._complex_mix(x_ft, kernel)
        y = torch.fft.irfftn(out_ft, s=spatial_shape, dim=(-2, -1))

        if self.output_dropout > 0.0 and self.training:
            y = F.dropout(y, p=self.output_dropout)
        y = y + self.bias.to(device=y.device, dtype=y.dtype)
        y = y.to(original_dtype)
        if return_kernel:
            return y, kernel
        return y


class PointwiseMLP2D(nn.Module):
    """Two-layer pointwise MLP implemented with 1x1 convolutions."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("PointwiseMLP2D accepts spatial_dim=2 only")
        if min(in_channels, out_channels, hidden_channels) <= 0:
            raise ValueError("all channel dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.linear1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class _CAFEPlusFNOOperatorLayer2D(nn.Module):
    """One residual 2-D CAFE+FNO operator layer."""

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
        factorization: str,
        rank: int,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectral = CAFEPlusSpectralConv2D(
            spatial_dim=2,
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
        self.ffn = PointwiseMLP2D(
            spatial_dim=2,
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


class CAFEPlusFNO2D(nn.Module):
    """Two-dimensional full-frequency CAFE+ Fourier neural operator.

    Parameters
    ----------
    factorization:
        ``"dense"`` (default), ``"cp"``, ``"tt"``, or ``"tucker"``.
        Factorization applies to the generated real and imaginary kernel
        tensors, not to the lifting, pointwise FFNs, or projection.
    rank:
        Positive integer CP rank, uniform TT bond rank, or uniform Tucker
        multilinear rank.  It is retained as metadata for ``dense``.

    Notes
    -----
    CP, TT, and Tucker use eight independent scalar-axis CAFE+ blocks: input,
    output, ``k_x``, and ``k_y`` for each of the real and imaginary tensors.
    TT follows ``C_in -> k_x -> k_y -> C_out``.  Tucker uses independent
    learned real/imaginary ``R^4`` cores.

    Accepted input layouts are ``[B,C_in,H,W]`` and ``[B,H,W,C_in]``.  Integer
    padding applies equally to height and width; a pair ``(pad_h,pad_w)``
    supports asymmetric axis padding.  FFT shape is inferred each forward.
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
        tensorization: Optional[str] = None,
        factor_rff_basis: int = 16,
        factor_cheb_basis: int = 8,
        factor_branch_dim: int = 12,
        factor_hidden_dim: int = 16,
        factor_output_init_mode: str = "xavier",
        factor_kernel_target_rms: float = 1e-3,
        # Legacy aliases; resolution is inferred at every forward pass.
        H: Optional[int] = None,
        W: Optional[int] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
    ) -> None:
        super().__init__()
        del H, W
        if m is not None:
            rff_basis = int(m)
        if n is not None:
            cheb_basis = int(n)

        if min(width, input_dim, output_dim, num_layers, ffn_expansion) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if input_layout not in {"auto", "channels_first", "channels_last"}:
            raise ValueError("unsupported input_layout")
        if output_layout not in {
            "match_input",
            "channels_first",
            "channels_last",
        }:
            raise ValueError("unsupported output_layout")
        if not 0.0 <= float(mlp_dropout) < 1.0:
            raise ValueError("mlp_dropout must be in [0, 1)")

        self.spatial_dim = 2
        self.width = int(width)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.padding = _normalize_padding(padding)
        self.mlp_dropout = float(mlp_dropout)
        self.add_grid = bool(add_grid)
        self.num_layers = int(num_layers)
        self.ffn_expansion = int(ffn_expansion)
        self.rff_basis = int(rff_basis)
        self.cheb_basis = int(cheb_basis)
        self.cafe_branches = int(cafe_branches)
        self.cafe_branch_dim = cafe_branch_dim
        self.kernel_hidden_dim = int(kernel_hidden_dim)
        self.kernel_mlp_type = _canonical_kernel_mlp_type(kernel_mlp_type)
        self.kernel_activation = str(kernel_activation).lower()
        self.kernel_output_init_std = float(kernel_output_init_std)
        self.sigma_init = float(sigma_init)
        self.learnable_sigma = bool(learnable_sigma)
        self.enforce_hermitian = bool(enforce_hermitian)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.factorization = _canonical_factorization(factorization)
        self.rank = _validate_rank(rank)

        if variant is None:
            self.variant = self.factorization
        else:
            normalized_variant = str(variant).strip().lower()
            if normalized_variant != self.factorization:
                raise ValueError(
                    "variant must match factorization; "
                    f"got variant={variant!r}, factorization={factorization!r}"
                )
            self.variant = normalized_variant
        supported_tensorizations = {
            JOINT_MODE_TENSORIZATION,
            FUNCTIONAL_AXIS_TENSORIZATION,
        }
        if tensorization is not None and tensorization not in supported_tensorizations:
            raise ValueError(
                "unsupported tensorization metadata: "
                f"{tensorization!r}"
            )
        self.tensorization = (
            JOINT_MODE_TENSORIZATION
            if self.factorization == "dense"
            else FUNCTIONAL_AXIS_TENSORIZATION
        )

        if self.factorization == "dense":
            self.factor_rff_basis = None
            self.factor_cheb_basis = None
            self.factor_embedding_dim = None
            self.factor_branch_dim = None
            self.factor_hidden_dim = None
            self.factor_output_init_mode = None
            self.factor_kernel_target_rms = None
        else:
            self.factor_rff_basis = _validate_rank(factor_rff_basis)
            self.factor_cheb_basis = _validate_rank(factor_cheb_basis)
            self.factor_embedding_dim = (
                2 * self.factor_rff_basis + self.factor_cheb_basis
            )
            self.factor_branch_dim = _validate_rank(factor_branch_dim)
            self.factor_hidden_dim = _validate_rank(factor_hidden_dim)
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

        if self.factorization == "dense":
            self.embedding_dim = 2 * self.rff_basis + 2 * self.cheb_basis
            self.branch_dim = (
                self.embedding_dim
                if cafe_branch_dim is None
                else int(cafe_branch_dim)
            )
        else:
            self.embedding_dim = int(self.factor_embedding_dim)
            self.branch_dim = int(self.factor_branch_dim)
        if self.branch_dim <= 0:
            raise ValueError("cafe_branch_dim must be positive")

        lifting_channels = self.input_dim + (2 if self.add_grid else 0)
        self.lifting = nn.Conv2d(lifting_channels, self.width, kernel_size=1)
        self.operator_layers = nn.ModuleList(
            [
                _CAFEPlusFNOOperatorLayer2D(
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
                    factorization=self.factorization,
                    rank=self.rank,
                    factor_rff_basis=factor_rff_basis,
                    factor_cheb_basis=factor_cheb_basis,
                    factor_branch_dim=factor_branch_dim,
                    factor_hidden_dim=factor_hidden_dim,
                    factor_output_init_mode=factor_output_init_mode,
                    factor_kernel_target_rms=factor_kernel_target_rms,
                    ffn_expansion=self.ffn_expansion,
                    dropout=self.mlp_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.projection = PointwiseMLP2D(
            spatial_dim=2,
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
        if x.ndim != 4:
            raise ValueError(f"expected a rank-4 tensor, got {tuple(x.shape)}")
        layout = self._infer_input_layout(x)
        if layout == "channels_last":
            return x.permute(0, 3, 1, 2).contiguous(), layout
        return x, layout

    def _format_output(self, y: Tensor, input_layout: str) -> Tensor:
        layout = self.output_layout
        if layout == "match_input":
            layout = input_layout
        if layout == "channels_last":
            return y.permute(0, 2, 3, 1).contiguous()
        return y

    @staticmethod
    def _make_grid(
        batch: int,
        spatial_shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        height, width = (int(value) for value in spatial_shape)
        xs = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        ys = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        mesh = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack(mesh, dim=0).unsqueeze(0).expand(
            batch, -1, height, width
        )

    @staticmethod
    def _prepare_grid(grid: Tensor, x: Tensor) -> Tensor:
        if grid.ndim != 4:
            raise ValueError("grid must be rank 4")
        channel_first = (
            grid.shape[1] == 2 and tuple(grid.shape[2:]) == tuple(x.shape[2:])
        )
        channel_last = (
            grid.shape[-1] == 2 and tuple(grid.shape[1:-1]) == tuple(x.shape[2:])
        )
        if channel_first and channel_last:
            raise ValueError(
                f"ambiguous coordinate axis for grid shape {tuple(grid.shape)}"
            )
        if channel_first:
            grid_cf = grid
        elif channel_last:
            grid_cf = grid.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("grid must have exactly two coordinate channels")
        if grid_cf.shape[0] != x.shape[0]:
            raise ValueError(
                "grid and input must have the same batch size; "
                f"got {grid_cf.shape[0]} and {x.shape[0]}"
            )
        if tuple(grid_cf.shape[2:]) != tuple(x.shape[2:]):
            raise ValueError(
                "grid and input must have the same spatial shape; "
                f"got {tuple(grid_cf.shape[2:])} and {tuple(x.shape[2:])}"
            )
        return grid_cf.to(device=x.device, dtype=x.dtype)

    def _pad(self, x: Tensor) -> Tensor:
        if not any(self.padding):
            return x
        return F.pad(x, (0, self.padding[1], 0, self.padding[0]))

    def _crop(self, x: Tensor) -> Tensor:
        if not any(self.padding):
            return x
        height_slice = slice(None, -self.padding[0]) if self.padding[0] else slice(None)
        width_slice = slice(None, -self.padding[1]) if self.padding[1] else slice(None)
        return x[..., height_slice, width_slice]

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
        """Count parameters belonging only to kernel generators.

        Spectral biases, pointwise FFNs, lifting, and projection parameters are
        intentionally excluded.  No kernel generators are shared between
        operator layers, so summing layer-local counts does not double count.
        """

        total = 0
        for layer in self.operator_layers:
            parameters: Iterable[nn.Parameter] = (
                layer.spectral.kernel_generator.parameters()
            )
            if trainable_only:
                parameters = (
                    parameter
                    for parameter in parameters
                    if parameter.requires_grad
                )
            total += sum(parameter.numel() for parameter in parameters)
        return total

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
            "spatial_dim": 2,
            "factorization": self.factorization,
            "rank": self.rank,
            "variant": self.variant,
            "tensorization": self.tensorization,
            "config": self.get_config(),
            "metadata": self.architecture_summary(),
            "state_dict": self.state_dict(),
        }

    def architecture_summary(self) -> Dict[str, object]:
        """Return architecture metadata suitable for logs and README tables."""

        generator = self.operator_layers[0].spectral.kernel_generator
        total_parameters = self.count_parameters()
        kernel_parameters = self.count_kernel_parameters()
        return {
            "model": self.__class__.__name__,
            "display_name": MODEL_NAME,
            "registry_key": MODEL_REGISTRY_KEY,
            "implementation_version": __version__,
            "config": self.get_config(),
            "spatial_dim": 2,
            "variant": self.variant,
            "factorization": self.factorization,
            "rank": self.rank,
            "tensorization": self.tensorization,
            "factorization_source": (
                "SirenFNO functional tensor decomposition"
                if self.factorization != "dense"
                else "canonical CAFE+FNO dense head"
            ),
            "factor_generator": "CAFE+",
            "decomposition_axes": ("C_in", "C_out", "k_x", "k_y"),
            "tt_mode_order": (
                "C_in",
                "k_x",
                "k_y",
                "C_out",
            ),
            "factor_axis_coordinate_convention": (
                "((0.5 + arange(N)) / N) * 2 - 1"
                if self.factorization != "dense"
                else None
            ),
            "real_imag_parameterization": (
                "one joint dense MLP"
                if self.factorization == "dense"
                else "eight independent CAFE+ blocks and parameters"
            ),
            **generator.rff_rng_metadata,
            "kernel_mlp_type": self.kernel_mlp_type,
            "kernel_mlps_per_layer": generator.num_kernel_mlps,
            "kernel_head_output_per_axis": generator.head_output_dim,
            "full_frequency": True,
            "full_kernel_materialized_before_spectral_mixing": True,
            "claims_reduced_spectral_flops_or_kernel_memory": False,
            "operator_layers": self.num_layers,
            "width": self.width,
            "rff_basis_m": (
                self.rff_basis
                if self.factorization == "dense"
                else self.factor_rff_basis
            ),
            "cheb_basis_n_per_axis": (
                self.cheb_basis
                if self.factorization == "dense"
                else self.factor_cheb_basis
            ),
            "embedding_dim": self.embedding_dim,
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
            "tucker_core_init_std": (
                1.0 / math.sqrt(self.rank)
                if self.factorization == "tucker"
                else None
            ),
            "learnable_sigma": self.learnable_sigma,
            "sigma_init": self.sigma_init,
            "enforce_hermitian": self.enforce_hermitian,
            "parameters": total_parameters,
            "kernel_generator_trainable_parameters": kernel_parameters,
            "non_kernel_generator_trainable_parameters": (
                total_parameters - kernel_parameters
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
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("spatial_shape must contain two positive values")
        fft_shape = tuple(size + pad for size, pad in zip(shape, self.padding))
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
        y: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        del y  # Trainer targets never enter the model prediction.

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
        spatial_shape = tuple(int(value) for value in x.shape[-2:])
        if any(value <= 0 for value in spatial_shape):
            raise ValueError(f"spatial dimensions must be positive, got {spatial_shape}")

        if self.add_grid:
            grid_cf = (
                self._make_grid(
                    x.shape[0],
                    spatial_shape,
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


def build_cafe_plus_fno_2d(**kwargs) -> CAFEPlusFNO2D:
    """Build :class:`CAFEPlusFNO2D` from keyword configuration."""

    return CAFEPlusFNO2D(**kwargs)


__all__ = [
    "MODEL_NAME",
    "MODEL_REGISTRY_KEY",
    "JOINT_MODE_TENSORIZATION",
    "CAFEPlusFNO2D",
    "build_cafe_plus_fno_2d",
]
