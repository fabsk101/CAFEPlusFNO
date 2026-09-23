import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.einsum_utils import einsum_complexhalf
from neuralop.layers.resample import resample

use_opt_einsum()

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    """
    x: (B, C_in, F1, ..., Fd)
    w: (C_in, C_out, F1, ..., Fd)  -> (B, C_out, F1, ..., Fd)
    """
    order = x.ndim
    x_syms = list(einsum_symbols[:order])   
    w_syms = list(x_syms[1:])                  
    if not separable:
        w_syms.insert(1, einsum_symbols[order])  
    out_syms = w_syms.copy()
    out_syms[0] = x_syms[0]                 
    eq = f"{''.join(x_syms)},{''.join(w_syms)}->{''.join(out_syms)}"

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
  
    if weight.device != x.device:
        weight = weight.to(x.device)
    if x.is_complex() and weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    elif (not x.is_complex()) and (weight.dtype != x.dtype):
        weight = weight.to(x.dtype)

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, weight)
    return torch.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    if weight.device != x.device:
        weight = weight.to(x.device)
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    return x * weight


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    if implementation == "reconstructed":
        return _contract_dense_separable if separable else _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight) or \
           (isinstance(weight, FactorizedTensor) and weight.name.lower().endswith(('dense', 'tucker', 'tt', 'cp'))):
            return _contract_dense
        else:
            raise ValueError(f"Unexpected weight type {type(weight)}")
    else:
        raise ValueError(f"Invalid implementation {implementation}")



def _reshape_vec(y: torch.Tensor, rank: int, length: int) -> torch.Tensor:
    if y.numel() != rank * length:
        raise RuntimeError(f"Size mismatch: expect {rank*length}, got {y.numel()}")
    return y.reshape(rank, length)

def _reshape_mat(y: torch.Tensor, r0: int, r1: int, length: int) -> torch.Tensor:
    if y.numel() != r0 * r1 * length:
        raise RuntimeError(f"Size mismatch: expect {r0*r1*length}, got {y.numel()}")
    return y.reshape(r0, r1, length).permute(0, 2, 1)



def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class mul_omega(nn.Module):
    def __init__(self, feature_dim, omega, omega_learnable=False):
        super().__init__()
        if omega_learnable:
            init = torch.rand((1, 1, feature_dim)) * omega
            self.omega = nn.Parameter(init)
        else:
            self.omega = omega

    def forward(self, x):
        return x * self.omega


class Periodic_activation(nn.Module):
    def __init__(self, nl="sin"):
        super().__init__()
        self.nl = nl

    def forward(self, x):
        if self.nl == "sin":
            return torch.sin(x)
        elif self.nl == "cos":
            return torch.cos(x)
        elif self.nl == "mix":
            shape = list(x.shape)
            x_sin = torch.sin(x[..., ::2])
            x_cos = torch.cos(x[..., 1::2])
            return torch.stack((x_sin, x_cos), dim=-1).view(*shape[:-1], -1)
        else:
            return x


def shape2coordinate(spatial_shape, batch_size, min_value=-1.0, max_value=1.0, upsample_ratio=1, device=None):
    coords = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
        _coords = min_value + (max_value - min_value) * _coords
        coords.append(_coords)
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords, dim=-1)
    ones_like_shape = (1,) * coords.ndim
    coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    return coords


class CoordSampler(nn.Module):
    """Generate normalized coordinates for up to ndims dimensions."""

    def __init__(self, ndims=1):
        super().__init__()
        self.coord_range = [-1, 1]
        self.ndims = ndims

    def base_sampler(self, L, coord_range=None, upsample_ratio=1.0, device=None):
        if isinstance(L, int):
            spatial_shape = [L] * self.ndims
        elif isinstance(L, (list, tuple)) and len(L) == self.ndims:
            spatial_shape = L
        else:
            raise AssertionError(f"L must be int or {self.ndims}-tuple (got {L!r})")

        coord_range = self.coord_range if coord_range is None else coord_range
        min_value, max_value = coord_range
        batch_size = 1
        return shape2coordinate(spatial_shape, batch_size, min_value, max_value, upsample_ratio, device)

    def forward(self, L=None, coord_range=None, upsample_ratio=1.0, device=None):
        return self.base_sampler(L, coord_range, upsample_ratio, device)


class fourier_mapping(nn.Module):
    def __init__(
        self,
        ff_dim,
        input_dim=1,
        ff_sigma=128.0,
        learnable_ff=True,
        ff_type="gaussian",
    ):
        super().__init__()
        if ff_dim % 2 != 0:
            raise AssertionError("ff_dim must be even")

        self.ff_dim_half = ff_dim // 2
        self.ff_sigma = ff_sigma
        self.input_dim = input_dim
        self.ff_type = ff_type

        if ff_type == "deterministic":
            ff_linear = 2 ** torch.linspace(0, self.ff_sigma, self.ff_dim_half // input_dim)
        elif ff_type == "deterministic_exp":
            log_freqs = torch.linspace(0, np.log(self.ff_sigma), self.ff_dim_half // input_dim)
            ff_linear = torch.exp(log_freqs)
        elif ff_type == "gaussian":
            ff_linear = torch.randn(input_dim, self.ff_dim_half) * self.ff_sigma
        else:
            raise ValueError(f"Unsupported ff_type {ff_type}")

        self.ff_linear = nn.Parameter(ff_linear, requires_grad=learnable_ff)
        self.coord_sampler = CoordSampler(ndims=self.input_dim)

    def forward(self, L, dev=None):
        coord = self.coord_sampler(L=L, device=dev)
        if self.ff_type in {"deterministic", "deterministic_exp"}:
            fourier_features = torch.matmul(coord, self.ff_linear.unsqueeze(0))
            fourier_features = fourier_features.view(1, coord.shape[1], -1)
        else:
            fourier_features = torch.matmul(coord, self.ff_linear)
        if self.ff_type != "deterministic":
            fourier_features = fourier_features * np.pi
        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        return torch.cat(fourier_features, dim=-1)


class Siren_block(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim,
        input_dim=1,
        omega=30.0,
        siren_dim_in=16,
        midlayer_num=1,
        default_init=False,
        ff_sigma=512.0,
        nl="mix",
    ):
        super().__init__()
        self.default_init = default_init
        self.omega = omega
        self.dim_in = siren_dim_in
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.ff_sigma = ff_sigma

        self.FF_mapping = fourier_mapping(
            ff_dim=siren_dim_in,
            input_dim=input_dim,
            ff_sigma=self.ff_sigma,
            learnable_ff=True,
            ff_type="gaussian",
        )
        self.phi_init = nn.Sequential(
            nn.Linear(siren_dim_in, hidden_dim, bias=False),
            mul_omega(hidden_dim, omega, True),
            Periodic_activation(nl=nl),
        )
        self.phi_mid = clones(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                mul_omega(hidden_dim, omega, True),
                Periodic_activation(nl=nl),
            ),
            midlayer_num,
        )
        self.phi_last = nn.Linear(hidden_dim, out_dim, bias=False)
        self.siren_initialization()

    def forward(self, L, dev=None):
        t = self.FF_mapping(L=L, dev=dev)
        t = self.phi_init(t)
        for midlayer in self.phi_mid:
            t = midlayer(t)
        return self.phi_last(t)

    def init_firstLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.weight is not None and not self.default_init:
            m.weight.data.uniform_(-1 / self.dim_in, 1 / self.dim_in)
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
            m.bias.data.zero_()

    def init_midLayers(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.weight is not None and not self.default_init:
            bound = np.sqrt(6.0 / self.dim_in) / self.omega
            m.weight.data.uniform_(-bound, bound)
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
            m.bias.data.zero_()

    def init_lastLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.weight is not None and not self.default_init:
            bound = np.sqrt(6.0 / self.dim_in) / self.omega
            m.weight.data.uniform_(-bound, bound)
        if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
            m.bias.data.zero_()

    def siren_initialization(self):
        for name, module in self.named_children():
            if name == "phi_init":
                module.apply(self.init_firstLayer)
            elif name == "phi_mid":
                self.dim_in = self.hidden_dim
                module.apply(self.init_midLayers)
            elif name == "phi_last":
                self.dim_in = self.hidden_dim
                module.apply(self.init_lastLayer)



class SirenHyperConv(BaseSpectralConv):
    """Drop-in spectral convolution for ``neuralop.models.FNO``.

    Implements the SirenFNO spectral block as a :class:`BaseSpectralConv`
    subclass so it can be passed to ``FNO(..., conv_module=SirenHyperConv)``
    for ablation experiments that share the rest of the neuraloperator FNO
    architecture (lifting, projection, skip connections). Per-mode complex
    weights are generated by a SIREN MLP conditioned on Fourier-feature-encoded
    mode indices, optionally tensor-factorized via tltorch.

    Args:
        in_channels / out_channels: Channel counts.
        n_modes: Tuple of retained Fourier modes per spatial dimension.
        factorization, rank: ``None`` / "cp" / "tt" / "tucker" (with rank).
        hidden_dim, omega, siren_dim_in: SIREN / RFF hyperparameters.
        resolution_scaling_factor: Optional zero-shot super-resolution factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        factorization: Optional[str] = None,
        rank: int = 8,
        hidden_dim: int = 32,
        omega: float = 30.0,
        siren_dim_in: int = 16,
        resolution_scaling_factor: Optional[Union[float, List[float]]] = None,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)
        self.n_modes = list(n_modes)
        self.factorization = factorization
        self.rank = rank
        self.resolution_scaling_factor = (
            [resolution_scaling_factor] * self.order
            if isinstance(resolution_scaling_factor, (int, float))
            else resolution_scaling_factor
        )
        self._contract = _contract_dense

 
        if factorization is None:
            self.siren_full_real = Siren_block(
                hidden_dim, in_channels * out_channels, self.order, omega, siren_dim_in
            )
            self.siren_full_imag = Siren_block(
                hidden_dim, in_channels * out_channels, self.order, omega, siren_dim_in
            )

        elif factorization.lower() == 'cp':
            self.siren_in_real  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_in_imag  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_out_real = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_out_imag = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_branches_real = nn.ModuleList([
                Siren_block(hidden_dim, rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])
            self.siren_branches_imag = nn.ModuleList([
                Siren_block(hidden_dim, rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])

        elif factorization.lower() == 'tt':
            self.siren_in_real  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_in_imag  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_branches_real = nn.ModuleList([
                Siren_block(hidden_dim, rank * rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])
            self.siren_branches_imag = nn.ModuleList([
                Siren_block(hidden_dim, rank * rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])
            self.siren_out_real = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_out_imag = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.tt_ranks = [1] + [rank] * self.order + [1]

        elif factorization.lower() == 'tucker':
            self.siren_in_real  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_in_imag  = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_out_real = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_out_imag = Siren_block(hidden_dim, rank, 1, omega, siren_dim_in)
            self.siren_branches_real = nn.ModuleList([
                Siren_block(hidden_dim, rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])
            self.siren_branches_imag = nn.ModuleList([
                Siren_block(hidden_dim, rank, 1, omega, siren_dim_in) for _ in range(self.order)
            ])
            self.core_real = nn.Parameter(torch.randn(*([rank] * (self.order + 2))))
            self.core_imag = nn.Parameter(torch.randn(*([rank] * (self.order + 2))))

        else:
            raise ValueError(f"Unsupported factorization={factorization}")

        self.bias = nn.Parameter(torch.zeros(out_channels, *([1] * self.order)))

    def transform(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None):
        in_shape = list(x.shape[2:])
        if self.resolution_scaling_factor and output_shape is None:
            out_shape = tuple(round(s * r) for s, r in zip(in_shape, self.resolution_scaling_factor))
        elif output_shape:
            out_shape = output_shape
        else:
            out_shape = tuple(in_shape)
        return x if in_shape == list(out_shape) else resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None):
        batch, channels, *mode_sizes = x.shape
        device = x.device
        freq_sizes = mode_sizes.copy()
        freq_sizes[-1] = freq_sizes[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        x_fft = torch.fft.rfftn(x, dim=fft_dims)

        if self.factorization is None:
            wr = self.siren_full_real(L=freq_sizes, dev=device)
            wi = self.siren_full_imag(L=freq_sizes, dev=device)
            wr = wr.reshape(self.in_channels, self.out_channels, *freq_sizes)
            wi = wi.reshape(self.in_channels, self.out_channels, *freq_sizes)
            weight = torch.complex(wr, wi)

        elif self.factorization.lower() == 'cp':
            fin_r = self.siren_in_real(L=[self.in_channels], dev=device)
            fin_i = self.siren_in_imag(L=[self.in_channels], dev=device)
            A_in_r = _reshape_vec(fin_r, self.rank, self.in_channels)
            A_in_i = _reshape_vec(fin_i, self.rank, self.in_channels)

            fout_r = self.siren_out_real(L=[self.out_channels], dev=device)
            fout_i = self.siren_out_imag(L=[self.out_channels], dev=device)
            A_out_r = _reshape_vec(fout_r, self.rank, self.out_channels)
            A_out_i = _reshape_vec(fout_i, self.rank, self.out_channels)

            G_r, G_i = [], []
            for Lk, s_r, s_i in zip(freq_sizes, self.siren_branches_real, self.siren_branches_imag):
                fk_r = s_r(L=[Lk], dev=device)
                fk_i = s_i(L=[Lk], dev=device)
                G_r.append(_reshape_vec(fk_r, self.rank, Lk))
                G_i.append(_reshape_vec(fk_i, self.rank, Lk))

            r = einsum_symbols[0]
            dinds = list(einsum_symbols[1:1 + self.order])
            lhs = [f'{r}i', f'{r}j'] + [f'{r}{d}' for d in dinds]
            rhs = 'ij' + ''.join(dinds)
            weight_real = torch.einsum(','.join(lhs) + '->' + rhs, A_in_r, A_out_r, *G_r)
            weight_imag = torch.einsum(','.join(lhs) + '->' + rhs, A_in_i, A_out_i, *G_i)
            weight = torch.complex(weight_real, weight_imag)

        elif self.factorization.lower() == 'tt':
            f0_r = self.siren_in_real(L=[self.in_channels], dev=device)
            f0_i = self.siren_in_imag(L=[self.in_channels], dev=device)
            cin_r = _reshape_vec(f0_r, self.rank, self.in_channels)
            cin_i = _reshape_vec(f0_i, self.rank, self.in_channels)
            core0_r = cin_r.mT.unsqueeze(0)
            core0_i = cin_i.mT.unsqueeze(0)

            cores_r = [core0_r]
            cores_i = [core0_i]

            for Lk, s_r, s_i in zip(freq_sizes, self.siren_branches_real, self.siren_branches_imag):
                fk_r = s_r(L=[Lk], dev=device) 
                fk_i = s_i(L=[Lk], dev=device)
                cores_r.append(_reshape_mat(fk_r, self.rank, self.rank, Lk))
                cores_i.append(_reshape_mat(fk_i, self.rank, self.rank, Lk))

            fd_r = self.siren_out_real(L=[self.out_channels], dev=device)
            fd_i = self.siren_out_imag(L=[self.out_channels], dev=device)
            rout_r = _reshape_vec(fd_r, self.rank, self.out_channels)
            rout_i = _reshape_vec(fd_i, self.rank, self.out_channels)
            core_last_r = rout_r.unsqueeze(-1)
            core_last_i = rout_i.unsqueeze(-1)
            cores_r.append(core_last_r)
            cores_i.append(core_last_i)

            K = len(cores_r)
            rinds = einsum_symbols[:K + 1]
            dinds = einsum_symbols[K + 1:K + 1 + K]
            terms = [f'{rinds[i]}{dinds[i]}{rinds[i+1]}' for i in range(K)]
            expr = ','.join(terms) + '->' + ''.join(dinds)

            full_w_r = torch.einsum(expr, *cores_r)
            full_w_i = torch.einsum(expr, *cores_i)
            weight_real = full_w_r.permute(0, -1, *range(1, self.order + 1))
            weight_imag = full_w_i.permute(0, -1, *range(1, self.order + 1))
            weight = torch.complex(weight_real, weight_imag)

        elif self.factorization.lower() == 'tucker':
            fin_r = self.siren_in_real(L=[self.in_channels], dev=device)
            fin_i = self.siren_in_imag(L=[self.in_channels], dev=device)
            A_in_r = _reshape_vec(fin_r, self.rank, self.in_channels)
            A_in_i = _reshape_vec(fin_i, self.rank, self.in_channels)

            fout_r = self.siren_out_real(L=[self.out_channels], dev=device)
            fout_i = self.siren_out_imag(L=[self.out_channels], dev=device)
            A_out_r = _reshape_vec(fout_r, self.rank, self.out_channels)
            A_out_i = _reshape_vec(fout_i, self.rank, self.out_channels)

            spatial_r, spatial_i = [], []
            for Lk, s_r, s_i in zip(freq_sizes, self.siren_branches_real, self.siren_branches_imag):
                fk_r = s_r(L=[Lk], dev=device)
                fk_i = s_i(L=[Lk], dev=device)
                spatial_r.append(_reshape_vec(fk_r, self.rank, Lk))
                spatial_i.append(_reshape_vec(fk_i, self.rank, Lk))

            core_r = self.core_real
            core_i = self.core_imag

            rank_inds = einsum_symbols[: self.order + 2]       
            dim_inds  = einsum_symbols[self.order + 2: self.order * 2 + 2]
            expr = (
                f"{''.join(rank_inds)},"
                f"{rank_inds[0]}i,"
                f"{rank_inds[1]}j,"
                + ",".join(f"{rank_inds[k + 2]}{dim_inds[k]}" for k in range(self.order))
                + f"->ij{''.join(dim_inds)}"
            )
            weight_real = torch.einsum(expr, core_r, A_in_r, A_out_r, *spatial_r)
            weight_imag = torch.einsum(expr, core_i, A_in_i, A_out_i, *spatial_i)
            weight = torch.complex(weight_real, weight_imag)  

        else:
            raise NotImplementedError(f"Unsupported factorization {self.factorization}")

        expected = (self.in_channels, self.out_channels, *freq_sizes)
        assert tuple(weight.shape) == expected, \
            f"weight shape {tuple(weight.shape)} != expected {expected}"

        out_fft = self._contract(x_fft, weight, separable=False)
        x_out = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims)
        return x_out + self.bias
