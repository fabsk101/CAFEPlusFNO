import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class mul_omega(nn.Module):
    def __init__(self, feature_dim, omega, omega_learnable=True):
        super().__init__()
        self.omega = (
            nn.Parameter(torch.rand((1, 1, feature_dim)) * omega)
            if omega_learnable
            else omega
        )

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
            C = x.shape[-1]
            if C % 2 != 0:
                raise ValueError(
                    "Periodic_activation(nl='mix') requires even feature dim, got {}".format(
                        C
                    )
                )
            x_sin = torch.sin(x[..., ::2])  # (..., C/2)
            x_cos = torch.cos(x[..., 1::2])  # (..., C/2)
            return torch.stack((x_sin, x_cos), dim=-1).view(*x.shape[:-1], C)
        else:
            raise ValueError(f"Unknown periodic activation {self.nl}")


def shape2coordinate(
    spatial_shape,
    batch_size,
    min_value=-1.0,
    max_value=1.0,
    upsample_ratio=1.0,
    device=None,
):
    coords = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
        _coords = min_value + (max_value - min_value) * _coords
        coords.append(_coords)
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords, dim=-1)  # (..., d)
    coords = coords.unsqueeze(0).repeat(1, *([1] * coords.ndim))
    return coords


class CoordSampler(nn.Module):
    def __init__(self, ndims=1, coord_range=(-1.0, 1.0)):
        super().__init__()
        self.coord_range = coord_range
        self.ndims = ndims

    def forward(self, L=None, coord_range=None, upsample_ratio=1.0, device=None):
        assert isinstance(L, int) or (
            isinstance(L, (list, tuple)) and len(L) == self.ndims
        )
        coord_range = self.coord_range if coord_range is None else coord_range
        min_value, max_value = coord_range
        spatial_shape = [L] * self.ndims if isinstance(L, int) else L
        return shape2coordinate(
            spatial_shape, 1, min_value, max_value, upsample_ratio, device
        )


class fourier_mapping(nn.Module):

    def __init__(
        self,
        ff_dim,
        input_dim=1,
        ff_sigma=1024.0,
        learnable_ff=True,
        ff_type="gaussian",
    ):
        super().__init__()
        assert (ff_dim % 2) == 0
        self.ff_dim_half = ff_dim // 2
        self.ff_sigma = ff_sigma
        self.input_dim = input_dim
        self.ff_type = ff_type
        if ff_type == "deterministic":
            ff_linear = 2 ** torch.linspace(
                0, self.ff_sigma, self.ff_dim_half // input_dim
            )
        elif ff_type == "deterministic_exp":
            log_freqs = torch.linspace(
                0, np.log(self.ff_sigma), self.ff_dim_half // input_dim
            )
            ff_linear = torch.exp(log_freqs)
        elif ff_type == "gaussian":
            ff_linear = torch.randn(input_dim, self.ff_dim_half) * self.ff_sigma
        else:
            raise ValueError(f"Unknown ff_type {ff_type}")
        self.ff_linear = nn.Parameter(ff_linear, requires_grad=learnable_ff)
        self.coord_sampler = CoordSampler(ndims=input_dim)

    def forward(self, L, dev=None):
        param_dev = self.ff_linear.device
        param_dtype = self.ff_linear.dtype
        coord = self.coord_sampler(L=L, device=dev if dev is not None else param_dev)
        coord = coord.to(device=param_dev, dtype=param_dtype)
        if self.ff_type in ("deterministic", "deterministic_exp"):
            fourier_features = torch.matmul(coord, self.ff_linear.unsqueeze(0)).view(
                1, *coord.shape[1:-1], -1
            )
        elif self.ff_type == "gaussian":
            fourier_features = torch.matmul(coord, self.ff_linear)
        if self.ff_type != "deterministic":
            fourier_features = fourier_features * np.pi
        return torch.cat(
            [torch.cos(fourier_features), torch.sin(fourier_features)], dim=-1
        )


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
        ff_sigma=1024.0,
        nl="sin",
        learnable_ff=True,
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
            learnable_ff=learnable_ff,
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
            max(0, midlayer_num),
        )
        self.phi_last = nn.Linear(hidden_dim, out_dim, bias=False)
        self.siren_initialization()

    def forward(self, L, dev=None):
        t = self.FF_mapping(L=L, dev=dev)
        t = self.phi_init(t)
        for midlayer in self.phi_mid:
            t = midlayer(t)
        t = self.phi_last(t)
        return t
    
    
    def init_firstLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None and not self.default_init:
                m.weight.data.uniform_(-1 / self.dim_in, 1 / self.dim_in)
            if m.bias is not None:
                m.bias.data.zero_()

    def init_midLayers(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None and not self.default_init:
                m.weight.data.uniform_(
                    -math.sqrt(6.0 / self.dim_in) / self.omega,
                    math.sqrt(6.0 / self.dim_in) / self.omega,
                )
            if m.bias is not None:
                m.bias.data.zero_()

    def init_lastLayer(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.weight is not None and not self.default_init:
                m.weight.data.uniform_(
                    -math.sqrt(6.0 / self.dim_in) / self.omega,
                    math.sqrt(6.0 / self.dim_in) / self.omega,
                )
            if m.bias is not None:
                m.bias.data.zero_()

    def siren_initialization(self):
        for var, m in self.named_children():
            if var == "phi_init":
                m.apply(self.init_firstLayer)
            elif var == "phi_mid":
                self.dim_in = self.hidden_dim
                m.apply(self.init_midLayers)
            elif var == "phi_last":
                self.dim_in = self.hidden_dim
                m.apply(self.init_lastLayer)


class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.linear2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=True)
        self.dropout = float(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.linear2(x)
        return x


class SpectralConv1d_Siren(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        dropout: float = 0.0,
        siren_dim_in: int = 16,
        ff_sigma: float = 1024.0,
        learnable_ff: bool = True,
        factorization: str = "dense",
        rank: int = 16,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dropout = float(dropout)

        self.factorization = (factorization or "dense").lower()
        if self.factorization in ("none", "full"):
            self.factorization = "dense"
        assert self.factorization in ("dense", "cp", "tt", "tucker")
        self.rank = int(rank)
        mid_layers = max(0, int(n_hidden) - 1)

        if self.factorization == "dense":
            out_dim = self.in_channels * self.out_channels
            self.siren_full_r = Siren_block(
                hidden_dim,
                out_dim,
                input_dim=1,
                omega=omega,
                siren_dim_in=siren_dim_in,
                midlayer_num=mid_layers,
                ff_sigma=ff_sigma,
                nl="mix",
                learnable_ff=learnable_ff,
            )
            self.siren_full_i = Siren_block(
                hidden_dim,
                out_dim,
                input_dim=1,
                omega=omega,
                siren_dim_in=siren_dim_in,
                midlayer_num=mid_layers,
                ff_sigma=ff_sigma,
                nl="mix",
                learnable_ff=learnable_ff,
            )
        elif self.factorization == "cp":
            R = self.rank

            def make_chan():
                return Siren_block(
                    hidden_dim,
                    R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            def make_spatial():
                return Siren_block(
                    hidden_dim,
                    R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            self.cin_r = make_chan()
            self.cout_r = make_chan()
            self.cin_i = make_chan()
            self.cout_i = make_chan()
            self.sG_r = make_spatial()
            self.sG_i = make_spatial()
        elif self.factorization == "tt":
            R = self.rank

            def make_chan():
                return Siren_block(
                    hidden_dim,
                    R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            def make_spatial_tt():
                return Siren_block(
                    hidden_dim,
                    R * R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            self.cin_r = make_chan()
            self.cout_r = make_chan()
            self.cin_i = make_chan()
            self.cout_i = make_chan()
            self.sG_r = make_spatial_tt()
            self.sG_i = make_spatial_tt()
        elif self.factorization == "tucker":
            R = self.rank

            def make_chan():
                return Siren_block(
                    hidden_dim,
                    R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            def make_spatial():
                return Siren_block(
                    hidden_dim,
                    R,
                    input_dim=1,
                    omega=omega,
                    siren_dim_in=siren_dim_in,
                    midlayer_num=mid_layers,
                    ff_sigma=ff_sigma,
                    nl="mix",
                    learnable_ff=learnable_ff,
                )

            self.cin_r = make_chan()
            self.cout_r = make_chan()
            self.cin_i = make_chan()
            self.cout_i = make_chan()
            self.sG_r = make_spatial()
            self.sG_i = make_spatial()
            self.core_r = nn.Parameter(torch.randn(R, R, R) * (1.0 / math.sqrt(R)))
            self.core_i = nn.Parameter(torch.randn(R, R, R) * (1.0 / math.sqrt(R)))

        self.out_dropout = float(dropout)
        self.bias = nn.Parameter(torch.zeros(out_channels, 1))

    @staticmethod
    def _compl_mul1d(x_ft: torch.Tensor, w_ft: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcl,col->bol", x_ft, w_ft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, L = x.shape
        assert Cin == self.in_channels, f"Expected Cin={self.in_channels}, got {Cin}"
        Lc = L // 2 + 1

        x_ft = torch.fft.rfft(x, dim=-1)
        ftype = x.dtype

        if self.factorization == "dense":
            wr = (
                self.siren_full_r(L=[Lc], dev=x.device).to(ftype).squeeze(0)
            )
            wi = self.siren_full_i(L=[Lc], dev=x.device).to(ftype).squeeze(0)
            wr = wr.view(Lc, Cin, self.out_channels).permute(1, 2, 0)
            wi = wi.view(Lc, Cin, self.out_channels).permute(1, 2, 0)
            w_ft = torch.complex(wr, wi)

        elif self.factorization == "cp":
            R = self.rank
            A_in_r = self.cin_r(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            A_out_r = (
                self.cout_r(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_r = self.sG_r(L=[Lc], dev=x.device).to(ftype).squeeze(0)

            A_in_i = self.cin_i(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            A_out_i = (
                self.cout_i(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_i = self.sG_i(L=[Lc], dev=x.device).to(ftype).squeeze(0)

            w_r = torch.einsum("ir,jr,lr->ijl", A_in_r, A_out_r, G_r)
            w_i = torch.einsum("ir,jr,lr->ijl", A_in_i, A_out_i, G_i)
            w_ft = torch.complex(w_r, w_i)

        elif self.factorization == "tt":
            R = self.rank
            Cin_r = self.cin_r(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            Cout_r = (
                self.cout_r(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_r = (
                self.sG_r(L=[Lc], dev=x.device)
                .to(ftype)
                .squeeze(0)
                .reshape(Lc, R, R)
                .permute(1, 0, 2)
            )

            Cin_i = self.cin_i(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            Cout_i = (
                self.cout_i(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_i = (
                self.sG_i(L=[Lc], dev=x.device)
                .to(ftype)
                .squeeze(0)
                .reshape(Lc, R, R)
                .permute(1, 0, 2)
            )

            tmp_r = torch.einsum("ir,rlq->ilq", Cin_r, G_r)
            w_r = torch.einsum("ilq,qj->ijl", tmp_r, Cout_r.t())

            tmp_i = torch.einsum("ir,rlq->ilq", Cin_i, G_i)
            w_i = torch.einsum("ilq,qj->ijl", tmp_i, Cout_i.t())

            w_ft = torch.complex(w_r, w_i)

        elif self.factorization == "tucker":
            R = self.rank
            A_in_r = self.cin_r(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            A_out_r = (
                self.cout_r(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_r = self.sG_r(L=[Lc], dev=x.device).to(ftype).squeeze(0)
            core_r = self.core_r.to(ftype)

            A_in_i = self.cin_i(L=[Cin], dev=x.device).to(ftype).squeeze(0)
            A_out_i = (
                self.cout_i(L=[self.out_channels], dev=x.device).to(ftype).squeeze(0)
            )
            G_i = self.sG_i(L=[Lc], dev=x.device).to(ftype).squeeze(0)
            core_i = self.core_i.to(ftype)
            
            w_r = torch.einsum("abx,ia,jb,lx->ijl", core_r, A_in_r, A_out_r, G_r)
            w_i = torch.einsum("abx,ia,jb,lx->ijl", core_i, A_in_i, A_out_i, G_i)
            w_ft = torch.complex(w_r, w_i)

        else:
            raise RuntimeError("Unreachable")
        
        if w_ft.dtype != x_ft.dtype:
            w_ft = w_ft.to(dtype=x_ft.dtype)
        if w_ft.device != x_ft.device:
            w_ft = w_ft.to(x_ft.device)

        out_ft = self._compl_mul1d(x_ft, w_ft)
        y = torch.fft.irfft(out_ft, n=L, dim=-1)

        if self.out_dropout and self.training:
            y = F.dropout(y, p=self.out_dropout)
        return y + self.bias


class SirenFNO1d(nn.Module):
    """1D SirenFNO operator network.

    Replaces the per-mode complex weights of a vanilla 1D FNO with a SIREN MLP
    conditioned on Fourier-feature-encoded mode indices, enabling efficient
    full-frequency learning. See the SirenFNO paper for details.

    Args:
        width: Channel width inside spectral blocks.
        input_dim / output_dim: Input / output channel counts.
        padding: Domain padding (0 for periodic data).
        add_grid: If True, append a normalized coordinate channel to the input.
        hidden_dim, omega, n_hidden: SIREN hyperparameters.
        siren_dim_in, ff_sigma, learnable_ff: Random Fourier feature encoder
            configuration for mode indices.
        factorization, rank: Tensor-factorized weights via tltorch
            ("dense" / "cp" / "tt" / "tucker"). "dense" uses no factorization.

    Input shape: ``(B, input_dim, X)``. Output shape: ``(B, output_dim, X)``.
    """

    def __init__(
        self,
        width: int,
        input_dim: int = 1,
        output_dim: int = 1,
        padding: int = 0,
        mlp_dropout: float = 0.0,
        add_grid: bool = True,
        # SIREN params
        hidden_dim: int = 64,
        omega: float = 30.0,
        n_hidden: int = 1,
        # RFF params
        siren_dim_in: int = 32,
        ff_sigma: float = 1024.0,
        learnable_ff: bool = True,
        # Factorization
        factorization: str = "dense",
        rank: int = 16,
    ):
        super().__init__()
        self.width = int(width)
        self.in_channels = int(input_dim)
        self.out_channels = int(output_dim)
        self.add_grid = bool(add_grid)
        self.padding = int(padding)

        lift_in = self.in_channels + (1 if self.add_grid else 0)
        self.p = nn.Conv1d(lift_in, self.width, kernel_size=1, bias=True)

        def block():
            return SpectralConv1d_Siren(
                in_channels=self.width,
                out_channels=self.width,
                hidden_dim=hidden_dim,
                omega=omega,
                n_hidden=n_hidden,
                dropout=mlp_dropout,
                siren_dim_in=siren_dim_in,
                ff_sigma=ff_sigma,
                learnable_ff=learnable_ff,
                factorization=factorization,
                rank=rank,
            )

        self.conv0, self.mlp0 = (
            block(),
            MLP1d(self.width, self.width, 4 * self.width, dropout=mlp_dropout),
        )
        self.conv1, self.mlp1 = (
            block(),
            MLP1d(self.width, self.width, 4 * self.width, dropout=mlp_dropout),
        )
        self.conv2, self.mlp2 = (
            block(),
            MLP1d(self.width, self.width, 4 * self.width, dropout=mlp_dropout),
        )
        self.conv3, self.mlp3 = (
            block(),
            MLP1d(self.width, self.width, 4 * self.width, dropout=mlp_dropout),
        )

        self.q = MLP1d(self.width, self.out_channels, 4 * self.width, dropout=0.0)

    @staticmethod
    def _make_grid(B, L, device, dtype):
        xs = (
            torch.linspace(0.0, 1.0, L, device=device, dtype=dtype)
            .view(1, 1, L)
            .expand(B, 1, L)
        )
        return xs

    def forward(self, x=None, **kwargs):
        if x is None:
            if "x" in kwargs:
                x = kwargs["x"]
            else:
                raise TypeError("SirenFNO1d.forward expected argument 'x'")
        assert x.ndim == 3, f"Expect x in NCL, got {x.shape}"
        B, C, L = x.shape

        if self.add_grid:
            if C == self.in_channels:
                grid = self._make_grid(B, L, x.device, x.dtype)
                x = torch.cat([x, grid], dim=1)
            elif C != self.in_channels + 1:
                raise ValueError(
                    f"Expected {self.in_channels} or {self.in_channels + 1} channels, got {C}"
                )

        x = self.p(x)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        x1 = self.mlp0(self.conv0(x))
        x = F.gelu(x + x1)
        x1 = self.mlp1(self.conv1(x))
        x = F.gelu(x + x1)
        x1 = self.mlp2(self.conv2(x))
        x = F.gelu(x + x1)
        x1 = self.mlp3(self.conv3(x))
        x = x + x1

        if self.padding > 0:
            x = x[..., : -self.padding]

        x = self.q(x)
        return x
