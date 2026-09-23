import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout=0.0):
        super(MLP1d, self).__init__()
        self.linear1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.linear2 = nn.Conv1d(mid_channels, out_channels, 1)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x


class SpectralConv1dMLP(nn.Module):
    def __init__(self, in_channels, out_channels, n1, dropout):
        super(SpectralConv1dMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1 = n1
        self.mlpxr = MLP1d(n1, in_channels * out_channels, 2 * n1, dropout=dropout)
        self.mlpxi = MLP1d(n1, in_channels * out_channels, 2 * n1, dropout=dropout)

        self.dropout = dropout

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, Tx):
        B, C, H = x.shape

        xr, xi = self.mlpxr(Tx), self.mlpxi(Tx)
        Re = xr.reshape(self.in_channels, self.out_channels, H // 2 + 1)
        Im = xi.reshape(self.in_channels, self.out_channels, H // 2 + 1)
        kernel = Re + 1j * Im

        out_ft = torch.zeros(B, C, H // 2 + 1)
        x_ft = torch.fft.rfft(x)
        out_ft = self.compl_mul1d(x_ft, kernel)

        x = torch.fft.irfft(out_ft, n=H)

        return x


class FNO1dMLP(nn.Module):
    def __init__(
        self, width, n1=10, padding=0, input_dim=1, output_dim=1, mlp_dropout=0.0, H=256
    ):
        super(FNO1dMLP, self).__init__()

        self.width = width
        self.padding = padding

        self.p = nn.Linear(input_dim + 1, self.width)  # (u, x) -> width
        self.conv0 = SpectralConv1dMLP(self.width, self.width, n1, dropout=mlp_dropout)
        self.conv1 = SpectralConv1dMLP(self.width, self.width, n1, dropout=mlp_dropout)
        self.conv2 = SpectralConv1dMLP(self.width, self.width, n1, dropout=mlp_dropout)
        self.conv3 = SpectralConv1dMLP(self.width, self.width, n1, dropout=mlp_dropout)
        self.mlp0 = MLP1d(self.width, self.width, 4 * self.width)
        self.mlp1 = MLP1d(self.width, self.width, 4 * self.width)
        self.mlp2 = MLP1d(self.width, self.width, 4 * self.width)
        self.mlp3 = MLP1d(self.width, self.width, 4 * self.width)

        self.n1 = n1
        self.grade1 = torch.arange(1, self.n1 + 1).reshape(self.n1, 1).float()
        self.gridx = torch.fft.rfftfreq(H + padding).unsqueeze(0)

        self.Tx = torch.zeros(self.n1, (H + padding) // 2 + 1)
        self.Tx = (
            (torch.cos(self.grade1 @ torch.acos(self.gridx)))
            .reshape(1, self.n1, (H + padding) // 2 + 1)
            .cuda()
        )

        self.q = nn.Conv1d(self.width, output_dim, 1)

    def forward(self, x, grid=None):
        x = x.permute(0, 2, 1)

        if grid is None:
            grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)  # [B, X, width]
        x = x.permute(0, 2, 1)  # [B, width, X]

        x = F.pad(x, [0, self.padding])
        B, C, H = x.shape

        grade1 = self.grade1.to(x.device)
        gridx = torch.fft.rfftfreq(H).to(x.device).unsqueeze(0)
        Tx = torch.cos(grade1 @ torch.acos(gridx)).reshape(1, self.n1, H // 2 + 1)

        x1 = self.conv0(x, Tx)
        x1 = self.mlp0(x1)
        x = x1 + x
        x = F.gelu(x)
        x1 = self.conv1(x, Tx)
        x1 = self.mlp1(x1)
        x = x1 + x
        x = F.gelu(x)
        x1 = self.conv2(x, Tx)
        x1 = self.mlp2(x1)
        x = x1 + x
        x = F.gelu(x)
        x1 = self.conv3(x, Tx)
        x1 = self.mlp3(x1)
        x = x1 + x

        if self.padding > 0:
            x = x[..., : -self.padding]

        x = self.q(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.linear2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x


class SpectralConv2dMLP(nn.Module):

    def __init__(self, in_channels, out_channels, n1, n2, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n1 = n1
        self.n2 = n2

        self.mlpxr = MLP(n1, in_channels * out_channels, 2 * n1, dropout=dropout)
        self.mlpxi = MLP(n1, in_channels * out_channels, 2 * n1, dropout=dropout)
        self.mlpyr = MLP(n2, in_channels * out_channels, 2 * n2, dropout=dropout)
        self.mlpyi = MLP(n2, in_channels * out_channels, 2 * n2, dropout=dropout)

    @torch.no_grad()
    def _cheb_basis(self, H, W, device, dtype):

        gridx = torch.fft.fftfreq(H, device=device, dtype=dtype).unsqueeze(
            0
        )
        gridy = torch.fft.rfftfreq(W, device=device, dtype=dtype).unsqueeze(
            0
        )
        
        thetax = torch.acos(torch.clamp(gridx, -1.0, 1.0))
        thetay = torch.acos(torch.clamp(gridy, -1.0, 1.0))
        
        grade1 = torch.arange(1, self.n1 + 1, device=device, dtype=dtype).view(
            -1, 1
        )
        grade2 = torch.arange(1, self.n2 + 1, device=device, dtype=dtype).view(
            -1, 1
        )
        
        Tx = torch.cos(grade1 @ thetax).reshape(1, self.n1, H, 1)
        Ty = torch.cos(grade2 @ thetay).reshape(
            1, self.n2, 1, W // 2 + 1
        )
        return Tx, Ty

    @staticmethod
    def _compl_mul2d(x_ft, w_ft):
        return torch.einsum("bchw,cohw->bohw", x_ft, w_ft)

    def forward(self, x):
        B, Cin, H, W = x.shape
        assert Cin == self.in_channels, (
            f"Expected {self.in_channels} input channels, got {Cin}"
        )

        x_ft = torch.fft.rfft2(x)

        Tx, Ty = self._cheb_basis(H, W, x.device, x.dtype)
        kx_r = self.mlpxr(Tx)
        kx_i = self.mlpxi(Tx)
        ky_r = self.mlpyr(Ty)
        ky_i = self.mlpyi(Ty)

        kx = (kx_r + 1j * kx_i).reshape(self.in_channels, self.out_channels, H, 1)
        ky = (ky_r + 1j * ky_i).reshape(
            self.in_channels, self.out_channels, 1, W // 2 + 1
        )
        kernel_ft = kx @ ky

        out_ft = self._compl_mul2d(x_ft, kernel_ft)
        y = torch.fft.irfft2(out_ft, s=(H, W))
        return y


class FNO2dMLP(nn.Module):
    def __init__(
        self,
        width: int,
        n1: int = 10,
        n2: int = 10,
        padding: int = 0,
        input_dim: int = 1,
        output_dim: int = 1,
        mlp_dropout: float = 0.0,
        add_grid: bool = True,
    ):
        super().__init__()
        self.width = width
        self.padding = int(padding)
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.add_grid = add_grid

        lift_in = self.in_channels + (2 if self.add_grid else 0)
        self.p = nn.Conv2d(lift_in, self.width, kernel_size=1)

        self.conv0 = SpectralConv2dMLP(
            self.width, self.width, n1, n2, dropout=mlp_dropout
        )
        self.conv1 = SpectralConv2dMLP(
            self.width, self.width, n1, n2, dropout=mlp_dropout
        )
        self.conv2 = SpectralConv2dMLP(
            self.width, self.width, n1, n2, dropout=mlp_dropout
        )
        self.conv3 = SpectralConv2dMLP(
            self.width, self.width, n1, n2, dropout=mlp_dropout
        )

        self.mlp0 = MLP(self.width, self.width, 4 * self.width, dropout=0.0)
        self.mlp1 = MLP(self.width, self.width, 4 * self.width, dropout=0.0)
        self.mlp2 = MLP(self.width, self.width, 4 * self.width, dropout=0.0)
        self.mlp3 = MLP(self.width, self.width, 4 * self.width, dropout=0.0)

        self.q = MLP(self.width, self.out_channels, 4 * self.width, dropout=0.0)

    @staticmethod
    def _make_grid(B, H, W, device, dtype):
        xs = (
            torch.linspace(0, 1, H, device=device, dtype=dtype)
            .view(1, 1, H, 1)
            .expand(B, 1, H, W)
        )
        ys = (
            torch.linspace(0, 1, W, device=device, dtype=dtype)
            .view(1, 1, 1, W)
            .expand(B, 1, H, W)
        )
        return torch.cat([xs, ys], dim=1)

    def forward(self, x=None, y=None, **kwargs):
        if x is None:
            if "x" in kwargs:
                x = kwargs["x"]
            else:
                for v in kwargs.values():
                    if torch.is_tensor(v):
                        x = v
                        break
            if x is None:
                raise TypeError("FNO2dMLP.forward expected argument 'x'")

        B, C, H, W = x.shape

        if getattr(self, "add_grid", False):
            if C == self.in_channels:
                grid = self._make_grid(B, H, W, x.device, x.dtype)
                x = torch.cat([x, grid], dim=1)
            elif C != self.in_channels + 2:
                raise ValueError(
                    f"Expected {self.in_channels} or {self.in_channels + 2} channels, got {C}"
                )

        x = self.p(x)

        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        x1 = self.mlp0(self.conv0(x))
        x = F.gelu(x + x1)
        x1 = self.mlp1(self.conv1(x))
        x = F.gelu(x + x1)
        x1 = self.mlp2(self.conv2(x))
        x = F.gelu(x + x1)
        x1 = self.mlp3(self.conv3(x))
        x = x + x1

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.q(x)
        return x
