from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND
from neuralop.layers.padding import DomainPadding
from neuralop.layers.spectral_convolution import SpectralConv


class U_net_1d(nn.Module):
    """
    1D U-Net used as a learned local refiner.
    Expects/returns (B, C, L), preserves length (with same padding).
    """

    def __init__(
        self, input_channels, output_channels, kernel_size=3, dropout_rate=0.0
    ):
        super().__init__()
        self.input_channels = input_channels

        self.conv1 = self._conv(
            input_channels,
            output_channels,
            kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = self._conv(
            input_channels,
            output_channels,
            kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self._conv(
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self._conv(
            input_channels,
            output_channels,
            kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self._conv(
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.deconv2 = self._deconv(input_channels, output_channels)
        self.deconv1 = self._deconv(input_channels * 2, output_channels)
        self.deconv0 = self._deconv(input_channels * 2, output_channels)

        self.output_layer = nn.Conv1d(
            input_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        # x: (B, C, L)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), dim=1)

        out = self.output_layer(concat0)
        return out  # (B, C, L)

    @staticmethod
    def _conv(in_planes, out_planes, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    @staticmethod
    def _deconv(in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_planes, out_planes, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def _make_norm_1d(
    kind: Optional[Literal["group_norm", "instance_norm", "batch_norm"]], C: int
) -> nn.Module:
    if kind is None:
        return nn.Identity()
    if kind == "batch_norm":
        return nn.BatchNorm1d(C)
    if kind == "instance_norm":
        return nn.InstanceNorm1d(C, affine=True)
    if kind == "group_norm":
        G = 32 if C >= 32 else max(1, C // 2)
        return nn.GroupNorm(G, C)
    raise ValueError(f"Unknown norm kind: {kind}")



class UFNOLayer1d(nn.Module):
    """
    One U-FNO layer (channels-first): y = act( Spectral(x) + PW(x) + UNet(x) ), with optional norm.
    - Spectral: either your SpectralConv1d or neuraloperator.SpectralConv
    - PW: 1x1 Conv1d (local channel mixing)
    - UNet: local multiscale refiner; internally pads L to multiple of 8, then crops back
    """

    def __init__(
        self,
        hidden_channels: int,
        modes: int,
        non_linearity=F.gelu,
        norm: Optional[Literal["group_norm", "instance_norm", "batch_norm"]] = None,
        conv_module: Optional[
            nn.Module
        ] = SpectralConv,
    ):
        super().__init__()
        C = hidden_channels
        self.non_linearity = non_linearity
        self.norm = _make_norm_1d(norm, C)

        if conv_module is SpectralConv or conv_module is None:
            self.spectral = SpectralConv(
                in_channels=C, out_channels=C, n_modes=(int(modes),)
            )
        else:

            self.spectral = conv_module(C, C, modes)

        # Local branches
        self.pointwise = nn.Conv1d(C, C, kernel_size=1, bias=True)
        self.unet = U_net_1d(C, C, kernel_size=3, dropout_rate=0.0)

    def forward(self, x):
        B, C, L = x.shape

        pad_right = (-L) % 8
        if pad_right:
            u_in = F.pad(x, (0, pad_right), mode="replicate")
        else:
            u_in = x
        u = self.unet(u_in)
        if pad_right:
            u = u[..., :L]

        y = self.spectral(x) + self.pointwise(x) + u
        y = self.norm(y)
        return self.non_linearity(y)



class UFNO1d(nn.Module):
    """
    1D U-FNO with FNO1d-style plumbing:
      (positional embedding) -> Lifting ChannelMLP -> [UFNO layers]^n -> Projection ChannelMLP

    Shapes: (B, C_in, L) -> (B, C_out, L)
    """

    def __init__(
        self,
        n_modes_height: int,
        hidden_channels: int,
        in_channels: int = 1,
        out_channels: int = 1,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        positional_embedding: Union[str, nn.Module, None] = "grid",
        domain_padding: Optional[Union[float, int, List[Union[float, int]]]] = None,
        resolution_scaling_factor: Optional[
            Union[float, int, List[Union[float, int]]]
        ] = None,
        non_linearity=F.gelu,
        norm: Optional[Literal["group_norm", "instance_norm", "batch_norm"]] = None,
        conv_module: Optional[
            nn.Module
        ] = SpectralConv,
    ):
        super().__init__()
        self.n_dim = 1
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self._n_modes = (int(n_modes_height),)
        self.non_linearity = non_linearity

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Unsupported positional_embedding: {positional_embedding}"
            )

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        lifting_in = self.in_channels + (
            self.n_dim if self.positional_embedding is not None else 0
        )
        self.lifting = ChannelMLP(
            in_channels=lifting_in,
            out_channels=self.hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        self.layers = nn.ModuleList(
            [
                UFNOLayer1d(
                    hidden_channels=self.hidden_channels,
                    modes=self._n_modes[0],
                    non_linearity=non_linearity,
                    norm=norm,
                    conv_module=conv_module,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    @property
    def n_modes(self) -> Tuple[int]:
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: Tuple[int]):
        assert isinstance(n_modes, tuple) and len(n_modes) == 1
        self._n_modes = (int(n_modes[0]),)
        for lyr in self.layers:
            if hasattr(lyr.spectral, "n_modes"):
                lyr.spectral.n_modes = self._n_modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_embedding is not None:
            x = self.positional_embedding(
                x
            )

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer in self.layers:
            x = layer(x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        y = self.projection(x)
        return y




class LocalUNet2D(nn.Module):
    def __init__(self, ch, drop=0.1):
        super().__init__()
        k=3; p=1
        def conv(in_c, out_c, s):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(drop)
            )
        def deconv(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.c1   = conv(ch, ch, 2)
        self.c2   = conv(ch, ch, 2)
        self.c2_1 = conv(ch, ch, 1)
        self.c3   = conv(ch, ch, 2)
        self.c3_1 = conv(ch, ch, 1)

        self.u2 = deconv(ch, ch)
        self.u1 = deconv(2*ch, ch)
        self.u0 = deconv(2*ch, ch)

        self.head = nn.Conv2d(2*ch, ch, k, padding=p)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2_1(self.c2(c1))
        c3 = self.c3_1(self.c3(c2))
        d2 = self.u2(c3)
        cat2 = torch.cat([c2, d2], dim=1)
        d1 = self.u1(cat2)
        cat1 = torch.cat([c1, d1], dim=1)
        d0 = self.u0(cat1)
        cat0 = torch.cat([x, d0], dim=1)
        return self.head(cat0)

class UFNOBlock2D(nn.Module):
    def __init__(self, hidden, n_modes, non_linearity=F.gelu,
                 use_channel_mlp=True, channel_mlp_expansion=0.5, channel_mlp_dropout=0.0,
                 use_unet=False, unet_dropout=0.1,
                 fno_skip='linear', fno_block_precision='full'):
        super().__init__()

        self.spec = SpectralConv(n_modes=n_modes,
                                 in_channels=hidden,
                                 out_channels=hidden,
                                 fno_block_precision=fno_block_precision)

        self.unet = LocalUNet2D(hidden, drop=unet_dropout) if use_unet else None

        self.use_mlp = use_channel_mlp
        if self.use_mlp:
            hidden_mlp = int(hidden * max(channel_mlp_expansion, 1e-6))
            self.mlp = ChannelMLP(in_channels=hidden,
                                  out_channels=hidden,
                                  hidden_channels=hidden_mlp,
                                  n_layers=1,
                                  n_dim=2,
                                  non_linearity=non_linearity,
                                  dropout=channel_mlp_dropout)
        self.act = non_linearity
        self.fno_skip = fno_skip

    def forward(self, x):
        y = self.spec(x)
        if self.unet is not None:
            y = y + self.unet(x)

        if self.use_mlp:
            y = y + self.mlp(x)

        x = self.act(y)

        return x

class UFNO(nn.Module):
    def __init__(self,
                 n_modes, 
                 in_channels, out_channels,
                 hidden_channels,
                 n_layers=6,
                 lifting_channel_ratio=2,
                 projection_channel_ratio=2,
                 positional_embedding="grid",
                 non_linearity=F.gelu,
                 use_channel_mlp=True,
                 channel_mlp_dropout=0.0,
                 channel_mlp_expansion=0.5,
                 fno_skip='linear',
                 resolution_scaling_factor=None,
                 domain_padding=None,
                 domain_padding_mode='symmetric',
                 fno_block_precision='full',
                 use_unet_from=3,
                 unet_dropout=0.1,
                 max_n_modes=None):
        super().__init__()
        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        if positional_embedding == "grid":
            grid_bounds = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels,
                                                        dim=self.n_dim,
                                                        grid_boundaries=grid_bounds)
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            self.positional_embedding = positional_embedding

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        lifting_in = in_channels + (self.n_dim if self.positional_embedding is not None else 0)
        self.lifting = ChannelMLP(in_channels=lifting_in,
                                  out_channels=hidden_channels,
                                  hidden_channels=int(lifting_channel_ratio*hidden_channels),
                                  n_layers=2, n_dim=self.n_dim,
                                  non_linearity=non_linearity)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            use_unet = (i >= use_unet_from)
            self.blocks.append(
                UFNOBlock2D(hidden=hidden_channels,
                            n_modes=self._n_modes,
                            non_linearity=non_linearity,
                            use_channel_mlp=use_channel_mlp,
                            channel_mlp_dropout=channel_mlp_dropout,
                            channel_mlp_expansion=channel_mlp_expansion,
                            use_unet=use_unet,
                            unet_dropout=unet_dropout,
                            fno_skip=fno_skip,
                            fno_block_precision=fno_block_precision)
            )

        self.projection = ChannelMLP(in_channels=hidden_channels,
                                     out_channels=out_channels,
                                     hidden_channels=int(projection_channel_ratio*hidden_channels),
                                     n_layers=2, n_dim=self.n_dim,
                                     non_linearity=non_linearity)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, new_modes):
        self._n_modes = new_modes
        for b in self.blocks:
            if hasattr(b.spec, "n_modes"):
                b.spec.n_modes = new_modes

    def forward(self, x=None, y=None, output_shape=None, **kwargs):
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        for blk in self.blocks:
            x = blk(x)
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        x = self.projection(x)
        return x