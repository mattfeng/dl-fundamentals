# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# typing
from typing import Callable
from torch import Tensor
from jaxtyping import Integer, Float

from einops import rearrange

class DilatedCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
            padding_mode="zeros"
        )
        self.dilation_ = dilation

    def forward(self, x):
        ret = super().forward(x)
        return ret[..., :-self.dilation_]


class GatedLinearUnit(nn.Module):
    # https://gist.github.com/hskim-solv/02b0782b56b2219ff9485f7baae5de59
    def __init__(
        self,
        activation_fn: Callable = F.tanh,
        channel_dim: int = -1,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.channel_dim = channel_dim

    def forward(self, x):
        x0, x1 = torch.chunk(x, 2, dim=self.channel_dim)

        act_o = self.activation_fn(x0)
        gate_o = F.sigmoid(x1)

        return act_o * gate_o


class WaveNetBlock(nn.Module):
    def __init__(
        self,
        dilation,
        conv_dim: int,
        residual_dim: int,
        skip_dim: int
    ):
        super().__init__()

        assert conv_dim % 2 == 0, "GLU requires conv_dim to be divisible by 2"

        self.dcc = DilatedCausalConv1d(
            residual_dim,
            conv_dim,
            dilation
        )
        self.glu = GatedLinearUnit(channel_dim=1)
        self.residual_conv = nn.Conv1d(
            conv_dim // 2,
            residual_dim,
            kernel_size=1
            )
        self.skip_conv = nn.Conv1d(
            conv_dim // 2,
            skip_dim,
            kernel_size=1
            )

    def forward(self, seq: Float[Tensor, "batch seq channels"]):
        seq = rearrange(seq, "b s c -> b c s")

        dcc_o = self.dcc(seq)
        glu_o = self.glu(dcc_o)

        resid_o = self.residual_conv(glu_o)
        skip_o = self.skip_conv(glu_o)

        return resid_o, skip_o


class WaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, seq: Integer[Tensor, "batch seq channels"]):
        pass