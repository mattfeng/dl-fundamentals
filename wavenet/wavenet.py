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

        assert conv_dim % 2 == 0, ("GLU requires conv_dim to be "
            "divisible by 2 (half are used as activations, and "
            "half are used as gates)")

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

    def forward(self, seq: Float[Tensor, "batch channels seq"]):
        dcc_o = self.dcc(seq)
        glu_o = self.glu(dcc_o)

        resid_o = self.residual_conv(glu_o)
        skip_o = self.skip_conv(glu_o)

        return seq + resid_o, skip_o


class WaveNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        depth: int = 3,
        conv_dim: int = 10,
        residual_dim: int = 3,
        skip_dim: int = 2,
        head_dim: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, residual_dim)
        self.blocks = nn.ModuleList([
            WaveNetBlock(2**d, conv_dim, residual_dim, skip_dim)
            for d in range(depth)
        ])
        self.conv1 = nn.Conv1d(skip_dim, head_dim, kernel_size=1)
        self.logits_conv = nn.Conv1d(head_dim, vocab_size, kernel_size=1)

    def forward(self, seq: Integer[Tensor, "batch seq"]) -> Float[Tensor, "batch seq logits"]:
        inp = self.embed(seq)
        inp = rearrange(inp, "b s c -> b c s")

        # TODO: just accumulate, don't store intermediate tensors
        skips = []
        for b in self.blocks:
            inp, skip = b(inp)
            skips.append(skip)

        wavenet_o = sum(skips)
        relu1_o = F.relu(wavenet_o)
        conv1_o = self.conv1(relu1_o)
        relu2_o = F.relu(conv1_o)
        logits = self.logits_conv(relu2_o)
        logits = rearrange(logits, "b c s -> b s c")

        return logits