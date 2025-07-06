import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float
from einops import rearrange

# TODO:
# - https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

# d: latent dimension
# theta_i = 10000 ** (-2*(i-1)/d): rotation frequency
# m (position): rotation amount

class RotaryPositionalEmbedding(nn.Module):
    """
    Implementation from https://nn.labml.ai/transformers/rope/index.html.
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        d: number of features
        base: constant used to calculate rotation
        """
        self.d = d
        self.base = base
        self.sin_cached = None
        self.cos_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        # if we've already computed rotation frequencies and
        # the length of the input sequence is within those
        # frequencies, don't do it again
        if self.cos_cached is not None:
            if seq_len <= self.cos_cached.shape[0]:
                return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2, dtype=torch.float) / self.d))
        seq_idx = torch.arange(seq_len, device=device, dtype=torch.float)

        # outer product of seq_idx and theta
        # abbreviations: t = theta (angle)
        # 0 * t1, 0 * t2, 0 * t3, ...
        # 1 * t1, 1 * t2, 1 * t3, ...
        # 2 * t1, 2 * t2, 2 * t3, ...
        # ...
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    def _neg_half(self, x: Float[Tensor, "... features"]):
        """
        Converts
            [ x1   x2   x3   x4  |  y1  y2  y3  y4 ]
        into
            [-y1  -y2  -y3  -y4  |  x1  x2  x3  x4 ]
        in preparation for rotating feature pairs.
        """
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, S: Float[Tensor, "... tokens features"]):
        self._build_cache(S)

        # for sequence S[tokens][features],
        #   rotate coordinates given by feature pair
        #     (S[m][i], S[m][i + d/2])
        #   by
        #     m * t_i radians

        # rotation formula
        #   x' = xcosθ - ysinθ
        #   y' = ycosθ + xsinθ

        # So rotation will work like
        #   [ x1   x2   x3   x4  |  y1  y2  y3  y4 ] * cos
        # + [-y1  -y2  -y3  -y4  |  x1  x2  x3  x4 ] * sin
        # -------------------------------------------------
        #   [ x1'  x2'  x3'  x4' |  y1' y2' y3' y4']

        # if d < features, then only a portion of the features
        #   of S will be rotated/positionally embedded
        S_rope, S_pass = S[..., :self.d], S[..., self.d:]

        neg_half_S_rope = self._neg_half(S_rope)

        num_tokens = S.shape[-2]

        S_rope = (
            (S_rope * self.cos_cached[:num_tokens]) +
            (neg_half_S_rope * self.sin_cached[:num_tokens])
        )

        return torch.cat((S_rope, S_pass), dim=-1)