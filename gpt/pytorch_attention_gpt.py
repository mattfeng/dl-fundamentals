#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor as T
from jaxtyping import Float


class PositionalEncoding(nn.Module):
    pass


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.mmha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: Float[T, "batch tokens {self.d_model}"]
    ):
        _, t, _ = x.shape
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        ln1_o = self.ln1(x)
        print(ln1_o.shape)
        attn_out, _ = self.mmha(ln1_o, ln1_o, ln1_o, attn_mask=causal_mask)
        mmha_o = x + attn_out
        print(mmha_o, mmha_o.shape)

        ln2_o = self.ln2(mmha_o)
        ffwd_o = mmha_o + self.ffwd(ln2_o)

        return ffwd_o


class GPT(nn.Module):
    def __init__(
        self,
        num_blocks
    ):
        pass

    def forward(self, x):
        pass