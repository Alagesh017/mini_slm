import torch
import torch.nn as nn

from .attention import SelfAttention
from .feedforward import FeedForward
from .layernorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        num_heads=12,
        num_kv_heads=4,
        hidden_dim=1536,
        max_seq_len=512
    ):
        super().__init__()

        self.attention = SelfAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len
        )

        self.feedforward = FeedForward(
            embed_dim,
            hidden_dim
        )

        self.norm1 = RMSNorm(
            embed_dim
        )

        self.norm2 = RMSNorm(
            embed_dim
        )

    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        return x
