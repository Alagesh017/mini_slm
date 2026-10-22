import torch
import torch.nn as nn

from .encoder import DecoderBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        embed_dim=384,
        num_heads=12,
        num_kv_heads=4,
        hidden_dim=1536,
        max_seq_len=512
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                num_heads,
                num_kv_heads,
                hidden_dim,
                max_seq_len
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
