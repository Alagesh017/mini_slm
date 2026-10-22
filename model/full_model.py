import torch
import torch.nn as nn

from .embeddings import Embeddings
from .transformer import Transformer
from .layernorm import RMSNorm
from .mlm_head import MLMHead


class FullModel(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        num_layers=6,
        embed_dim=384,
        num_heads=12,
        num_kv_heads=4,
        hidden_dim=1536,
        max_seq_len=512,
        embedding_path=None
    ):
        super().__init__()

        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding_path=embedding_path
        )

        self.transformer = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len
        )

        self.final_norm = RMSNorm(embed_dim)

        self.lm_head = MLMHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.transformer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
