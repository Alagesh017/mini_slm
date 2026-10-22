import torch
import torch.nn as nn
import numpy as np


class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=384,
        embedding_path=None,
    ):
        super().__init__()
        if embedding_path:
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            embedding_path = os.path.join(project_root, embedding_path)
            embedding_matrix = np.load(embedding_path)
            self.token_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False
            )
        else:
            self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.token_embeddings(token_ids)
