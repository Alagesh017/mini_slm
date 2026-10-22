import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=1536):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = x1 * torch.sigmoid(x1) * x2
        return self.w3(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=1536):
        super().__init__()
        self.swiglu = SwiGLU(embed_dim, hidden_dim)

    def forward(self, x):
        return self.swiglu(x)
