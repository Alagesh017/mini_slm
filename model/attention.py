import torch
import torch.nn as nn
import math


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    q = q.float()
    k = k.float()
    # freqs_cos: (seq_len, dim//2)
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    # Expand freqs to (1, 1, seq_len, head_dim//2)
    freqs_cos = freqs_cos[:seq_len, :head_dim//2][None, None, :, :]
    freqs_sin = freqs_sin[:seq_len, :head_dim//2][None, None, :, :]
    # Split q into two parts
    q1 = q[..., :head_dim//2]
    q2 = q[..., head_dim//2:]
    k1 = k[..., :head_dim//2]
    k2 = k[..., head_dim//2:]
    # Apply RoPE
    q_rot1 = q1 * freqs_cos - q2 * freqs_sin
    q_rot2 = q1 * freqs_sin + q2 * freqs_cos
    k_rot1 = k1 * freqs_cos - k2 * freqs_sin
    k_rot2 = k1 * freqs_sin + k2 * freqs_cos
    # Concatenate
    q_rot = torch.cat([q_rot1, q_rot2], dim=-1)
    k_rot = torch.cat([k_rot1, k_rot2], dim=-1)
    return q_rot, k_rot


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        num_heads=12,
        num_kv_heads=4,
        max_seq_len=512
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.max_seq_len = max_seq_len
        self._build_rotary_embeddings(max_seq_len, self.head_dim)

    def _build_rotary_embeddings(self, seq_len, dim):
        assert dim % 2 == 0
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, self.freqs_cos, self.freqs_sin)

        k_rot = k_rot.repeat(1, self.num_heads // self.num_kv_heads, 1, 1)
        v = v.repeat(1, self.num_heads // self.num_kv_heads, 1, 1)

        attn_weights = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask, -float('inf'))

        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.o_proj(output)

        return output
