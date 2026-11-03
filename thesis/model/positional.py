import math

import torch
import torch.nn as nn


def time_grid(K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """t_i = i / K, the same grid the data generator uses."""
    return torch.arange(K, device=device, dtype=dtype) / K


class TimeEncoding(nn.Module):
    """pos(t) = Linear([sin(w_k t), cos(w_k t)]_k), w_k geometric in [pi, pi * max_freq].

    Depends only on t in [0, 1], never on the absolute index, so it is defined for any K.
    """

    def __init__(self, n_embd: int, n_freqs: int = 16, max_freq: float = 32.0):
        super().__init__()
        freqs = math.pi * torch.logspace(0, math.log2(max_freq), n_freqs, base=2.0)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_freqs, n_embd)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = t[:, None] * self.freqs[None, :]
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))
