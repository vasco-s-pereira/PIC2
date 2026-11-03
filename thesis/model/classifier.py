import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .attn_continuous import ContinuousBasis, ContinuousSelfAttention
from .attn_discrete import DiscreteSelfAttention
from .blocks import Block, LayerNorm
from .positional import TimeEncoding, time_grid


@dataclass
class ModelConfig:
    vocab_size: int = 32
    n_classes: int = 8
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    attention: str = "discrete"     # "discrete" | "continuous"
    pooling: str = "mean"           # "mean" | "cls"
    pos_n_freqs: int = 16
    pos_max_freq: float = 32.0
    # continuous attention only
    n_basis: int = 32
    basis_sigma: float | None = None  # default 1 / n_basis
    ridge: float = 1e-3
    init_var: float = 0.01          # initial sigma_i^2 of the attention densities


class SequenceClassifier(nn.Module):
    """tok_emb + pos(t) -> non-causal blocks -> pooling -> Linear(n_embd, N)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = TimeEncoding(cfg.n_embd, cfg.pos_n_freqs, cfg.pos_max_freq)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        elif cfg.pooling != "mean":
            raise ValueError(cfg.pooling)

        if cfg.attention == "continuous":
            self.basis = ContinuousBasis(cfg.n_basis, cfg.basis_sigma, cfg.ridge)
            make_attn = lambda: ContinuousSelfAttention(cfg, self.basis)
        elif cfg.attention == "discrete":
            make_attn = lambda: DiscreteSelfAttention(cfg)
        else:
            raise ValueError(cfg.attention)
        self.blocks = nn.ModuleList([Block(cfg, make_attn()) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.n_classes)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.pooling == "cls":
            nn.init.normal_(self.cls, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, K = tokens.shape
        t = time_grid(K, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos(t)[None])
        n_prefix = 0
        if self.cfg.pooling == "cls":
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
            n_prefix = 1
        for blk in self.blocks:
            x = blk(x, n_prefix=n_prefix)
        x = self.ln_f(x)
        pooled = x[:, 0] if n_prefix else x.mean(1)
        return self.head(pooled)

    def set_record(self, on: bool = True):
        for blk in self.blocks:
            if isinstance(blk.attn, ContinuousSelfAttention):
                blk.attn.record = on

    def recorded_densities(self):
        """[(mu, sigma^2)] per layer, each (B, H, T), from the last forward with recording on."""
        return [(b.attn.last_mu, b.attn.last_var) for b in self.blocks
                if isinstance(b.attn, ContinuousSelfAttention)]
