import torch.nn as nn
import torch.nn.functional as F


class DiscreteSelfAttention(nn.Module):
    """nanoGPT CausalSelfAttention with the causal mask removed: bidirectional softmax attention."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, n_prefix: int = 0):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (z.view(B, T, H, C // H).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
