"""Continuous attention (Martins et al. 2020; infinity-former, Martins et al. 2022, Sec. 3).

The K hidden states are turned into a continuous signal X_bar(t) = B^T psi(t) by ridge
regression on N_b Gaussian basis functions; each query attends with a Gaussian density
p_i = N(mu_i, sigma_i^2) over t in [0, 1] and reads out z_i = V^T E_{p_i}[psi(t)], which has
a closed form. Cost is O(K N_b d) instead of O(K^2 d).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import time_grid


def gaussian_pdf(x: torch.Tensor, mean: torch.Tensor, var) -> torch.Tensor:
    return torch.exp(-0.5 * (x - mean) ** 2 / var) / (2 * math.pi * var) ** 0.5


def inverse_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class ContinuousBasis(nn.Module):
    """psi_j(t) = N(t; c_j, s^2), c_j equispaced in [0, 1]. Shared by all layers and heads."""

    def __init__(self, n_basis: int = 32, sigma: float | None = None, ridge: float = 1e-3):
        super().__init__()
        self.n_basis = n_basis
        self.sigma = float(sigma) if sigma is not None else 1.0 / n_basis
        self.ridge = float(ridge)
        self.register_buffer("centers", torch.linspace(0, 1, n_basis), persistent=False)
        self._G_cache: dict = {}

    def psi(self, t: torch.Tensor) -> torch.Tensor:
        """(...,) -> (..., N_b)."""
        return gaussian_pdf(t[..., None], self.centers.to(t.dtype), self.sigma ** 2)

    def G(self, K: int, device, dtype) -> torch.Tensor:
        """G_K = (F F^T + lambda I)^{-1} F, (N_b, K); depends only on K."""
        key = (K, str(device), dtype)
        if key not in self._G_cache:
            t = time_grid(K, device=device, dtype=torch.float64)
            Fm = self.psi(t).T                                            # (N_b, K)
            A = Fm @ Fm.T + self.ridge * torch.eye(self.n_basis, device=device, dtype=torch.float64)
            self._G_cache[key] = torch.linalg.solve(A, Fm).to(dtype)
        return self._G_cache[key]

    def fit(self, X: torch.Tensor) -> torch.Tensor:
        """Ridge coefficients B for X on the grid t_i = i/K: (Bsz, K, d) -> (Bsz, N_b, d)."""
        return torch.einsum("nk,bkd->bnd", self.G(X.shape[1], X.device, X.dtype), X)

    def evaluate(self, Bcoef: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """X_bar(t) = B^T psi(t): (Bsz, N_b, d), (T,) -> (Bsz, T, d)."""
        return torch.einsum("tn,bnd->btd", self.psi(t.to(Bcoef.dtype)), Bcoef)

    def expected_psi(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """E_{N(t; mu, var)}[psi_j(t)] = N(mu; c_j, var + s^2), ignoring truncation to [0, 1]."""
        return gaussian_pdf(mu[..., None], self.centers.to(mu.dtype), var[..., None] + self.sigma ** 2)


class ContinuousSelfAttention(nn.Module):
    def __init__(self, cfg, basis: ContinuousBasis):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        dh = cfg.n_embd // cfg.n_head
        self.basis = basis
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # per-head affine maps of the query to (mu, sigma^2), as in the infinity-former
        self.w_mu = nn.Parameter(torch.randn(cfg.n_head, dh) / math.sqrt(dh))
        self.b_mu = nn.Parameter(torch.zeros(cfg.n_head))
        self.w_var = nn.Parameter(torch.randn(cfg.n_head, dh) / math.sqrt(dh))
        self.b_var = nn.Parameter(torch.full((cfg.n_head,), inverse_softplus(cfg.init_var)))
        self.record = False
        self.last_mu = self.last_var = None

    def density_params(self, q: torch.Tensor):
        """q: (Bsz, H, T, dh) -> mu, sigma^2 each (Bsz, H, T)."""
        mu = torch.sigmoid(torch.einsum("bhtd,hd->bht", q, self.w_mu) + self.b_mu[:, None])
        var = F.softplus(torch.einsum("bhtd,hd->bht", q, self.w_var) + self.b_var[:, None])
        return mu, var

    def forward(self, x: torch.Tensor, n_prefix: int = 0) -> torch.Tensor:
        """x: (Bsz, n_prefix + K, d). Prefix tokens (e.g. [CLS]) query but are not part of the signal."""
        Bsz, T, d = x.shape
        H, dh = self.n_head, d // self.n_head
        Bcoef = self.basis.fit(self.v(x[:, n_prefix:]))                   # (Bsz, N_b, d)
        Vh = Bcoef.view(Bsz, -1, H, dh).transpose(1, 2)                   # (Bsz, H, N_b, dh)
        q = self.q(x).view(Bsz, T, H, dh).transpose(1, 2)                 # (Bsz, H, T, dh)
        mu, var = self.density_params(q)
        if self.record:
            self.last_mu, self.last_var = mu.detach(), var.detach()
        w = self.basis.expected_psi(mu, var)                              # (Bsz, H, T, N_b)
        z = w @ Vh                                                        # (Bsz, H, T, dh)
        return self.resid_dropout(self.c_proj(z.transpose(1, 2).reshape(Bsz, T, d)))
