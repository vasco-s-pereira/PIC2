import math
from types import SimpleNamespace

import pytest
import torch

from model.attn_continuous import ContinuousBasis, ContinuousSelfAttention, gaussian_pdf
from model.positional import time_grid


def _cfg(**kw):
    base = dict(n_embd=8, n_head=2, bias=True, dropout=0.0, init_var=0.01)
    base.update(kw)
    return SimpleNamespace(**base)


def _recon_error(K, n_basis, ridge, d=4, seed=0):
    torch.manual_seed(seed)
    basis = ContinuousBasis(n_basis=n_basis, ridge=ridge).double()
    # smooth-ish random signal plus noise, so reconstruction error is meaningful for N_b < K
    t = time_grid(K, dtype=torch.float64)
    X = torch.stack([torch.sin(2 * math.pi * (j + 1) * t + j) for j in range(d)], -1)[None]
    X = X + 0.3 * torch.randn_like(X)
    Xbar = basis.evaluate(basis.fit(X), t)
    return ((Xbar - X).norm() / X.norm()).item()


def test_reconstruction_exact_when_basis_covers_grid():
    K = 32
    for n_basis in (K, 2 * K):
        assert _recon_error(K, n_basis, ridge=1e-10) < 1e-5


def test_reconstruction_error_grows_smoothly_as_basis_shrinks():
    # N_b = K exactly is nearly singular (grid i/K beats against centres j/(N_b-1)), so start at 2K
    K = 128
    errs = [_recon_error(K, nb, ridge=1e-6) for nb in (256, 96, 64, 32, 16, 8)]
    assert all(b >= a - 1e-6 for a, b in zip(errs, errs[1:])), errs
    assert errs[0] < 1e-3 and errs[-1] < 1.0


def test_G_is_cached_per_length():
    basis = ContinuousBasis(n_basis=16)
    g1 = basis.G(64, "cpu", torch.float32)
    assert basis.G(64, "cpu", torch.float32) is g1
    assert basis.G(128, "cpu", torch.float32).shape == (16, 128)


@pytest.mark.parametrize("mu,var", [(0.5, 0.01), (0.2, 0.002), (0.9, 0.05), (0.5, 1e-4)])
def test_expected_psi_closed_form_matches_numerical_integration(mu, var):
    basis = ContinuousBasis(n_basis=16, sigma=0.03).double()
    t = torch.linspace(-3, 4, 400001, dtype=torch.float64)    # whole real line, effectively
    p = gaussian_pdf(t, torch.tensor(mu, dtype=torch.float64), var)
    numeric = torch.trapezoid(p[:, None] * basis.psi(t), t, dim=0)
    closed = basis.expected_psi(torch.tensor([mu], dtype=torch.float64),
                                torch.tensor([var], dtype=torch.float64))[0]
    torch.testing.assert_close(closed, numeric, rtol=1e-6, atol=1e-8)


def test_hard_attention_limit_reads_out_signal_at_mu():
    torch.manual_seed(0)
    cfg = _cfg(n_embd=8, n_head=2)
    basis = ContinuousBasis(n_basis=24, ridge=1e-6).double()
    attn = ContinuousSelfAttention(cfg, basis).double()
    K, k = 40, 13
    t_k = k / K
    with torch.no_grad():
        for lin in (attn.q, attn.v, attn.c_proj):
            lin.bias.zero_()
        attn.q.weight.zero_()
        attn.v.weight.copy_(torch.eye(8))
        attn.c_proj.weight.copy_(torch.eye(8))
        attn.w_mu.zero_(); attn.w_var.zero_()
        attn.b_mu.fill_(math.log(t_k / (1 - t_k)))   # sigmoid(b) = t_k
        attn.b_var.fill_(-40.0)                      # softplus(-40) ~ 4e-18 -> sigma_i^2 -> 0
    x = torch.randn(3, K, 8, dtype=torch.float64)
    out = attn(x)                                                   # (3, K, 8), same for all queries
    target = basis.evaluate(basis.fit(x), torch.tensor([t_k], dtype=torch.float64))  # (3, 1, 8)
    torch.testing.assert_close(out, target.expand_as(out), rtol=1e-6, atol=1e-8)


def test_gradcheck():
    torch.manual_seed(0)
    cfg = _cfg(n_embd=4, n_head=2)
    attn = ContinuousSelfAttention(cfg, ContinuousBasis(n_basis=6, ridge=1e-2)).double()
    x = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: attn(x), (x,), eps=1e-6, atol=1e-5)
    params = [p for p in attn.parameters()]
    assert torch.autograd.gradcheck(
        lambda *ps: torch.func.functional_call(attn, dict(zip([n for n, _ in attn.named_parameters()], ps)), (x.detach(),)),
        tuple(p.detach().clone().requires_grad_(True) for p in params), eps=1e-6, atol=1e-5,
    )


def test_prefix_tokens_query_but_are_not_in_signal():
    torch.manual_seed(0)
    attn = ContinuousSelfAttention(_cfg(), ContinuousBasis(n_basis=8)).double()
    x = torch.randn(2, 11, 8, dtype=torch.float64)
    y = attn(x, n_prefix=1)
    x2 = x.clone(); x2[:, 0] += 5.0
    y2 = attn(x2, n_prefix=1)
    # changing the prefix changes only its own output (its query), not the others'
    torch.testing.assert_close(y[:, 1:], y2[:, 1:])
    assert not torch.allclose(y[:, 0], y2[:, 0])
