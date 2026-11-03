"""Synthetic sequence classification with smooth position-dependent token mixtures.

For class n in {0..N-1}, position i in {0..K-1}, t_i = i / K:

    P_n(x_i) = f_n(t_i) Q1^(n)(x_i) + (1 - f_n(t_i)) Q2^(n)(x_i)

Q1, Q2 are categoricals over a vocabulary of size V drawn from a symmetric Dirichlet,
f_n : [0, 1] -> [0, 1] is drawn from one of a few smooth families. Since f_n lives on
t in [0, 1], the same classes generate sequences of any length K.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

FAMILIES = ("linear", "sine", "gmm")
_FINE_GRID = np.linspace(0.0, 1.0, 4097)


def time_grid(K: int) -> np.ndarray:
    return np.arange(K) / K


@dataclass
class TaskConfig:
    n_classes: int = 8
    vocab_size: int = 32
    concentration: float = 1.0      # Dirichlet concentration; high -> Q's look alike -> harder
    position_only: bool = False     # all classes share (Q1, Q2), differ only in f_n
    families: list = field(default_factory=lambda: list(FAMILIES))
    match_mean: bool | None = None  # rescale f_n so that int_0^1 f_n = 1/2; default: = position_only
    signal_scale: float = 1.0       # shrink every Q towards the global mean (1 = untouched)
    seed: int = 0


# ---------------------------------------------------------------------------- f families

def _sample_f(rng: np.random.Generator, family: str) -> dict:
    if family == "linear":
        a = rng.uniform(-2.0, 2.0)
        return dict(family="linear", a=a, b=rng.uniform(0.0, 1.0) - a / 2)
    if family == "sine":
        return dict(family="sine", omega=rng.uniform(0.5, 3.0), phi=rng.uniform(0, 2 * math.pi))
    if family == "gmm":
        m = int(rng.integers(1, 4))
        return dict(
            family="gmm",
            weights=rng.dirichlet(np.ones(m)).tolist(),
            centers=rng.uniform(0.0, 1.0, m).tolist(),
            widths=rng.uniform(0.05, 0.2, m).tolist(),
        )
    raise ValueError(f"unknown f family {family!r}")


def _eval_raw_f(p: dict, t: np.ndarray) -> np.ndarray:
    if p["family"] == "linear":
        return np.clip(p["a"] * t + p["b"], 0.0, 1.0)
    if p["family"] == "sine":
        return 0.5 + 0.5 * np.sin(2 * math.pi * p["omega"] * t + p["phi"])
    if p["family"] == "gmm":
        g = sum(
            w * np.exp(-0.5 * ((t - c) / s) ** 2) / (s * math.sqrt(2 * math.pi))
            for w, c, s in zip(p["weights"], p["centers"], p["widths"])
        )
        return (g - p["_lo"]) / (p["_hi"] - p["_lo"])
    raise ValueError(p["family"])


def _finalize_f(p: dict, match_mean: bool) -> dict:
    """Attach normalisation constants computed on a fine grid of [0, 1]."""
    p = dict(p)
    if p["family"] == "gmm":
        p["_lo"], p["_hi"] = 0.0, 1.0
        g = _eval_raw_f(p, _FINE_GRID)
        p["_lo"], p["_hi"] = float(g.min()), float(g.max())
    p["_shift"], p["_scale"] = 0.0, 1.0
    if match_mean:
        f = _eval_raw_f(p, _FINE_GRID)
        m = float(f.mean())
        dev = float(np.abs(f - m).max())
        p["_shift"], p["_scale"] = m, (min(1.0, 0.5 / dev) if dev > 0 else 1.0)
    return p


def eval_f(p: dict, t: np.ndarray) -> np.ndarray:
    f = _eval_raw_f(p, t)
    if p["_shift"] or p["_scale"] != 1.0:
        f = 0.5 + p["_scale"] * (f - p["_shift"])
    return np.clip(f, 0.0, 1.0)


# ---------------------------------------------------------------------------- task

class SyntheticTask:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        N, V = cfg.n_classes, cfg.vocab_size
        rng = np.random.default_rng(cfg.seed)
        alpha = np.full(V, cfg.concentration)
        if cfg.position_only:
            q1, q2 = rng.dirichlet(alpha), rng.dirichlet(alpha)
            Q1, Q2 = np.tile(q1, (N, 1)), np.tile(q2, (N, 1))
        else:
            Q1, Q2 = rng.dirichlet(alpha, size=N), rng.dirichlet(alpha, size=N)
        match_mean = cfg.position_only if cfg.match_mean is None else cfg.match_mean
        fams = rng.choice(list(cfg.families), size=N)
        self.f_params = [_finalize_f(_sample_f(rng, f), match_mean) for f in fams]

        self.Q1_raw, self.Q2_raw = Q1, Q2
        qbar = np.concatenate([Q1, Q2]).mean(0, keepdims=True)
        s = cfg.signal_scale
        self.Q1 = qbar + s * (Q1 - qbar)
        self.Q2 = qbar + s * (Q2 - qbar)
        self._cache: dict[int, dict] = {}

    @property
    def n_classes(self) -> int:
        return self.cfg.n_classes

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    def with_signal_scale(self, s: float) -> "SyntheticTask":
        return SyntheticTask(replace(self.cfg, signal_scale=s))

    def f(self, t: np.ndarray) -> np.ndarray:
        """(N, len(t)) mixing weights f_n(t)."""
        return np.stack([eval_f(p, t) for p in self.f_params])

    def probs(self, K: int) -> torch.Tensor:
        """(N, K, V) float64 table of P_n(x_i = v)."""
        return self._tables(K)["P"]

    def _tables(self, K: int) -> dict:
        if K not in self._cache:
            f = self.f(time_grid(K))[:, :, None]
            P = f * self.Q1[:, None, :] + (1 - f) * self.Q2[:, None, :]
            P = torch.from_numpy(P)
            cdf = P.cumsum(-1)
            cdf[..., -1] = 1.0
            self._cache[K] = dict(P=P, logP=P.clamp_min(1e-300).log(), cdf=cdf.float())
        return self._cache[K]

    def sample(self, batch_size: int, K: int, generator: torch.Generator | None = None,
               labels: torch.Tensor | None = None):
        """Returns tokens (B, K) long and labels (B,) long, on CPU."""
        cdf = self._tables(K)["cdf"]
        if labels is None:
            labels = torch.randint(0, self.n_classes, (batch_size,), generator=generator)
        u = torch.rand(batch_size, K, 1, generator=generator)
        tokens = (u > cdf[labels]).sum(-1).clamp_max_(self.vocab_size - 1)
        return tokens, labels

    def bayes_log_posterior(self, tokens: torch.Tensor) -> torch.Tensor:
        """log P(n | x) under a uniform class prior, (B, N)."""
        B, K = tokens.shape
        logP = self._tables(K)["logP"]                                  # (N, K, V)
        idx = tokens.cpu().T.unsqueeze(0).expand(self.n_classes, K, B)  # (N, K, B)
        ll = logP.gather(2, idx).sum(1).T                               # (B, N)
        return torch.log_softmax(ll, dim=-1)

    def bayes_predict(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.bayes_log_posterior(tokens).argmax(-1)

    def bayes_accuracy(self, K: int, n: int = 20000, seed: int = 12345, batch: int = 4096) -> float:
        g = torch.Generator().manual_seed(seed)
        correct = 0
        for start in range(0, n, batch):
            x, y = self.sample(min(batch, n - start), K, generator=g)
            correct += (self.bayes_predict(x) == y).sum().item()
        return correct / n

    def positional_information(self, K: int) -> np.ndarray:
        """I(n; x_i) in nats at each position, under a uniform prior: where classes differ."""
        P = self.probs(K).numpy()                                       # (N, K, V)
        M = P.mean(0, keepdims=True)
        kl = (P * (np.log(np.clip(P, 1e-300, None)) - np.log(np.clip(M, 1e-300, None)))).sum(-1)
        return kl.mean(0)

    def describe(self) -> dict:
        d = asdict(self.cfg)
        d["f_families"] = [p["family"] for p in self.f_params]
        return d


def calibrate_signal_scale(cfg: TaskConfig, K: int, target_acc: float, n: int = 8000,
                           tol: float = 0.005, max_iter: int = 20, seed: int = 4321) -> float:
    """Bisect signal_scale in (0, 1] so that the Bayes-optimal accuracy at length K ~= target.

    Uses common random numbers across candidate scales, so the curve is monotone in practice.
    """
    full = SyntheticTask(replace(cfg, signal_scale=1.0))
    if full.bayes_accuracy(K, n, seed) <= target_acc:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        acc = full.with_signal_scale(mid).bayes_accuracy(K, n, seed)
        if abs(acc - target_acc) < tol:
            return mid
        lo, hi = (mid, hi) if acc < target_acc else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------- baselines

def unigram_features(tokens: torch.Tensor, V: int) -> np.ndarray:
    """Normalised token counts, (B, V). Blind to position."""
    B, K = tokens.shape
    c = torch.zeros(B, V, dtype=torch.float64).scatter_add_(1, tokens, torch.ones(B, K, dtype=torch.float64))
    return (c / K).numpy()


def binned_features(tokens: torch.Tensor, V: int, n_bins: int = 8) -> np.ndarray:
    """Normalised token counts per position bin of t, (B, n_bins * V)."""
    B, K = tokens.shape
    bins = torch.from_numpy(np.floor(time_grid(K) * n_bins).astype(np.int64))
    idx = bins[None, :] * V + tokens
    c = torch.zeros(B, n_bins * V, dtype=torch.float64).scatter_add_(1, idx, torch.ones(B, K, dtype=torch.float64))
    per_bin = torch.bincount(bins, minlength=n_bins).clamp_min(1).double().repeat_interleave(V)
    return (c / per_bin).numpy()


def run_baselines(task: SyntheticTask, K: int, n_train: int = 20000, n_test: int = 10000,
                  n_bins: int = 8, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    g = torch.Generator().manual_seed(seed)
    xtr, ytr = task.sample(n_train, K, generator=g)
    xte, yte = task.sample(n_test, K, generator=g)
    out = {"bayes": (task.bayes_predict(xte) == yte).double().mean().item()}
    V = task.vocab_size
    for name, feat in [("unigram", lambda x: unigram_features(x, V)),
                       ("binned", lambda x: binned_features(x, V, n_bins))]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(feat(xtr), ytr.numpy())
        out[name] = float((clf.predict(feat(xte)) == yte.numpy()).mean())
    out["chance"] = 1.0 / task.n_classes
    return out
