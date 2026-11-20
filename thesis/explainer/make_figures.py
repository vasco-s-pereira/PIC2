"""Figures for explainer/index.html.

    .venv/bin/python explainer/make_figures.py
"""
import csv
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import SyntheticTask, TaskConfig  # noqa: E402
from model.attn_continuous import ContinuousBasis, gaussian_pdf  # noqa: E402
from model.positional import time_grid  # noqa: E402
from train import load_config  # noqa: E402

IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def task_from(cfg_name):
    return SyntheticTask(TaskConfig(**load_config(str(ROOT / "configs" / cfg_name))["task"]))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------- 1. the task
def fig_task():
    task = task_from("position_only.yaml")
    grid = np.linspace(0, 1, 400)
    f = task.f(grid)
    K = 128
    P = task.probs(K).numpy()
    info = task.positional_information(K)

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 0.9], hspace=0.85, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    for n in range(task.n_classes):
        ax.plot(grid, f[n], lw=1.6, label=f"$f_{n}$ ({task.f_params[n]['family']})")
    ax.set_xlabel("t = i / K"); ax.set_ylabel("$f_n(t)$")
    ax.set_title("Mixing functions: each class n decides how much of $Q_1$ vs $Q_2$ to use at time t")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(task.vocab_size), task.Q1[0], alpha=0.7, label="$Q_1$")
    ax.bar(range(task.vocab_size), task.Q2[0], alpha=0.7, label="$Q_2$")
    ax.set_xlabel("token v"); ax.set_title("Shared token distributions"); ax.legend()

    for j, n in enumerate([0, 5]):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(P[n].T, aspect="auto", origin="lower", extent=[0, 1, -0.5, task.vocab_size - 0.5], cmap="magma")
        ax.set_title(f"$P_{n}(x_i = v)$ for class {n}"); ax.set_xlabel("t"); ax.set_ylabel("token v")
    ax = fig.add_subplot(gs[1, 2])
    ax.fill_between(time_grid(K), info, color="k", alpha=0.3)
    ax.set_title("Positional information $I(n; x_t)$"); ax.set_xlabel("t"); ax.set_ylabel("nats")

    ax = fig.add_subplot(gs[2, :])
    g = torch.Generator().manual_seed(3)
    x, _ = task.sample(8, K, generator=g, labels=torch.tensor([0, 0, 0, 0, 5, 5, 5, 5]))
    ax.imshow(x.numpy(), aspect="auto", cmap="tab20b", interpolation="nearest", extent=[0, 1, 7.5, -0.5])
    ax.set_yticks(range(8)); ax.set_yticklabels(["class 0"] * 4 + ["class 5"] * 4, fontsize=8)
    ax.set_xlabel("t"); ax.set_title("8 sampled sequences (K = 128, colour = token id)")
    fig.savefig(IMG / "task.png"); plt.close(fig)


def fig_bayes_vs_K():
    Ks = [16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cfg in ("easy.yaml", "base.yaml", "position_only.yaml"):
        task = task_from(cfg)
        acc = [task.bayes_accuracy(K, n=4000) for K in Ks]
        ax.plot(Ks, acc, "o-", label=cfg.replace(".yaml", ""))
    ax.axhline(1 / 8, color="gray", ls=":", label="chance (1/8)")
    ax.set_xscale("log", base=2); ax.set_xlabel("sequence length K"); ax.set_ylabel("Bayes-optimal accuracy")
    ax.set_title("More tokens = more evidence: the ceiling rises with K"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(IMG / "bayes_vs_K.png"); plt.close(fig)


# ---------------------------------------------------------------------------- 2. continuous attention
def fig_basis():
    torch.manual_seed(1)
    K, Nb = 64, 16
    basis = ContinuousBasis(n_basis=Nb, ridge=1e-3).double()
    t = time_grid(K, dtype=torch.float64)
    X = (torch.sin(2 * math.pi * 1.3 * t) + 0.5 * torch.cos(2 * math.pi * 3.1 * t) + 0.25 * torch.randn(K))[None, :, None]
    B = basis.fit(X)
    fine = torch.linspace(0, 1, 500, dtype=torch.float64)
    Xbar = basis.evaluate(B, fine)[0, :, 0]
    psi = basis.psi(fine)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axes[0]
    ax.plot(fine, psi.numpy(), lw=0.9)
    ax.set_title(f"$N_b$={Nb} Gaussian basis functions $\\psi_j(t)$"); ax.set_xlabel("t")
    ax = axes[1]
    ax.stem(t.numpy(), X[0, :, 0].numpy(), basefmt=" ", markerfmt="C0.", linefmt="C0-")
    ax.plot(fine, Xbar.numpy(), "C3", lw=2.2, label="$\\bar X(t) = B^\\top \\psi(t)$")
    ax.set_title(f"K={K} discrete states $\\to$ continuous signal (ridge)"); ax.set_xlabel("t"); ax.legend(fontsize=8)
    ax = axes[2]
    contrib = psi.numpy() * B[0, :, 0].numpy()[None]
    ax.plot(fine, contrib, lw=0.8, alpha=0.8)
    ax.plot(fine, Xbar.numpy(), "k", lw=2, label="sum")
    ax.set_title("$\\bar X(t) = \\sum_j b_j \\psi_j(t)$"); ax.set_xlabel("t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "basis_fit.png"); plt.close(fig)


def fig_attention_compare():
    torch.manual_seed(0)
    K = 32
    t = time_grid(K).numpy()
    scores = np.random.randn(K) * 1.2
    scores[18:22] += 3
    a = np.exp(scores) / np.exp(scores).sum()

    fine = np.linspace(0, 1, 500)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].bar(t, a, width=0.8 / K, color="C0")
    axes[0].set_title("Discrete: softmax over K positions\n$p = \\mathrm{softmax}(q^\\top k_1, \\dots, q^\\top k_K)$")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("weight")
    for mu, s2, c in [(0.6, 0.004, "C3"), (0.3, 0.02, "C2"), (0.85, 0.0004, "C4")]:
        axes[1].plot(fine, gaussian_pdf(torch.tensor(fine), torch.tensor(mu), s2).numpy(), c,
                     label=f"$\\mu$={mu}, $\\sigma^2$={s2}")
    axes[1].set_title("Continuous: one Gaussian density per query\n$p(t) = \\mathcal{N}(t;\\mu(q),\\sigma^2(q))$")
    axes[1].set_xlabel("t"); axes[1].legend(fontsize=8)

    basis = ContinuousBasis(n_basis=32).double()
    mu, var = torch.tensor([0.6], dtype=torch.float64), torch.tensor([0.004], dtype=torch.float64)
    w = basis.expected_psi(mu, var)[0].numpy()
    axes[2].bar(basis.centers.numpy(), w, width=0.02, color="C3")
    axes[2].set_title("Read-out weights over the basis\n$\\mathbb{E}_p[\\psi_j] = \\mathcal{N}(\\mu; c_j, \\sigma^2 + s^2)$")
    axes[2].set_xlabel("basis centre $c_j$")
    fig.tight_layout(); fig.savefig(IMG / "attention_compare.png"); plt.close(fig)


def fig_conditioning():
    """Why continuous attention breaks at K=32: with N_b=32, the ridge problem is (nearly) singular."""
    Nb = 32
    Ks = [16, 24, 32, 40, 48, 64, 96, 128, 256, 512]
    conds, errs = [], []
    for K in Ks:
        basis = ContinuousBasis(n_basis=Nb, ridge=1e-3).double()
        Fm = basis.psi(time_grid(K, dtype=torch.float64)).T
        conds.append(torch.linalg.cond(Fm @ Fm.T).item())
        # same underlying smooth signal, seen at resolution K; how far is X_bar from the K=512 X_bar?
        def xbar(KK):
            tt = time_grid(KK, dtype=torch.float64)
            X = torch.stack([torch.sin(2 * math.pi * (j + 1) * tt + j) for j in range(4)], -1)[None]
            return basis.evaluate(basis.fit(X), torch.linspace(0, 1, 400, dtype=torch.float64))
        ref = xbar(512)
        errs.append(((xbar(K) - ref).norm() / ref.norm()).item())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].semilogy(Ks, conds, "o-"); axes[0].axvline(Nb, color="C3", ls="--", label="K = $N_b$ = 32")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("K"); axes[0].set_title("cond$(F F^\\top)$ (before adding $\\lambda I$)")
    axes[0].legend()
    axes[1].plot(Ks, errs, "o-", color="C1"); axes[1].axvline(Nb, color="C3", ls="--")
    axes[1].axvline(128, color="gray", ls=":", label="train K = 128")
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("K")
    axes[1].set_title("Same smooth signal, sampled at K:\nrelative change in $\\bar X(t)$ vs K=512"); axes[1].legend()
    fig.tight_layout(); fig.savefig(IMG / "conditioning.png"); plt.close(fig)


# ---------------------------------------------------------------------------- 3. results
def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, name in zip(axes, ("compare_easy", "compare_position_only")):
        bayes = None
        for attn, c in (("discrete", "C0"), ("continuous", "C1")):
            d = ROOT / "runs" / name / f"{attn}_s0"
            tr = read_csv(d / "train_log.csv")
            ev = read_csv(d / "eval_log.csv")
            steps = np.array([int(r["step"]) for r in tr]); acc = np.array([float(r["acc"]) for r in tr])
            sm = np.convolve(acc, np.ones(5) / 5, mode="valid")
            ax.plot(steps[2:-2], sm, c, alpha=0.35, lw=1)
            ax.plot([int(r["step"]) for r in ev], [float(r["acc"]) for r in ev], c + "o-", label=f"{attn} (eval)")
            bayes = float(ev[0]["bayes_acc"])
        ax.axhline(bayes, color="k", ls="--", label="Bayes-optimal")
        ax.set_title(name.replace("compare_", "") + " task"); ax.set_xlabel("step"); ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "training_curves.png"); plt.close(fig)


def fig_compare_bars():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, name in zip(axes, ("easy", "position_only")):
        rows = {r["attention"]: r for r in read_csv(ROOT / "experiments/out" / f"compare_{name}.csv")}
        r0 = rows["discrete"]
        labels = ["chance", "unigram LR", "binned LR", "discrete attn", "continuous attn", "Bayes-optimal"]
        vals = [float(r0["chance"]), float(r0["unigram_lr"]), float(r0["binned_lr"]),
                float(rows["discrete"]["acc"]), float(rows["continuous"]["acc"]), float(r0["bayes"])]
        colors = ["#bbb", "#999", "#777", "C0", "C1", "k"]
        b = ax.barh(labels, vals, color=colors)
        ax.bar_label(b, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlim(0, 1.12); ax.set_title(name + " task (K = 128)")
    fig.tight_layout(); fig.savefig(IMG / "compare_bars.png"); plt.close(fig)


def fig_ablations():
    rows = read_csv(ROOT / "experiments/out/ablations_position_only.csv")
    axes_names = ["n_basis", "basis_sigma", "ridge", "n_head", "pooling"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for ax, a in zip(axes, axes_names):
        for attn, c in (("discrete", "C0"), ("continuous", "C1")):
            pts = [(r["value"], float(r["acc"])) for r in rows if r["axis"] == a and r["attention"] == attn]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], c + "o-", label=attn)
        ax.axhline(float(rows[0]["bayes"]), color="k", ls="--", lw=1)
        ax.set_title(a); ax.set_ylim(0.84, 0.875); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("accuracy (K=128)"); axes[3].legend(fontsize=8)
    fig.suptitle("Ablations on the position-only task (dashed = Bayes-optimal 0.869)", y=1.03)
    fig.tight_layout(); fig.savefig(IMG / "ablations.png"); plt.close(fig)


def fig_cost():
    rows = read_csv(ROOT / "experiments/out/sweepK.csv")
    bench = read_csv(ROOT / "experiments/out/sweepK_bench.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for v, c in (("discrete", "C0"), ("cont_fixed", "C1"), ("cont_prop", "C2")):
        r = [x for x in rows if x["variant"] == v]
        b = [x for x in bench if x["variant"] == v]
        Ks = [int(x["K"]) for x in r]
        axes[0].plot(Ks, [float(x["gap"]) for x in r], c + "o-", label=v)
        axes[1].loglog(Ks, [float(x["step_ms"]) for x in r], c + "o-", label=v, base=2)
        axes[2].loglog([int(x["K"]) for x in b], [float(x["step_ms"]) for x in b], c + "o-", label=v, base=2)
    Kr = np.array([256, 2048])
    axes[2].loglog(Kr, 120 * (Kr / 256) ** 2, "k:", lw=1, label="$\\propto K^2$", base=2)
    axes[2].loglog(Kr, 100 * (Kr / 256), "k--", lw=1, label="$\\propto K$", base=2)
    axes[0].set_xscale("log", base=2); axes[0].set_title("gap to Bayes-optimal (trained, K = train = eval)")
    axes[1].set_title("time / train step, GPU (ms)"); axes[2].set_title("time / train step, CPU benchmark (ms)")
    for ax in axes:
        ax.set_xlabel("K"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "cost.png"); plt.close(fig)


def copy_existing():
    for src, dst in [("experiments/out/lengthgen_base.png", "lengthgen_base.png"),
                     ("experiments/out/lengthgen_position_only.png", "lengthgen_position_only.png"),
                     ("experiments/out/sweepK.png", "sweepK.png"),
                     ("runs/compare_position_only/continuous_s0/diagnose.png", "diagnose_position_only.png"),
                     ("runs/compare_easy/continuous_s0/diagnose.png", "diagnose_easy.png")]:
        shutil.copy(ROOT / src, IMG / dst)


def fig_roadmap():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, 1.0, 2.8, 1.5, "#eaf7ef", "#2a9d59",
         "Phase 1 — DONE\nContinuous attention\nvs discrete, on synthetic\nclassification"),
        (3.4, 1.0, 2.8, 1.5, "#eef3fd", "#2f6fdf",
         "Phase 2 — NEXT\nSequences ↔ continuous\nsignals; reconstruct &\nlength-free decode"),
        (6.6, 1.0, 2.8, 1.5, "#fff6e5", "#d98b00",
         "Phase 3 — GOAL\nSignal-to-signal diffusion\nfor language / seq2seq\nof arbitrary length"),
    ]
    for x, y, w, h, face, edge, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=face, edgecolor=edge, linewidth=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for x in (3.1, 6.3):
        ax.annotate("", xy=(x + 0.25, 1.75), xytext=(x - 0.15, 1.75),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#5b6475"))
    ax.text(5, 0.35,
            "Current code & experiments live entirely in Phase 1; they validate the continuous-time backbone the later phases need.",
            ha="center", fontsize=9.5, color="#5b6475")
    fig.savefig(IMG / "roadmap.png"); plt.close(fig)


def fig_diffusion_sketch():
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    t = np.linspace(0, 1, 200)
    x0 = np.sin(2 * np.pi * t) * np.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
    for i, (a, lab) in enumerate([(1.0, "x₀"), (0.55, "x_t"), (0.0, "x_T≈ε")]):
        noise = np.random.default_rng(i + 3).normal(0, 1, size=len(t))
        y = a * x0 + math.sqrt(max(1 - a * a, 0)) * noise * 0.55
        axes[0].plot(t, y - i * 2.2, lw=1.5, label=lab)
    axes[0].annotate("forward q(x_t|x₀)\nadd noise", xy=(0.82, -1.5), xytext=(0.55, 1.0),
                     fontsize=8, color="#5b6475",
                     arrowprops=dict(arrowstyle="->", color="#5b6475"))
    axes[0].annotate("reverse p_θ(x_{t-1}|x_t)\ndenoise", xy=(0.18, -3.2), xytext=(0.35, -4.6),
                     fontsize=8, color="#2f6fdf",
                     arrowprops=dict(arrowstyle="->", color="#2f6fdf"))
    axes[0].set_title("Continuous diffusion (DDPM-style)")
    axes[0].set_xlabel("coordinate / time"); axes[0].set_yticks([])
    axes[0].legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.3, 4.2); ax.axis("off")
    ax.set_title("Discrete language diffusion (fixed slots)")

    def row(y, chars, mask, lab):
        ax.text(-0.3, y + 0.35, lab, ha="right", va="center", fontsize=8, color="#5b6475")
        for i, (ch, m) in enumerate(zip(chars, mask)):
            face = "#cfcfcf" if m else "#4c78a8"
            ax.add_patch(FancyBboxPatch((i * 0.85, y), 0.75, 0.7,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=face, edgecolor="k", lw=0.8))
            ax.text(i * 0.85 + 0.375, y + 0.35, "M" if m else ch, ha="center", va="center",
                    color="#333" if m else "white", fontsize=9, fontweight="bold")

    chars = list("HELLOWORLD!")
    row(3.1, chars, [0] * 11, "x₀")
    row(1.7, chars, [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0], "x_t")
    row(0.3, chars, [1] * 11, "x_T")
    ax.text(5, -0.15, "Length K is fixed; a mask cannot insert or reorder tokens.",
            ha="center", fontsize=9, color="#8a4b00")
    fig.tight_layout(); fig.savefig(IMG / "diffusion_sketch.png"); plt.close(fig)


def fig_signal_diffusion():
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1.15], hspace=0.55, wspace=0.35)
    rng = np.random.default_rng(0)
    K = 12
    tokens0 = rng.integers(0, 8, size=K)
    noise = rng.random(K)
    tokens_mid = tokens0.copy()
    tokens_mid[noise > 0.45] = rng.integers(0, 8, size=(noise > 0.45).sum())
    tokens_noise = rng.integers(0, 8, size=K)

    ax = fig.add_subplot(gs[0, :5])
    ax.set_xlim(-0.5, 14.5); ax.set_ylim(-1.8, 2.8); ax.axis("off")
    ax.set_title("Discrete language diffusion: tokens live on a fixed-length lattice", fontsize=12, pad=8)

    def draw_seq(ax, y, tokens, title):
        for i, v in enumerate(tokens):
            ax.add_patch(Rectangle((i, y), 0.85, 0.85, facecolor=plt.cm.tab10(v / 10),
                                   edgecolor="k", linewidth=1.2))
            ax.text(i + 0.42, y + 0.42, str(v), ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        ax.text(-0.3, y + 0.4, title, ha="right", va="center", fontsize=9)

    draw_seq(ax, 1.5, tokens0, "clean x₀")
    draw_seq(ax, 0.2, tokens_mid, "noisy xₜ")
    draw_seq(ax, -1.1, tokens_noise, "noise x_T")
    ax.annotate("", xy=(12.2, 1.9), xytext=(12.2, -0.7),
                arrowprops=dict(arrowstyle="<->", color="#5b6475", lw=1.5))
    ax.text(12.5, 0.6, "fixed K slots\npositions locked", fontsize=9, color="#5b6475", va="center")
    ax.text(6, 2.55, "Limitation: length is chosen in advance; tokens cannot insert, delete, or shift.",
            ha="center", fontsize=10, color="#8a4b00",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff6e5", edgecolor="#d98b00"))

    t = np.linspace(0, 1, 400)
    sig = 0.55 * np.sin(2 * math.pi * 1.5 * t) + 0.35 * np.cos(2 * math.pi * 3.2 * t) + 0.15 * np.sin(2 * math.pi * 5 * t)
    noise_sig = rng.normal(0, 1.0, size=400)
    panels = [
        (sig, "clean signal  X̄₀(t)", "C0"),
        (0.55 * sig + 0.45 * noise_sig, "noisy signal  X̄_τ(t)", "C1"),
        (noise_sig, "noise  X̄_T(t)", "0.45"),
    ]
    for i, (y, title, c) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t, y, color=c, lw=1.6)
        ax.set_xlim(0, 1); ax.set_ylim(-2.8, 2.8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("t ∈ [0,1]"); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("ℝᵈ")

    ax = fig.add_subplot(gs[1, 3:])
    ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.35); ax.axis("off")
    ax.set_title("Decode at any length K'  →  tokens of arbitrary length", fontsize=10)
    for Kp, y0, col in [(6, 0.95, "#1f77b4"), (12, 0.55, "#ff7f0e"), (24, 0.15, "#2ca02c")]:
        tt = np.arange(Kp) / Kp
        yy = 0.55 * np.sin(2 * math.pi * 1.5 * tt) + 0.35 * np.cos(2 * math.pi * 3.2 * tt) + 0.15 * np.sin(2 * math.pi * 5 * tt)
        toks = ((yy - yy.min()) / (yy.max() - yy.min() + 1e-9) * 7).astype(int)
        for i, v in enumerate(toks):
            ax.add_patch(Rectangle((i / Kp, y0), 0.7 / Kp, 0.22, facecolor=plt.cm.tab10(v / 10),
                                   edgecolor="k", linewidth=0.6))
        ax.text(1.02, y0 + 0.11, f"K'={Kp}", va="center", fontsize=9, color=col)

    fig.suptitle("Thesis idea: diffuse continuous-time signals, then sample a sequence of any length",
                 fontsize=13, y=0.98)
    fig.savefig(IMG / "signal_diffusion.png"); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_roadmap, fig_diffusion_sketch, fig_signal_diffusion,
               fig_task, fig_bayes_vs_K, fig_basis, fig_attention_compare, fig_conditioning,
               fig_training_curves, fig_compare_bars, fig_ablations, fig_cost, copy_existing):
        fn(); print("done", fn.__name__)
