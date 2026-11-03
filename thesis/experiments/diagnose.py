"""Where do the continuous-attention heads look? Overlay the learned densities on f_n(t).

    python experiments/diagnose.py runs/compare_position_only/continuous_s0

Per layer and head: (top) the average attention density (1/K) sum_i N(t; mu_i, sigma_i^2) over a
batch, against the positional information I(n; x_t) (where the classes differ); (bottom) mu_i vs
the query position t_i, coloured by sigma_i.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import load_trained  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--K", type=int, default=None)
ap.add_argument("--n", type=int, default=256)
args = ap.parse_args()

model, task, cfg = load_trained(args.run_dir)
assert cfg["model"]["attention"] == "continuous", "diagnose needs a continuous-attention run"
K = args.K or cfg["train"]["K"]
x, y = task.sample(args.n, K, generator=torch.Generator().manual_seed(0))
model.set_record(True)
with torch.no_grad():
    model(x)
dens = model.recorded_densities()
n_prefix = 1 if cfg["model"]["pooling"] == "cls" else 0

grid = np.linspace(0, 1, 400)
tq = np.arange(K) / K
f = task.f(grid)
info = task.positional_information(K)
L, H = len(dens), dens[0][0].shape[1]

fig = plt.figure(figsize=(3.2 * H, 2.6 * (2 * L + 1)))
gs = fig.add_gridspec(2 * L + 1, H)
ax = fig.add_subplot(gs[0, :])
for n in range(task.n_classes):
    ax.plot(grid, f[n], lw=1, label=f"f_{n}")
ax2 = ax.twinx()
ax2.fill_between(tq, info, color="k", alpha=0.15, label="I(n; x_t)")
ax.set_title("class mixing functions f_n(t) and positional information I(n; x_t) (shaded)")
ax.set_xlim(0, 1); ax.legend(ncol=task.n_classes, fontsize=7, loc="upper right")

for l, (mu, var) in enumerate(dens):
    mu, sd = mu[:, :, n_prefix:].numpy(), var[:, :, n_prefix:].sqrt().numpy()
    for h in range(H):
        m, s = mu[:, h].reshape(-1), sd[:, h].reshape(-1)
        avg = np.exp(-0.5 * ((grid[:, None] - m[None]) / s[None]) ** 2) / (s[None] * np.sqrt(2 * np.pi))
        a = fig.add_subplot(gs[1 + 2 * l, h])
        a.plot(grid, avg.mean(1), color="C3")
        b = a.twinx(); b.fill_between(tq, info, color="k", alpha=0.15); b.set_yticks([])
        a.set_xlim(0, 1); a.set_title(f"layer {l} head {h}: mean density", fontsize=9)
        c = fig.add_subplot(gs[2 + 2 * l, h])
        sc = c.scatter(np.tile(tq, args.n)[:4000], m[:4000], c=s[:4000], s=2, cmap="viridis")
        c.set_xlim(0, 1); c.set_ylim(0, 1); c.set_xlabel("query position t_i"); c.set_ylabel("mu_i")
        fig.colorbar(sc, ax=c, label="sigma_i")

fig.tight_layout()
out = Path(args.run_dir) / "diagnose.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
