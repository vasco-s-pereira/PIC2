"""Sweep the basis count N_b downward and compare with the discrete transformer.

Same protocol as sweep_K (4000 steps, batch 32). Finished runs are reused.

    python experiments/sweep_nb.py --steps 4000 --batch_size 32
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, ROOT, default_workers, print_rows, run_many, write_rows


DISCRETE_CSV = OUT / "sweepK.csv"
# Accuracy within this of the discrete model (and of N_b=32 continuous) still "holds".
HOLD_TOL = 0.02


def train_sweep(args, tag):
    jobs, meta = [], []
    for K in args.Ks:
        for nb in args.nbs:
            jobs.append({
                "model": {"attention": "continuous", "n_basis": nb},
                "train": {
                    "K": K,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "eval_batch": args.eval_batch,
                    "out_dir": f"{ROOT}/runs/{tag}/cont_Nb{nb}_K{K}",
                },
            })
            meta.append((K, nb))
    workers = args.workers or default_workers(args.threads)
    # K=1024/2048 peaks near the 8 GB card even at small N_b, so those run one at a time.
    small = [(j, m) for j, m in zip(jobs, meta) if m[0] <= 512]
    large = [(j, m) for j, m in zip(jobs, meta) if m[0] > 512]
    results, meta_out = [], []
    for group, n in ((small, workers), (large, 1)):
        if not group:
            continue
        gj, gm = zip(*group)
        results.extend(run_many(args.config, list(gj), n, args.threads))
        meta_out.extend(gm)
    meta = meta_out
    rows = []
    for (K, nb), r in zip(meta, results):
        e = r["eval"][str(K)]
        rows.append(dict(
            variant="continuous", K=K, n_basis=nb,
            acc=e["acc"], bayes=e["bayes_acc"], gap=e["gap"],
            step_ms=1000 * r["bench"]["step_time_s"],
            act_MB=r["bench"]["activation_MB"],
            peak_cuda_MB=r["bench"].get("peak_cuda_MB", ""),
        ))
    print_rows(rows)
    return rows


def load_reference():
    """Discrete and fixed-N_b=32 continuous accuracies from the original sweep."""
    ref = {}
    if not DISCRETE_CSV.exists():
        return ref
    with open(DISCRETE_CSV) as f:
        for r in csv.DictReader(f):
            ref.setdefault(int(r["K"]), {})[r["variant"]] = float(r["acc"])
    return ref


def _lowest(pts, anchor):
    if anchor is None:
        return ""
    ok = [p for p in pts if p["acc"] >= anchor - HOLD_TOL]
    return "" if not ok else min(p["n_basis"] for p in ok)


def holds_table(rows, ref):
    """Smallest N_b at each K within HOLD_TOL of the discrete model and of N_b=32."""
    by_K = {}
    for r in rows:
        by_K.setdefault(r["K"], []).append(r)
    summary = []
    for K in sorted(by_K):
        pts = sorted(by_K[K], key=lambda r: r["n_basis"])
        discrete = ref.get(K, {}).get("discrete")
        fixed = ref.get(K, {}).get("cont_fixed")
        best = max(pts, key=lambda p: p["acc"])
        summary.append(dict(
            K=K,
            discrete=discrete if discrete is not None else "",
            cont_Nb32=fixed if fixed is not None else "",
            acc_Nb1=pts[0]["acc"],
            best_acc=best["acc"],
            best_Nb=best["n_basis"],
            lowest_vs_discrete=_lowest(pts, discrete),
            lowest_vs_Nb32=_lowest(pts, fixed),
        ))
    return summary


def plot(rows, ref, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    Ks = sorted({r["K"] for r in rows})
    for K in Ks:
        pts = sorted((r["n_basis"], r["acc"]) for r in rows if r["K"] == K)
        axes[0].plot(*zip(*pts), "o-", label=f"K={K}")
        gaps = sorted((r["n_basis"], r["gap"]) for r in rows if r["K"] == K)
        axes[1].plot(*zip(*gaps), "o-", label=f"K={K}")
        d = ref.get(K, {}).get("discrete")
        if d is not None:
            axes[0].plot(pts[0][0], d, "k|", markersize=12, markeredgewidth=2)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("gap to Bayes-optimal")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"$N_b$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("accuracy (black ticks: discrete)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--Ks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--nbs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="sweepNb")
    args = ap.parse_args()

    rows = train_sweep(args, args.tag)
    write_rows(OUT / f"{args.tag}.csv", rows)
    ref = load_reference()
    summary = holds_table(rows, ref)
    print_rows(summary)
    write_rows(OUT / f"{args.tag}_bound.csv", summary)
    plot(rows, ref, OUT / f"{args.tag}.png")


if __name__ == "__main__":
    main()
