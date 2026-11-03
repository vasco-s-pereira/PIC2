"""Length generalisation: train at K=128, evaluate at other K (the classes are the same for every K).

    python experiments/length_gen.py --config configs/position_only.yaml
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, ROOT, default_workers, print_rows, run_many, write_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--train_K", type=int, default=128)
    ap.add_argument("--eval_Ks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    name = Path(args.config).stem
    jobs = []
    for seed in args.seeds:
        for attn in ("discrete", "continuous"):
            train = {"K": args.train_K, "eval_K": args.eval_Ks, "seed": seed,
                     "out_dir": f"{ROOT}/runs/lengthgen_{name}/{attn}_K{args.train_K}_s{seed}"}
            if args.steps:
                train["steps"] = args.steps
            jobs.append({"model": {"attention": attn}, "train": train})

    results = run_many(args.config, jobs, args.workers or default_workers(args.threads), args.threads)
    rows = []
    for job, r in zip(jobs, results):
        for K, e in sorted(r["eval"].items(), key=lambda kv: int(kv[0])):
            rows.append(dict(attention=r["attention"], seed=job["train"]["seed"], train_K=args.train_K,
                             eval_K=int(K), acc=e["acc"], bayes=e["bayes_acc"], gap=e["gap"]))
    print_rows(rows)
    write_rows(OUT / f"lengthgen_{name}.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    s0 = args.seeds[0]
    for attn in ("discrete", "continuous"):
        pts = sorted((r["eval_K"], r["acc"], r["gap"]) for r in rows if r["attention"] == attn and r["seed"] == s0)
        Ks, acc, gap = zip(*pts)
        axes[0].plot(Ks, acc, "o-", label=attn)
        axes[1].plot(Ks, gap, "o-", label=attn)
    bayes = sorted({(r["eval_K"], r["bayes"]) for r in rows if r["seed"] == s0})
    axes[0].plot(*zip(*bayes), "k--", label="Bayes-optimal")
    for ax, lab in zip(axes, ("accuracy", "gap to Bayes-optimal")):
        ax.axvline(args.train_K, color="gray", ls=":")
        ax.set_xscale("log", base=2); ax.set_xlabel("eval K"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"lengthgen_{name}.png", dpi=130)
    print(f"wrote {OUT / f'lengthgen_{name}.png'}")


if __name__ == "__main__":
    main()
