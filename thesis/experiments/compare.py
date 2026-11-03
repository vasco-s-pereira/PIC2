"""Discrete vs continuous attention vs Bayes-optimal and baselines on one task configuration.

    python experiments/compare.py --config configs/easy.yaml           # sanity check
    python experiments/compare.py --config configs/position_only.yaml  # position-only variant
"""
import argparse
from pathlib import Path

from common import OUT, ROOT, default_workers, print_rows, run_many, write_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    name = Path(args.config).stem
    jobs = []
    for seed in args.seeds:
        for attn in ("discrete", "continuous"):
            train = {"out_dir": f"{ROOT}/runs/compare_{name}/{attn}_s{seed}", "seed": seed, "baselines": True}
            if args.steps:
                train["steps"] = args.steps
            jobs.append({"model": {"attention": attn}, "train": train})

    results = run_many(args.config, jobs, args.workers or default_workers(args.threads), args.threads)
    rows = []
    for job, r in zip(jobs, results):
        K = str(r["train_K"])
        e, b = r["eval"][K], r["baselines"]
        rows.append(dict(attention=r["attention"], seed=job["train"]["seed"], K=int(K),
                         acc=e["acc"], bayes=e["bayes_acc"], gap=e["gap"],
                         unigram_lr=b["unigram"], binned_lr=b["binned"], chance=b["chance"],
                         step_ms=1000 * r["bench"]["step_time_s"], act_MB=r["bench"]["activation_MB"]))
    print_rows(rows)
    write_rows(OUT / f"compare_{name}.csv", rows)


if __name__ == "__main__":
    main()
