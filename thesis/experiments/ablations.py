"""One-at-a-time ablations around the base configuration.

    python experiments/ablations.py --config configs/position_only.yaml --steps 2000
    python experiments/ablations.py --only n_basis ridge
"""
import argparse
from pathlib import Path

from common import OUT, ROOT, default_workers, print_rows, run_many, write_rows

# (axis, attentions it applies to, values). Avoid n_basis == K: that basis is nearly singular.
AXES = [
    ("n_basis", ["continuous"], [8, 16, 32, 64, 96]),
    ("basis_sigma", ["continuous"], [0.01, 0.02, 0.05, 0.1]),
    ("ridge", ["continuous"], [1e-6, 1e-3, 1e-1, 1.0, 10.0]),
    ("n_head", ["discrete", "continuous"], [1, 2, 4, 8]),
    ("pooling", ["discrete", "continuous"], ["mean", "cls"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    name = Path(args.config).stem
    jobs, meta = [], []
    for axis, attns, values in AXES:
        if args.only and axis not in args.only:
            continue
        for attn in attns:
            for v in values:
                train = {"seed": args.seed, "out_dir": f"{ROOT}/runs/ablation_{name}/{attn}_{axis}={v}_s{args.seed}"}
                if args.steps:
                    train["steps"] = args.steps
                jobs.append({"model": {"attention": attn, axis: v}, "train": train})
                meta.append((axis, attn, v))

    results = run_many(args.config, jobs, args.workers or default_workers(args.threads), args.threads)
    rows = []
    for (axis, attn, v), r in zip(meta, results):
        e = r["eval"][str(r["train_K"])]
        rows.append(dict(axis=axis, value=v, attention=attn, acc=e["acc"], bayes=e["bayes_acc"], gap=e["gap"],
                         step_ms=1000 * r["bench"]["step_time_s"], params=r["params"]))
    print_rows(rows)
    write_rows(OUT / f"ablations_{name}.csv", rows)


if __name__ == "__main__":
    main()
