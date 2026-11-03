"""Sweep the sequence length K: gap to Bayes-optimal, time per step, memory.

Variants: discrete; continuous with fixed N_b; continuous with N_b proportional to K.

    python experiments/sweep_K.py --bench_only                    # cost only, untrained models (fast)
    python experiments/sweep_K.py --steps 3000                    # full sweep
    python experiments/sweep_K.py --constant_ceiling 0.9          # recalibrate Q's so Bayes ~0.9 at every K
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import OUT, ROOT, default_workers, print_rows, run_many, write_rows
from data.synthetic import SyntheticTask, TaskConfig, calibrate_signal_scale
from model.classifier import ModelConfig, SequenceClassifier
from train import benchmark_step, load_config

VARIANT_ORDER = ("discrete", "cont_fixed", "cont_prop")


def variants(K, args):
    return [("discrete", {"attention": "discrete"}),
            ("cont_fixed", {"attention": "continuous", "n_basis": args.nb_fixed}),
            ("cont_prop", {"attention": "continuous", "n_basis": max(4, int(round(args.nb_ratio * K)))})]


def bench(args):
    base = load_config(args.config)
    task = SyntheticTask(TaskConfig(**base["task"]))
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = args.batch_size or base["train"]["batch_size"]
    rows = []
    for K in args.Ks:
        for name, mo in variants(K, args):
            torch.manual_seed(0)
            m = SequenceClassifier(ModelConfig(**{**base["model"], **mo})).to(device)
            b = benchmark_step(m, task, K, bs, device, n_iter=10, warmup=3)
            rows.append(dict(variant=name, K=K, n_basis=mo.get("n_basis", ""), device=device.type,
                             step_ms=1000 * b["step_time_s"], act_MB=b["activation_MB"],
                             peak_cuda_MB=b.get("peak_cuda_MB", "")))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print_rows(rows[-1:])
    return rows


def train_sweep(args, tag):
    jobs, meta = [], []
    for K in args.Ks:
        task_over = {}
        if args.constant_ceiling:
            s = calibrate_signal_scale(TaskConfig(**load_config(args.config)["task"]), K, args.constant_ceiling)
            task_over = {"signal_scale": s}
            print(f"K={K}: signal_scale={s:.4f}")
        for name, mo in variants(K, args):
            train = {"K": K, "out_dir": f"{ROOT}/runs/{tag}/{name}_K{K}"}
            if args.steps:
                train["steps"] = args.steps
            if args.batch_size:
                train["batch_size"] = args.batch_size
            jobs.append({"task": task_over, "model": mo, "train": train})
            meta.append((name, K, mo.get("n_basis", "")))
    results = run_many(args.config, jobs, args.workers or default_workers(args.threads), args.threads)
    rows = []
    for (name, K, nb), job, r in zip(meta, jobs, results):
        e = r["eval"][str(K)]
        rows.append(dict(variant=name, K=K, n_basis=nb, signal_scale=job["task"].get("signal_scale", 1.0),
                         acc=e["acc"], bayes=e["bayes_acc"], gap=e["gap"],
                         step_ms=1000 * r["bench"]["step_time_s"], act_MB=r["bench"]["activation_MB"],
                         peak_cuda_MB=r["bench"].get("peak_cuda_MB", "")))
    print_rows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--Ks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--nb_fixed", type=int, default=32)
    ap.add_argument("--nb_ratio", type=float, default=0.25, help="N_b = ratio * K for the proportional variant")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--constant_ceiling", type=float, default=None)
    ap.add_argument("--bench_only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    tag = "sweepK" + (f"_ceil{args.constant_ceiling}" if args.constant_ceiling else "")
    if args.bench_only:
        tag += "_bench"
    rows = bench(args) if args.bench_only else train_sweep(args, tag)
    write_rows(OUT / f"{tag}.csv", rows)

    metrics = [("step_ms", "time / step (ms)"), ("act_MB", "saved activations (MB)")]
    if all(isinstance(r.get("peak_cuda_MB"), float) for r in rows):
        metrics.append(("peak_cuda_MB", "peak CUDA memory (MB)"))
    if not args.bench_only:
        metrics = [("gap", "gap to Bayes-optimal")] + metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (key, label) in zip(axes, metrics):
        for name in VARIANT_ORDER:
            pts = [(r["K"], r[key]) for r in rows if r["variant"] == name]
            ax.plot(*zip(*pts), "o-", label=name)
        ax.set_xscale("log", base=2); ax.set_xlabel("K"); ax.set_ylabel(label)
        if key != "gap":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}.png", dpi=130)
    print(f"wrote {OUT / tag}.png")


if __name__ == "__main__":
    main()
