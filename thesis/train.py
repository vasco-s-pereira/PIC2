"""Train a sequence classifier on on-the-fly synthetic data.

    python train.py --config configs/base.yaml model.attention=continuous train.steps=2000

Every run writes to out_dir: config.yaml, task.json, train_log.csv, eval_log.csv, results.json, model.pt.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from data.synthetic import SyntheticTask, TaskConfig, run_baselines
from model.classifier import ModelConfig, SequenceClassifier

DEFAULTS = {
    "task": asdict(TaskConfig()),
    "model": asdict(ModelConfig()),
    "train": dict(
        K=128, steps=3000, batch_size=64, lr=1e-3, min_lr=1e-4, warmup=100,
        weight_decay=0.01, betas=[0.9, 0.99], grad_clip=1.0,
        eval_every=500, eval_size=4096, eval_K=None,   # None -> [K]
        baselines=False, seed=0, device="auto", threads=None,
        out_dir="runs/default", log_every=50, save_model=True,
    ),
}


# ---------------------------------------------------------------------------- config

def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, val = item.split("=", 1)
        node = out
        *path, leaf = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return out


def load_config(path: str | None = None, overrides: dict | list | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, parse_overrides(overrides) if isinstance(overrides, list) else copy.deepcopy(overrides))
    cfg["model"]["vocab_size"] = cfg["task"]["vocab_size"]
    cfg["model"]["n_classes"] = cfg["task"]["n_classes"]
    return cfg


def _dataclass_from(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


# ---------------------------------------------------------------------------- utils

def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup"]:
        return tc["lr"] * (step + 1) / tc["warmup"]
    p = (step - tc["warmup"]) / max(1, tc["steps"] - tc["warmup"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * min(p, 1.0)))


def activation_bytes(model, tokens, labels) -> int:
    """Bytes of tensors saved for backward in one forward pass (device-independent memory proxy)."""
    total = 0
    seen = set()

    def pack(t):
        nonlocal total
        key = (t.data_ptr(), t.shape, t.dtype)
        if key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        F.cross_entropy(model(tokens), labels)
    return total


def benchmark_step(model, task, K, batch_size, device, n_iter=10, warmup=3) -> dict:
    """Forward+backward wall time per step and memory at length K."""
    model.train()
    g = torch.Generator().manual_seed(0)
    x, y = task.sample(batch_size, K, generator=g)
    x, y = x.to(device), y.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup + n_iter):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    out = {"step_time_s": (time.perf_counter() - t0) / n_iter,
           "activation_MB": activation_bytes(model, x, y) / 2**20}
    if device.type == "cuda":
        out["peak_cuda_MB"] = torch.cuda.max_memory_allocated(device) / 2**20
    model.zero_grad(set_to_none=True)
    return out


class Evaluator:
    """Fixed evaluation sets per K; model and Bayes-optimal scored on the same sequences."""

    def __init__(self, task: SyntheticTask, Ks, n: int, seed: int):
        self.task, self.sets = task, {}
        for K in Ks:
            g = torch.Generator().manual_seed(10_000 + seed * 7919 + K)
            x, y = task.sample(n, K, generator=g)
            bayes = (task.bayes_predict(x) == y).double().mean().item()
            self.sets[K] = (x, y, bayes)

    @torch.no_grad()
    def __call__(self, model, device, batch=512) -> dict:
        model.eval()
        out = {}
        for K, (x, y, bayes) in self.sets.items():
            correct, loss = 0, 0.0
            for i in range(0, len(y), batch):
                xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
                logits = model(xb)
                loss += F.cross_entropy(logits, yb, reduction="sum").item()
                correct += (logits.argmax(-1) == yb).sum().item()
            acc = correct / len(y)
            se = math.sqrt(max(bayes * (1 - bayes), 1e-4) / len(y))
            if acc > bayes + 4 * se:
                print(f"  WARNING: model beats Bayes-optimal at K={K} ({acc:.4f} > {bayes:.4f}): bug?")
            out[K] = dict(acc=acc, loss=loss / len(y), bayes_acc=bayes, gap=bayes - acc)
        model.train()
        return out


# ---------------------------------------------------------------------------- main loop

def run(cfg: dict, verbose: bool = True) -> dict:
    tc = cfg["train"]
    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if tc.get("threads"):
        torch.set_num_threads(tc["threads"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if tc["device"] == "auto" else tc["device"])

    task = SyntheticTask(_dataclass_from(TaskConfig, cfg["task"]))
    torch.manual_seed(tc["seed"])
    model = SequenceClassifier(_dataclass_from(ModelConfig, cfg["model"])).to(device)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "task.json", "w") as f:
        json.dump({**task.describe(), "f_params": task.f_params}, f, indent=1)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=tuple(tc["betas"]))

    K = tc["K"]
    eval_K = tc["eval_K"] or [K]
    evaluator = Evaluator(task, sorted(set(eval_K) | {K}), tc["eval_size"], tc["seed"])
    gen = torch.Generator().manual_seed(tc["seed"])
    if verbose:
        print(f"[{out_dir}] {cfg['model']['attention']} | params {model.num_params():,} | device {device} | "
              f"Bayes@K={K}: {evaluator.sets[K][2]:.4f}")

    train_f = open(out_dir / "train_log.csv", "w", newline="")
    eval_f = open(out_dir / "eval_log.csv", "w", newline="")
    tw, ew = csv.writer(train_f), csv.writer(eval_f)
    tw.writerow(["step", "loss", "acc", "lr", "step_time_s"])
    ew.writerow(["step", "K", "acc", "loss", "bayes_acc", "gap"])

    step_times = []
    model.train()
    for step in range(tc["steps"] + 1):
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            ev = evaluator(model, device)
            for k, r in ev.items():
                ew.writerow([step, k, r["acc"], r["loss"], r["bayes_acc"], r["gap"]])
            eval_f.flush()
            if verbose:
                print(f"  step {step:5d} eval " + " ".join(
                    f"K={k}: acc {r['acc']:.4f} (Bayes {r['bayes_acc']:.4f})" for k, r in ev.items()))
        if step == tc["steps"]:
            break
        lr = lr_at(step, tc)
        for gp in opt.param_groups:
            gp["lr"] = lr
        t0 = time.perf_counter()
        x, y = task.sample(tc["batch_size"], K, generator=gen)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= 10:
            step_times.append(dt)
        if step % tc["log_every"] == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            tw.writerow([step, loss.item(), acc, lr, dt])
            train_f.flush()
    train_f.close(); eval_f.close()

    results = {
        "out_dir": str(out_dir),
        "attention": cfg["model"]["attention"],
        "params": model.num_params(),
        "train_K": K,
        "eval": {str(k): v for k, v in ev.items()},
        "mean_step_time_s": sum(step_times) / max(1, len(step_times)),
        "bench": benchmark_step(model, task, K, tc["batch_size"], device),
    }
    if tc["baselines"]:
        results["baselines"] = run_baselines(task, K, seed=tc["seed"])
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    if tc["save_model"]:
        torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "model.pt")
    return results


def load_trained(run_dir: str | os.PathLike, device="cpu"):
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SequenceClassifier(_dataclass_from(ModelConfig, cfg["model"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), SyntheticTask(_dataclass_from(TaskConfig, cfg["task"])), cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted key=value, e.g. model.attention=continuous")
    args = ap.parse_args()
    res = run(load_config(args.config, args.overrides))
    print(json.dumps({k: v for k, v in res.items() if k != "out_dir"}, indent=1))
