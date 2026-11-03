import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import load_config, run  # noqa: E402

OUT = ROOT / "experiments" / "out"


def _job(args):
    config_path, overrides, threads = args
    cfg = load_config(config_path, overrides)
    cfg["train"]["threads"] = threads
    res_path = Path(cfg["train"]["out_dir"]) / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            return json.load(f)
    return run(cfg)


def run_many(config_path, jobs: list[dict], workers: int = 4, threads: int = 4) -> list[dict]:
    """jobs: nested override dicts, each must set train.out_dir. Finished runs are reused."""
    args = [(str(config_path), j, threads) for j in jobs]
    if workers <= 1:
        return [_job(a) for a in args]
    pool = mp.get_context("spawn").Pool(workers)
    try:
        results = pool.map(_job, args, chunksize=1)
    finally:
        pool.close()
        pool.join()
    return results


def write_rows(path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def print_rows(rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    widths = {k: max(len(k), *(len(fmt(r.get(k, ""))) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(fmt(r.get(k, "")).ljust(widths[k]) for k in keys))


def default_workers(threads: int = 4) -> int:
    return max(1, (os.cpu_count() or 4) // threads)
