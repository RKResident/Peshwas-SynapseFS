#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path

# Suppress Matplotlib cache directory warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt

# Tell pytest to ignore this file during test discovery
__test__ = False

# Import SynapseFS components
from synapsefs.store.repo import Repo
from synapsefs import graph
from synapsefs.fuse.cache import ChunkCache
from synapsefs.fuse.reconstruct import VirtualSafetensorsFile
from synapsefs.codec import chunk

# Register F32 dynamically if missing so tests work without mutating codebase files
if "F32" not in chunk._DTYPES:
    chunk._DTYPES["F32"] = (4, chunk.FLOAT)
    chunk._UINT_OF[4] = np.uint32

import re


def parse_cache_size(value) -> int:
    if isinstance(value, int):
        return max(0, value)
    s = str(value).strip()
    if not s or s == "0":
        return 0
    if s.isdigit():
        return int(s)
    units = {
        "b": 1, "k": 1024, "kb": 1000, "kib": 1024, "m": 1024 * 1024,
        "mb": 1000 * 1000, "mib": 1024 * 1024, "g": 1024 * 1024 * 1024,
        "gb": 1000 * 1000 * 1000, "gib": 1024 * 1024 * 1024,
    }
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?$", s)
    if not m:
        return 512 * 1024 * 1024
    num_str, unit_str = m.groups()
    num = float(num_str)
    if unit_str:
        unit = unit_str.lower()
        if unit in units:
            return int(num * units[unit])
    return int(num)


def run_benchmark(repo_path: str = "examples/mnist_repo", max_trials: int = 100,
                  cache_size_str: str = "512MiB", output_png: str = "bench-results/caching_speedup_curve.png"):
    cache_bytes = parse_cache_size(cache_size_str)

    print("=" * 75)
    print(f" SynapseFS Cache Benchmark: Repeated Reads (1 to {max_trials}) vs Speedup")
    print(f" Repository: {repo_path} | Cache Size: {cache_size_str} ({cache_bytes:,} bytes)")
    print("=" * 75)

    repo = Repo.find(repo_path)
    _, commit_hash = repo.read_head()
    checkpoint = graph.CommitCheckpoint(repo.store, commit_hash)

    # 1. Without Cache
    vfile_no_cache = VirtualSafetensorsFile(checkpoint, cache=None)
    size = vfile_no_cache.total_size

    # 2. With LRU Cache
    cache = ChunkCache(max_bytes=cache_bytes)
    vfile_cached = VirtualSafetensorsFile(checkpoint, cache=cache)

    print(f"Measuring {max_trials} sequential reads (Model size: {size / 1024:.1f} KiB)...")

    # Record individual read timings
    nocache_times = []
    for _ in range(max_trials):
        t0 = time.perf_counter()
        _ = vfile_no_cache.read(0, size)
        nocache_times.append(time.perf_counter() - t0)

    cached_times = []
    for _ in range(max_trials):
        t0 = time.perf_counter()
        _ = vfile_cached.read(0, size)
        cached_times.append(time.perf_counter() - t0)

    # Compute cumulative times and speedup for each N in [1, max_trials]
    trials = np.arange(1, max_trials + 1)
    cum_nocache = np.cumsum(nocache_times)
    cum_cached = np.cumsum(cached_times)
    speedups = cum_nocache / cum_cached

    avg_cold = cached_times[0] * 1000
    avg_warm = np.mean(cached_times[1:]) * 1000
    avg_nocache_ms = np.mean(nocache_times) * 1000
    asymptotic_speedup = avg_nocache_ms / avg_warm

    if max_trials <= 50:
        milestones = [1, 5, 10, 25, max_trials]
    elif max_trials <= 200:
        milestones = [1, 10, 25, 50, max_trials]
    else:
        milestones = [1, 10, 50, max_trials // 2, max_trials]
    milestones = sorted(list(set(m for m in milestones if 1 <= m <= max_trials)))

    print("\nBenchmark Summary:")
    print(f"  • Without Cache Avg Latency: {avg_nocache_ms:6.2f} ms / read")
    print(f"  • With Cache Cold Latency:   {avg_cold:6.2f} ms (1st read)")
    print(f"  • With Cache Warm Latency:   {avg_warm:6.2f} ms / read (subsequent)")
    for m in milestones:
        print(f"  • Speedup at N={m:<5}:        {speedups[m - 1]:6.2f}x")
    print(f"  • Asymptotic Ceiling:        {asymptotic_speedup:6.2f}x")

    # -------------------------------------------------------------
    # Plotting (2-Panel High-Res Figure)
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Plot 1: TRIALS vs Speedup
    ax1.plot(trials, speedups, color="#2563eb", linewidth=2.5, label=f"Cumulative Speedup (Cache: {cache_size_str})")
    ax1.axhline(asymptotic_speedup, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Asymptotic Upper Bound ({asymptotic_speedup:.1f}x)")
    ax1.set_title(f"Cache Speedup vs Number of Reads / Trials ({cache_size_str} Cap)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Repeated Model Reads (Trials $N$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speedup Multiplier ($\\times$)", fontsize=11, fontweight="bold")
    ax1.set_xlim(1, max_trials)
    ax1.set_ylim(0, max(speedups) * 1.18)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Annotate key milestones
    for pt in milestones:
        idx = pt - 1
        ax1.scatter(pt, speedups[idx], color="#1d4ed8", s=45, zorder=5)
        ax1.annotate(f"{speedups[idx]:.1f}x", (pt, speedups[idx]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=9, fontweight="bold", color="#1e40af")

    # Plot 2: Cumulative Latency Comparison
    ax2.plot(trials, cum_nocache, color="#ef4444", linewidth=2.2, label="Without Cache ($O(N)$ Decompression)")
    ax2.plot(trials, cum_cached, color="#10b981", linewidth=2.2, label=f"With LRU Cache [{cache_size_str}] ($O(1)$ Amortized)")
    ax2.set_title(f"Cumulative Load Time ({cache_size_str} LRU Cache)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Repeated Model Reads (Trials $N$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Time Elapsed (seconds)", fontsize=11, fontweight="bold")
    ax2.set_xlim(1, max_trials)
    ax2.set_ylim(0, max(cum_nocache) * 1.05)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved benchmark plot to: {out_path.resolve()}")
    plt.close()


def main():
    default_cache = os.environ.get("CACHE_SIZE", "512MiB")
    parser = argparse.ArgumentParser(description="Plot SynapseFS caching speedup vs number of trials")
    parser.add_argument("-C", "--repo", default="examples/mnist_repo", help="Path to repository (default: examples/mnist_repo)")
    parser.add_argument("-N", "--trials", type=int, default=100, help="Number of repeated reads/trials (default: 100)")
    parser.add_argument("--cache-size", default=default_cache, help=f"Cache size cap (e.g. 512MiB, 32MiB, 1GiB, default: {default_cache})")
    parser.add_argument("-o", "--out", default="bench-results/caching_speedup_curve.png", help="Output PNG path (default: bench-results/caching_speedup_curve.png)")
    args = parser.parse_args()

    run_benchmark(
        repo_path=args.repo,
        max_trials=args.trials,
        cache_size_str=args.cache_size,
        output_png=args.out
    )


if __name__ == "__main__":
    main()