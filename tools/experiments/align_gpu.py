"""Where alignment time goes, and what a GPU would move.

Reproduces ARCHITECTURE.md's GPU discussion.

Alignment wall-clock is graded at 8% -- more than the residual ratio (7%) --
and the PS budgets 8 GB VRAM in its constraints, so GPU use is anticipated.
The inner loop is C = W_target @ W_base.T, a matmul.

The result that matters is the *second* one: once matmuls move to the GPU, the
LAP solve (scipy, Hungarian, O(n^3), CPU-only) becomes the bottleneck by two
orders of magnitude. Optimising the matmuls alone buys little.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--groups", type=int, default=100, help="solvable groups in a 7B model")
    ap.add_argument("--sweeps", type=int, default=4)
    args = ap.parse_args()

    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        torch, gpu = None, False
    print(f"device: {torch.cuda.get_device_name(0) if gpu else 'CPU only (no CUDA)'}\n")

    print("Cost matrix  C = W_t @ W_b.T  (Algorithm 1's inner loop)")
    print(f"  {'group':>7}{'CPU matmul':>13}{'GPU matmul':>13}{'speedup':>9}{'LAP solve':>13}")
    last = None
    for n in args.sizes:
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, n).astype(np.float32)
        t0 = time.perf_counter(); C = A @ B.T; cpu = time.perf_counter() - t0
        if gpu:
            At = torch.from_numpy(A).cuda(); Bt = torch.from_numpy(B).cuda()
            torch.cuda.synchronize()
            for _ in range(3):
                _ = At @ Bt.T
            torch.cuda.synchronize()
            t0 = time.perf_counter(); _ = At @ Bt.T; torch.cuda.synchronize()
            g = time.perf_counter() - t0
        else:
            g = float("nan")
        t0 = time.perf_counter(); linear_sum_assignment(C, maximize=True)
        lap = time.perf_counter() - t0
        last = (cpu, g, lap)
        print(f"  {n:>7}{cpu*1000:>11.1f}ms{g*1000:>11.2f}ms"
              f"{(cpu/g if gpu else float('nan')):>8.0f}x{lap*1000:>11.0f}ms")

    cpu, g, lap = last
    N = args.groups * args.sweeps
    print(f"\n7B-class model: ~{args.groups} groups x {args.sweeps} sweeps = {N} builds")
    print(f"  matmuls on CPU                {N*cpu:>9.0f} s")
    if gpu:
        print(f"  matmuls on GPU                {N*g:>9.1f} s")
    print(f"  LAP solves (CPU, no GPU path) {N*lap:>9.0f} s   <- the real bottleneck")
    print("\n  Attack the assignment step (auction / Sinkhorn / sweep cap),")
    print("  not the matmuls.")


if __name__ == "__main__":
    main()
