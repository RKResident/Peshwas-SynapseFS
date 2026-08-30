"""Does weight matching optimise the thing the codec actually stores?

Git Re-Basin maximises the dot product between a target layer and a permuted
base layer. That is not an arbitrary similarity score -- for a permutation P,

    ||W_t - P W_b||^2 = ||W_t||^2 - 2<W_t, P W_b> + ||P W_b||^2

and P only reorders rows, so ||P W_b|| = ||W_b|| and both norm terms are
constants. Maximising the dot product IS minimising squared error. The
objective is exactly L2-optimal.

The question is whether L2 is the right target. The codec stores a residual
whose cost in bits scales roughly as `numel * log2(typical |delta|)`, and:

  L2  squares the errors, so it is dominated by the few worst elements;
  L1  tracks the bulk of the distribution, which is nearer to what bits cost;
  bits is the thing itself -- sum of log2(1 + |delta|) in ULP space.

All three are linear assignment problems; they differ only in how the cost
matrix is built. L2's cost is a matmul, which is why it is fast. L1 and bits
need a pairwise reduction with no BLAS equivalent, which is why they are not.

The controls matter. On a clean permuted pair every objective recovers the
true permutation and the comparison says nothing, so the interesting case is a
permuted pair that has ALSO drifted -- where the objectives can disagree.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zstandard as zstd
from scipy.optimize import linear_sum_assignment

from synapsefs.codec.chunk import DEFAULT_LEVEL
from synapsefs.safetensors_io import SafetensorsFile

C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)


def load_bits(path: str, name: str) -> np.ndarray:
    with SafetensorsFile(path) as f:
        s = f.spec(name)
        return f.rows(name, 0, s.num_rows).reshape(s.num_rows, -1).copy()


def stored_bytes(t_bits: np.ndarray, b_bits: np.ndarray) -> int:
    """What the shipping codec would store for this residual."""
    d = (t_bits.ravel() - b_bits.ravel()).astype(np.uint16)
    by = d.view(np.uint8).reshape(-1, 2)
    stream = np.concatenate([np.ascontiguousarray(by[:, 0]),
                             np.ascontiguousarray(by[:, 1])])
    return len(C.compress(stream.tobytes()))


def cost_dot(T, B, torch, dev):
    """<t_i, b_j>. Maximised. One matmul -- the current objective."""
    t = torch.from_numpy(T).to(dev).float()
    b = torch.from_numpy(B).to(dev).float()
    return (t @ b.T).cpu().numpy(), True


def cost_l1(T, B, torch, dev):
    """||t_i - b_j||_1. Minimised. Chunked: the full 3-D difference will not fit."""
    t = torch.from_numpy(T).to(dev).float()
    b = torch.from_numpy(B).to(dev).float()
    n = t.shape[0]
    out = torch.empty((n, n), device=dev)
    step = max(1, 2 ** 26 // (b.shape[0] * b.shape[1]))
    for i in range(0, n, step):
        out[i:i + step] = (t[i:i + step, None, :] - b[None, :, :]).abs().sum(-1)
    return out.cpu().numpy(), False


def cost_bits(T, B, torch, dev, ulp_t, ulp_b):
    """sum_k log2(1 + |ULP delta|). Minimised. The codec's own cost, near enough.

    Distances are taken on the raw bit patterns with wraparound, because that
    is what the codec subtracts: a residual is `target - base` mod 2**16.
    """
    t = torch.from_numpy(ulp_t.astype(np.float32)).to(dev)
    b = torch.from_numpy(ulp_b.astype(np.float32)).to(dev)
    n = t.shape[0]
    out = torch.empty((n, n), device=dev)
    step = max(1, 2 ** 25 // (b.shape[0] * b.shape[1]))
    for i in range(0, n, step):
        d = (t[i:i + step, None, :] - b[None, :, :]).abs()
        d = torch.minimum(d, 65536.0 - d)          # wraparound
        out[i:i + step] = torch.log2(1.0 + d).sum(-1)
    return out.cpu().numpy(), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--cols", type=int, default=2048,
                    help="columns subsampled for the L1 and bits costs; the "
                         "full width is quadratic in memory and they are "
                         "estimating a per-row distance, not a full residual")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Tb = load_bits(args.target, args.tensor)
    Bb0 = load_bits(args.base, args.tensor)
    n = Tb.shape[0]
    rng = np.random.default_rng(args.seed)
    pi = rng.permutation(n)
    Bb = Bb0[pi]                       # the base the solver is handed
    truth = np.empty(n, dtype=np.int64)
    truth[pi] = np.arange(n)           # p[i] is the BASE index for TARGET i

    Tf = Tb.view(np.float16).astype(np.float32)
    Bf = Bb.view(np.float16).astype(np.float32)
    cols = rng.choice(Tb.shape[1], min(args.cols, Tb.shape[1]), replace=False)

    print(f"{args.tensor}  {n} rows x {Tb.shape[1]} cols   device {dev}")
    print(f"base = {args.base} permuted;  target = {args.target}\n")
    raw = Tb.nbytes
    print(f"  {'objective':<26} {'recovered':>10} {'stored':>12} {'ratio':>8} {'build':>8}")

    for label, fn, args_ in (
        ("identity (no alignment)", None, None),
        ("dot product  (= L2, now)", cost_dot, (Tf, Bf)),
        ("L1 / MAE", cost_l1, (Tf[:, cols], Bf[:, cols])),
        ("bits: sum log2(1+|dULP|)", cost_bits, (Tf[:, cols], Bf[:, cols])),
    ):
        if fn is None:
            p, acc, dt = np.arange(n), (np.arange(n) == truth).mean(), 0.0
        else:
            t0 = time.perf_counter()
            if fn is cost_bits:
                Cm, maximise = fn(None, None, torch, dev,
                                  Tb[:, cols], Bb[:, cols])
            else:
                Cm, maximise = fn(args_[0], args_[1], torch, dev)
            dt = time.perf_counter() - t0
            p = linear_sum_assignment(Cm, maximize=maximise)[1]
            acc = (p == truth).mean()
        sz = stored_bytes(Tb, Bb[p])
        print(f"  {label:<26} {acc*100:9.1f}% {sz:12,} {sz/raw*100:7.2f}% {dt:7.2f}s")

    sz = stored_bytes(Tb, Bb[truth])
    print(f"  {'TRUE permutation (oracle)':<26} {100.0:9.1f}% {sz:12,} "
          f"{sz/raw*100:7.2f}%")


if __name__ == "__main__":
    main()
