"""Can a permutation make one layer resemble a different layer of the same model?

The question behind MCWC (arXiv 2605.24754), which aligns permutation-symmetric
blocks ACROSS depth and codes layer L+1 as a prediction from layer L.

The framing deserves care, because it predicts the answer. Permutation symmetry
is a symmetry of the network's FUNCTION: relabel layer L's output units,
relabel layer L+1's input columns, and the network computes the same thing. It
relates the network to ITSELF. No group action carries layer L's weight matrix
to layer M's -- they are distinct parameters computing distinct functions. So
aligning L to M is not recovering a symmetry, it is FITTING the permutation
that makes M most resemble L, which is what "motion compensation" means: a
video codec never claims frame t+1 is frame t translated, it finds the best
predictor and codes the error.

That distinction is survivable for MCWC and fatal for us. MCWC is lossy --
quantized residuals under a rate-distortion objective, benchmarked against
GPTQ and AWQ -- so any shrinkage of the residual is a bitrate win. We are
lossless, where break-even is 1.0: a residual at 1.3x the tensor is worse than
storing it raw.

Neither benchmark has two layers of the same shape (both are widening
pyramids), so pairs are built by slicing the wider layer's output channels.
That is favourable to the hypothesis, not hostile: it lets a layer choose
which of its filters to match against.

Controls, as always:
  self-permuted   a known permutation of the SAME tensor. Must reach ~0.
  no-perm         the pair with no alignment at all. The starting point.
  epoch delta     the same tensor against its own previous checkpoint --
                  what the shipping codec already achieves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zstandard as zstd
from scipy.optimize import linear_sum_assignment

from synapsefs.codec.chunk import DEFAULT_LEVEL
from synapsefs.safetensors_io import SafetensorsFile

C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)


def load(path: str, name: str) -> np.ndarray:
    with SafetensorsFile(path) as f:
        s = f.spec(name)
        bits = f.rows(name, 0, s.num_rows).ravel()
        return bits.view(np.float16).reshape(s.shape).astype(np.float32)


def flat(a: np.ndarray) -> np.ndarray:
    return a.reshape(a.shape[0], -1)


def resid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).astype(np.float64)) /
                 np.linalg.norm(a.astype(np.float64)))


def align(target: np.ndarray, base: np.ndarray, sweeps: int, block: int):
    """Coordinate descent over a row permutation and a column-block permutation.

    Exactly the two axes a layer has. Rows are output units; columns move in
    blocks of `block` because a conv kernel's spatial taps travel with their
    input channel.
    """
    T, B = flat(target), flat(base)
    rows, cols = T.shape
    nblk = cols // block
    p_row = p_col = None
    for _ in range(sweeps):
        Bc = B if p_col is None else B.reshape(rows, nblk, block)[:, p_col, :].reshape(rows, cols)
        p_row = linear_sum_assignment(T @ Bc.T, maximize=True)[1]
        Br = B[p_row]
        Tb = T.reshape(rows, nblk, block).transpose(1, 0, 2).reshape(nblk, -1)
        Bb = Br.reshape(rows, nblk, block).transpose(1, 0, 2).reshape(nblk, -1)
        p_col = linear_sum_assignment(Tb @ Bb.T, maximize=True)[1]
    out = B[p_row].reshape(rows, nblk, block)[:, p_col, :].reshape(rows, cols)
    return out.reshape(target.shape)


def ratio(target: np.ndarray, base: np.ndarray | None) -> float:
    """What the shipping codec would store for this pair."""
    t = target.astype(np.float16).view(np.uint16).ravel()
    if base is None:
        d = t
    else:
        d = (t - base.astype(np.float16).view(np.uint16).ravel()).astype(np.uint16)
    by = d.view(np.uint8).reshape(-1, 2)
    st = np.concatenate([np.ascontiguousarray(by[:, 0]),
                         np.ascontiguousarray(by[:, 1])])
    return len(C.compress(st.tobytes())) / d.nbytes * 100


def run(label: str, target: np.ndarray, base: np.ndarray, sweeps: int,
        block: int) -> None:
    pre = resid(target, base)
    aligned = align(target, base, sweeps, block)
    post = resid(target, aligned)
    print(f"  {label:<34} {pre:8.4f} {post:8.4f} {ratio(target, aligned):9.2f}% "
          f"{ratio(target, None):9.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=3)
    args = ap.parse_args()

    cnn = "tools/tools/benchmark/epoch25.safetensors"
    cnn_prev = "tools/tools/benchmark/epoch24.safetensors"
    mlp = "tools/tools/benchmark-mlp/epoch20.safetensors"
    mlp_prev = "tools/tools/benchmark-mlp/epoch19.safetensors"

    print("pre/post are relative residuals; break-even for storing a delta is "
          "1.0.\n'delta%' is what our codec would store; 'raw%' is that tensor "
          "stored alone.\n")
    print(f"  {'pair':<34} {'pre':>8} {'post':>8} {'delta%':>9} {'raw%':>9}")

    a = load(cnn, "features.24.weight")                       # [1344,1344,3,3]
    b = load(cnn, "features.28.weight")[:1344]                # [1792,...] sliced
    ctrl = a.copy()[np.random.default_rng(0).permutation(1344)]
    run("CONTROL self-permuted (cnn 24)", a, ctrl, args.sweeps, 9)
    run("cnn: features.24 <- features.28", a, b, args.sweeps, 9)

    a2 = load(cnn, "features.17.weight")                      # [896,896,3,3]
    b2 = load(cnn, "features.21.weight")[:896]                # [1344,896,3,3]
    run("cnn: features.17 <- features.21", a2, b2, args.sweeps, 9)

    m = load(mlp, "features.6.weight")                        # [2048,4096]
    mb = load(mlp, "features.3.weight")[:2048]                # [4096,4096]
    run("mlp: features.6 <- features.3", m, mb, args.sweeps, 1)

    print()
    print("  for scale, the SAME tensors against their own previous epoch:")
    for tag, cur, prev, nm in (("cnn features.24", cnn, cnn_prev, "features.24.weight"),
                               ("mlp features.6", mlp, mlp_prev, "features.6.weight")):
        t, p = load(cur, nm), load(prev, nm)
        print(f"  {tag + ' <- epoch t-1':<34} {resid(t, p):8.4f} "
              f"{resid(t, p):8.4f} {ratio(t, p):9.2f}% {ratio(t, None):9.2f}%")


if __name__ == "__main__":
    main()
