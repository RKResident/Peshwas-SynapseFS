"""How much lossless headroom is actually left, and where.

Reproduces the headroom decomposition in `old_docs/ARCHITECTURE.md` §4.1.3.

The byte shuffle splits the residual into two planes of exactly equal size, and
they could not be more different: the high plane (sign + exponent + top
mantissa bits) carries the drift structure, the low plane carries mantissa
noise. Measuring them separately is what turns "we are near the entropy floor"
into a number -- it says which half of the stream any future idea has to
attack, and how many points are on the table there.

Order-1 conditional entropy is included because it is the cheap test for
whether a context model could beat zstd. If H1 ~= H0 the plane's bytes are
conditionally independent and no amount of modelling helps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL
from synapsefs.safetensors_io import SafetensorsFile

DEFAULT_CHECKPOINTS = Path("tools/tools/benchmark")


def planes(delta: np.ndarray):
    """The two byte planes the shuffle produces, as contiguous arrays."""
    b = delta.view(np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(b[:, 0]), np.ascontiguousarray(b[:, 1])


def h0(a: np.ndarray) -> float:
    p = np.bincount(a, minlength=256).astype(np.float64)
    p = p[p > 0] / a.size
    return float(-(p * np.log2(p)).sum())


def h1(a: np.ndarray) -> float:
    """Entropy of each byte given its predecessor."""
    joint = (a[:-1].astype(np.int32) << 8) | a[1:]
    cnt = np.bincount(joint, minlength=65536).reshape(256, 256).astype(np.float64)
    row = cnt.sum(1)
    nz = row > 0
    cond = cnt[nz] / row[nz, None]
    logs = np.where(cond > 0, np.log2(np.where(cond > 0, cond, 1)), 0.0)
    return float(-(cnt[nz] * logs).sum() / cnt.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--tensor", default="features.31.weight")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--anchor", action="store_true",
                    help="decompose the RAW tensor (what a FULL commit stores) "
                         "instead of the residual")
    args = ap.parse_args()

    fmt = str(args.checkpoints / "epoch{:02d}.safetensors")
    with SafetensorsFile(fmt.format(args.base)) as b, \
         SafetensorsFile(fmt.format(args.target)) as t:
        spec = t.spec(args.tensor)
        tb = t.rows(args.tensor, 0, spec.num_rows).ravel().copy()
        bb = b.rows(args.tensor, 0, spec.num_rows).ravel().copy()

    # A FULL commit stores the weights themselves, not a residual, and it is
    # ~4x the bytes of a delta. Whether its low plane is equally incompressible
    # decides whether anchors have any headroom at all -- the delta measurement
    # says nothing about them, because the two streams are different data.
    delta = tb.astype(np.uint16) if args.anchor else (tb - bb).astype(np.uint16)
    raw = delta.nbytes
    lo, hi = planes(delta)
    C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)

    what = (f"RAW weights, epoch {args.target} (anchor)" if args.anchor
            else f"residual, epoch {args.base}->{args.target}")
    print(f"{args.tensor}  {raw/2**20:.1f} MiB  {what}")
    print()
    print("PLANE DECOMPOSITION")
    print(f"  {'plane':<16} {'MiB':>7} {'zstd':>8} {'H0':>8} {'H1':>8}")
    for name, pl in (("high (sgn/exp)", hi), ("low (mantissa)", lo)):
        r = len(C.compress(pl.tobytes())) / pl.nbytes * 100
        print(f"  {name:<16} {pl.nbytes/2**20:7.2f} {r:7.2f}% "
              f"{h0(pl)/8*100:7.2f}% {h1(pl)/8*100:7.2f}%")
    joined = len(C.compress(np.concatenate([hi, lo]).tobytes())) / raw * 100
    print(f"  {'COMBINED':<16} {raw/2**20:7.2f} {joined:7.2f}%")
    print(f"\n  ceiling with a perfect order-1 coder on the high plane: "
          f"{(0.5 * h1(hi) / 8 + 0.5) * 100:.2f}%")
    print("  (the low plane is incompressible, so 50.00% is the hard floor)")

    print()
    print("RAW LOW PLANE  -- same ratio, less work")
    print(f"  {'scheme':<28} {'ratio':>8} {'enc s':>7} {'dec s':>7}")
    D = zstd.ZstdDecompressor()
    for label, payload, overhead in (
        ("single frame (current)", np.concatenate([hi, lo]).tobytes(), 0),
        ("raw low + compressed high", hi.tobytes(), lo.nbytes),
    ):
        t0 = time.perf_counter(); blob = C.compress(payload); enc = time.perf_counter() - t0
        t0 = time.perf_counter(); D.decompress(blob, max_output_size=raw); dec = time.perf_counter() - t0
        print(f"  {label:<28} {(len(blob)+overhead)/raw*100:7.2f}% {enc:7.2f} {dec:7.2f}")

    print()
    print("LOSSY: round the target's mantissa, then delta as usual")
    print(f"  {'kept mantissa bits':<20} {'ratio':>8} {'mean rel err':>13}")
    truth = tb.view(np.float16).astype(np.float64)
    nonzero = np.abs(truth) > 0
    for drop in (0, 2, 3, 4, 5):
        half = 1 << (drop - 1) if drop else 0
        mask = 0xFFFF ^ ((1 << drop) - 1)
        q = ((tb.astype(np.uint32) + half) & mask).astype(np.uint16)
        qb = ((bb.astype(np.uint32) + half) & mask).astype(np.uint16)
        qlo, qhi = planes((q - qb).astype(np.uint16))
        r = len(C.compress(np.concatenate([qhi, qlo]).tobytes())) / raw * 100
        got = q.view(np.float16).astype(np.float64)
        rel = np.abs(got[nonzero] - truth[nonzero]) / np.abs(truth[nonzero])
        print(f"  {10-drop:>2} of 10          {r:7.2f}% {rel.mean()*100:12.4f}%")


if __name__ == "__main__":
    main()
