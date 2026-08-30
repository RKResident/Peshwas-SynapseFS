"""Bidirectional prediction between anchors, against the star and the chain.

The star stores every Nth checkpoint whole and diffs the rest against the
nearest anchor. In video terms those are I-frames and P-frames, and the
arrangement is missing the third kind: a B-frame, predicted from a frame on
each side rather than from one behind.

This is NOT the second-order delta that was measured and rejected at +3.83pp.
That one extrapolated forward -- `x[k] - 2x[k-1] + x[k-2]` -- compounding two
steps of noise in the same direction. A centred estimate averages two
independent errors instead of stacking them, which halves the variance rather
than doubling it. Same ingredients, opposite sign.

Three layouts over one rebase interval, anchors at both ends:

  star     2<-1  3<-1  4<-1                  gaps 1, 2, 3 from the anchor
  chain    2<-1  3<-2  4<-3                  gap 1 always, but depth 3
  dyadic   3<-mean(1,5)  2<-mean(1,3)  4<-mean(3,5)

The dyadic order is what HEVC calls hierarchical-B: bisect the interval, then
bisect each half. Every frame is predicted from two already-stored frames at
most two levels deep.

Reconstruction is the cost. A B-frame needs both neighbours, so decoding is a
small tree rather than a walk, and a chunk-level read has to materialise more
than it does today. Whether that is acceptable is a separate question from
whether the bytes are smaller, which is what this measures.
"""

from __future__ import annotations

import argparse

import numpy as np

from synapsefs.codec.chunk import encode_chunk
from synapsefs.safetensors_io import SafetensorsFile


def widen(bits: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(np.float16).astype(np.float32)


def narrow(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u + 0x8000 + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return np.ascontiguousarray(x, dtype=np.float16).view(np.uint16)


def midpoint(a: np.ndarray, b: np.ndarray, dtype: str) -> np.ndarray:
    """The prediction. Rounded back to the stored width, because the decoder
    only ever has the stored values to work from."""
    return narrow((widen(a, dtype) + widen(b, dtype)) / 2.0, dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--dtype", required=True, choices=["F16", "BF16"])
    ap.add_argument("--anchor", type=int, default=21,
                    help="first anchor; the next is anchor + interval")
    ap.add_argument("--interval", type=int, default=4)
    args = ap.parse_args()

    lo, hi = args.anchor, args.anchor + args.interval
    mid = lo + args.interval // 2
    eps = list(range(lo, hi + 1))

    frames = {}
    names = []
    raw = 0
    for e in eps:
        with SafetensorsFile(f"{args.checkpoints}/epoch{e:02d}.safetensors") as f:
            cur = {}
            for n in f.names():
                s = f.spec(n)
                if s.dtype != args.dtype:
                    continue
                cur[n] = f.rows(n, 0, s.num_rows).ravel().copy()
                if e == lo:
                    names.append(n)
                    raw += cur[n].nbytes
            frames[e] = cur

    def enc(target_ep, base_arrays) -> int:
        return sum(encode_chunk(frames[target_ep][n], base_arrays[n],
                                dtype=args.dtype, allow_raw_fallback=False).stored_len
                   for n in names)

    star = {e: enc(e, frames[lo]) for e in (lo + 1, mid, hi - 1)}
    chain = {}
    for e in (lo + 1, mid, hi - 1):
        chain[e] = enc(e, frames[e - 1])
    mid_pred = {n: midpoint(frames[lo][n], frames[hi][n], args.dtype) for n in names}
    lo_half = {n: midpoint(frames[lo][n], frames[mid][n], args.dtype) for n in names}
    hi_half = {n: midpoint(frames[mid][n], frames[hi][n], args.dtype) for n in names}
    dyadic = {mid: enc(mid, mid_pred),
              lo + 1: enc(lo + 1, lo_half),
              hi - 1: enc(hi - 1, hi_half)}

    print(f"{args.checkpoints}  {args.dtype}   anchors {lo} and {hi}, "
          f"{raw/2**20:.1f} MiB per checkpoint\n")
    print(f"  {'epoch':>6} {'star (from anchor)':>20} {'chain (from prev)':>19} "
          f"{'dyadic B':>12}")
    for e in (lo + 1, mid, hi - 1):
        print(f"  {e:>6} {star[e]/raw*100:19.2f}% {chain[e]/raw*100:18.2f}% "
              f"{dyadic[e]/raw*100:11.2f}%")
    ts, tc, td = sum(star.values()), sum(chain.values()), sum(dyadic.values())
    tot = raw * 3
    print(f"  {'TOTAL':>6} {ts/tot*100:19.2f}% {tc/tot*100:18.2f}% {td/tot*100:11.2f}%")
    print(f"\n  dyadic vs star  {(td-ts)/tot*100:+.2f}pp"
          f"    dyadic vs chain {(td-tc)/tot*100:+.2f}pp")
    print(f"  decode depth:   star 1, chain 3, dyadic 2 (and needs 2 parents)")


if __name__ == "__main__":
    main()
