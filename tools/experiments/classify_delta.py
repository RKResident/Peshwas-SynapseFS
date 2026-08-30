"""Exponent-aware classification of a residual, against magnitude-based.

The proposal: split each residual into a class, keep the small ones in a
narrow payload stream, push the rest into separate high/low byte planes, and
let zstd do the compressing. That last part is the shipping codec's premise
too, so what is actually new is the CLASSIFIER -- three exponent-aware classes
(same exponent, exponent +1, exponent -1) instead of one magnitude test.

The question is therefore narrow and answerable: does knowing the exponent
relationship capture more elements in the narrow stream than simply asking
whether the residual is small?

Two things are held constant so the comparison is about the classifier and
nothing else. Every scheme separates its streams the same way, and every
stream is zstd level 1. The residual operator is varied deliberately, because
XOR and subtraction are not interchangeable here: an XOR's magnitude is the
position of the highest differing bit rather than a distance, so it jumps to
the next power of two at every carry -- measured median 63 against
subtraction's 26 on bf16.
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL, shuffle
from synapsefs.safetensors_io import SafetensorsFile

C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)
GEOM = {"F16": (5, 10), "BF16": (8, 7)}      # exponent bits, mantissa bits


def z(b: bytes) -> int:
    return len(C.compress(b))


def planes(v16: np.ndarray) -> tuple[bytes, bytes]:
    by = v16.view(np.uint8).reshape(-1, 2)
    return (np.ascontiguousarray(by[:, 0]).tobytes(),
            np.ascontiguousarray(by[:, 1]).tobytes())


def zigzag(delta_bits: np.ndarray) -> np.ndarray:
    v = delta_bits.view(np.int16).astype(np.int32)
    return ((v << 1) ^ (v >> 31)).astype(np.uint16)


def classify_exponent(t: np.ndarray, b: np.ndarray, dtype: str, k: int = 7):
    """The proposed classifier. Returns (class_id, payload) per element.

    class 0 = same sign and exponent, mantissa XOR fits in k bits
    class 1 = same sign, exponent +1, mantissa relationship fits in k-1 bits
    class 2 = same sign, exponent -1, likewise
    class 3 = everything else, stored as a full XOR
    """
    ebits, mbits = GEOM[dtype]
    mmask = (1 << mbits) - 1
    st, sb = t >> 15, b >> 15
    et = (t >> mbits) & ((1 << ebits) - 1)
    eb = (b >> mbits) & ((1 << ebits) - 1)
    mt, mb = t & mmask, b & mmask

    same_sign = st == sb
    de = et.astype(np.int32) - eb.astype(np.int32)
    mx = (mt ^ mb)

    cls = np.full(t.size, 3, dtype=np.uint8)
    payload = np.zeros(t.size, dtype=np.uint16)

    c0 = same_sign & (de == 0) & (mx < (1 << k))
    cls[c0] = 0
    payload[c0] = mx[c0]

    # An exponent step doubles or halves the value, so the mantissa of the
    # larger side is the smaller side's shifted by one. What has to be small
    # is the residual AFTER undoing that shift, not the raw XOR -- the raw XOR
    # always carries the exponent bit and is never small.
    lim = 1 << max(0, k - 1)
    up = same_sign & (de == 1) & (cls == 3)
    r_up = np.abs(mt.astype(np.int32) - ((mb.astype(np.int32) + (1 << mbits)) >> 1))
    ok = up & (r_up < lim)
    cls[ok] = 1
    payload[ok] = r_up[ok]

    dn = same_sign & (de == -1) & (cls == 3)
    r_dn = np.abs(((mt.astype(np.int32) + (1 << mbits)) >> 1) - mb.astype(np.int32))
    ok = dn & (r_dn < lim)
    cls[ok] = 2
    payload[ok] = r_dn[ok]
    return cls, payload


def size_exponent_scheme(t, b, dtype, k=7) -> tuple[int, dict]:
    cls, payload = classify_exponent(t, b, dtype, k)
    small = cls != 3
    xor = (t ^ b).astype(np.uint16)
    lo, hi = planes(np.ascontiguousarray(xor[~small]))
    total = (z(np.packbits(small).tobytes())            # smalldiff bitmap
             + z(cls[small].tobytes())                  # which of the 3 classes
             + z(payload[small].astype(np.uint8).tobytes())
             + z(lo) + z(hi))
    frac = {f"class {i}": float((cls == i).mean()) for i in range(4)}
    return total, frac


def size_magnitude_scheme(v16: np.ndarray) -> int:
    """What ships today: zigzag residual, escape byte, wide plane split out."""
    small = v16 < 255
    narrow = np.where(small, v16, 255).astype(np.uint8)
    wide = shuffle(np.ascontiguousarray(v16[~small]).astype("<u2").tobytes(), 2)
    return z(narrow.tobytes()) + z(wide)


def load(root: str, ep: int, dtype: str):
    T = []
    with SafetensorsFile(f"{root}/epoch{ep:02d}.safetensors") as f:
        for n in f.names():
            if f.spec(n).dtype == dtype:
                T.append(f.rows(n, 0, f.spec(n).num_rows).ravel().copy())
    return np.concatenate(T)


def midpoint(a, b, dtype):
    if dtype == "BF16":
        w = lambda v: (v.astype(np.uint32) << 16).view(np.float32)
        u = np.ascontiguousarray((w(a) + w(b)) / 2.0, dtype=np.float32).view(np.uint32)
        return ((u + 0x8000 + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    m = (a.view(np.float16).astype(np.float32) + b.view(np.float16).astype(np.float32)) / 2
    return np.ascontiguousarray(m, dtype=np.float16).view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--dtype", required=True, choices=["F16", "BF16"])
    ap.add_argument("--target", type=int, default=24)
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()

    t = load(args.checkpoints, args.target, args.dtype)
    prev = load(args.checkpoints, args.target - 1, args.dtype)
    nxt = load(args.checkpoints, args.target + 1, args.dtype)
    raw = t.nbytes
    print(f"{args.checkpoints}  {args.dtype}  epoch {args.target}, "
          f"{raw/2**20:.1f} MiB, k={args.k}\n")

    for pred_name, base in (("P-frame (previous epoch)", prev),
                            ("B-frame (mean of neighbours)", midpoint(prev, nxt, args.dtype))):
        sub = (t - base).astype(np.uint16)
        xor = (t ^ base).astype(np.uint16)
        exp_total, frac = size_exponent_scheme(t, base, args.dtype, args.k)
        print(f"  {pred_name}")
        print(f"    {'scheme':<44} {'ratio':>8}")
        print(f"    {'byte shuffle on subtract (old codec)':<44} "
              f"{z(shuffle(sub.tobytes(), 2))/raw*100:7.2f}%")
        print(f"    {'exponent-aware classify on XOR (proposed)':<44} "
              f"{exp_total/raw*100:7.2f}%")
        print(f"    {'magnitude classify on XOR':<44} "
              f"{size_magnitude_scheme(xor)/raw*100:7.2f}%")
        print(f"    {'magnitude classify on zigzag(sub)  (ships)':<44} "
              f"{size_magnitude_scheme(zigzag(sub))/raw*100:7.2f}%")
        share = "  ".join(f"{k}={v*100:.1f}%" for k, v in frac.items())
        print(f"    classifier hit rates: {share}")
        print(f"    magnitude hit rate  : "
              f"{(zigzag(sub) < 255).mean()*100:.1f}% fit in a byte\n")


if __name__ == "__main__":
    main()
