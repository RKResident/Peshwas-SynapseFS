"""Why byte alignment is the whole game: the k=8 cliff.

Reproduces ARCHITECTURE.md 4.1.1.

Splitting the 16-bit residual into a low-k / high-(16-k) pair, swept over k.
Only k=8 keeps both planes byte-aligned, and it is ~10 points better than any
neighbour -- not because 8 bits is the natural field boundary (it is not; fp16
splits 1/5/10) but because every other k packs sub-byte fields, smearing each
element across byte boundaries and destroying the LZ matches.

Also measures the two ways of escaping that, both of which lose:
  - zero-padding each field to a whole byte  (aligned, but 2x the data)
  - naive bitpacking at per-block width      (one outlier taxes its block)
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL, to_monotone_key, dtype_spec, zigzag
from tools.experiments._common import add_common_args, header, load_pair


def bit_columns(d):
    return np.unpackbits(d.view(np.uint8).reshape(-1, 2), axis=1, bitorder="little")


def pack_cols(cols) -> bytes:
    return np.packbits(np.ascontiguousarray(cols).ravel(), bitorder="little").tobytes()


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    args = ap.parse_args()
    c = zstd.ZstdCompressor(level=args.level)

    t, b, raw = load_pair(args.checkpoints, args.newest - args.gap, args.newest)
    d = (t - b).astype(np.uint16)
    n = len(d)
    header(f"gap {args.gap}, {raw/1048576:.2f} MiB, {n:,} elements, zstd L{args.level}")

    bits = bit_columns(d)
    print("  split-point sweep -- low k bits in one frame, high 16-k in another:")
    for k in range(5, 13):
        size = sum(len(c.compress(pack_cols(bits[:, s])))
                   for s in (slice(0, k), slice(k, 16)))
        note = ("  <-- BYTE boundary" if k == 8 else
                "  <-- fp16 mantissa boundary" if k == 10 else "")
        print(f"    k={k:<3}{size:>12,}{size/raw*100:>9.2f}%{note}")

    print("\n  zero-padded field planes (byte-aligned, but expands the data):")
    variants = {
        "2 byte planes (the k=8 split)":
            lambda: (d.view(np.uint8).reshape(-1, 2)[:, 0].tobytes(),
                     (d >> 8).astype(np.uint8).tobytes()),
        "3 padded: sign+exp | mant-hi | mant-lo":
            lambda: (((d >> 10) & 0x3F).astype(np.uint8).tobytes(),
                     ((d >> 8) & 0x03).astype(np.uint8).tobytes(),
                     (d & 0xFF).astype(np.uint8).tobytes()),
        "4 padded: sign | exp | mant-hi | mant-lo":
            lambda: ((d >> 15).astype(np.uint8).tobytes(),
                     ((d >> 10) & 0x1F).astype(np.uint8).tobytes(),
                     ((d >> 8) & 0x03).astype(np.uint8).tobytes(),
                     (d & 0xFF).astype(np.uint8).tobytes()),
    }
    for label, fn in variants.items():
        parts = fn()
        pre = sum(len(p) for p in parts)
        size = sum(len(c.compress(p)) for p in parts)
        print(f"    {label:<40}{pre/raw:>5.1f}x{size:>12,}{size/raw*100:>9.2f}%")

    print("\n  why splitting fields loses despite collapsing the alphabet:")
    hi = (d >> 8).astype(np.uint8)
    for label, arr in (("whole high byte", hi), ("sign", hi >> 7),
                       ("exp", (hi >> 2) & 0x1F), ("mant-hi", hi & 0x03)):
        print(f"    {label:<18}{len(np.unique(arr)):>4} distinct values"
              f"{len(c.compress(np.ascontiguousarray(arr).tobytes())):>12,} bytes alone")

    print("\n  naive bitpacking at per-block width (values are heavy-tailed):")
    zz = zigzag(d)
    print(f"    median {int(np.median(zz))}, p99 {int(np.percentile(zz,99))}, "
          f"max {int(zz.max())}  -> one outlier taxes its whole block")
    for B in (16, 32, 128):
        nb = -(-n // B)
        pad = np.zeros(nb * B, np.uint16); pad[:n] = zz
        k = np.maximum(1, np.ceil(np.log2(pad.reshape(nb, B).max(axis=1)
                                          .astype(np.float64) + 1)).astype(int))
        total = ((k.astype(np.int64) * B).sum() + nb * 8) / 8
        print(f"    block {B:<5} mean k {k.mean():>5.2f}  {total/raw*100:>7.1f}% of raw"
              f"  (before any entropy coding)")


if __name__ == "__main__":
    main()
