"""Why bit-level transforms fail: the sign runs are too short.

Reproduces ARCHITECTURE.md 4.1.1.

Byte planes store one element per byte, so a run of N same-sign weights is N
identical bytes and zstd's matcher removes them. Bit-packing puts 8 elements in
a byte, so any run shorter than 8 never fills one and consecutive packed bytes
are near-random combinations of unrelated signs.
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from tools.experiments._common import add_common_args, header, load_pair


def entropy(buf: bytes) -> float:
    a = np.frombuffer(buf, np.uint8)
    counts = np.bincount(a, minlength=256)
    p = counts[counts > 0] / len(a)
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    c1 = zstd.ZstdCompressor(level=1)
    t, b, raw = load_pair(args.checkpoints, args.newest - 1, args.newest)
    d = (t - b).astype(np.uint16)
    n = len(d)
    header(f"{n:,} elements ({raw/1048576:.2f} MiB), epoch "
           f"{args.newest-1} -> {args.newest}")

    sign = (d >> 15).astype(np.uint8)
    breaks = np.diff(sign.astype(np.int8)) != 0
    lengths = np.diff(np.flatnonzero(np.r_[True, breaks, True]))
    print(f"  mean run length of the sign bit : {n / (int(breaks.sum()) + 1):.2f} elements")
    print(f"  fraction of runs >= 8 elements  : {float((lengths >= 8).mean())*100:.1f}%")
    print("  -> bit-packing needs runs of 8 to produce a repeated byte\n")

    byte_plane = (d >> 8).astype(np.uint8).tobytes()
    bit_plane = np.packbits(sign).tobytes()
    print(f"  {'representation':<34}{'raw':>11}{'zstd':>11}{'entropy/byte':>14}")
    for label, buf in (("high BYTE plane (1 byte/elem)", byte_plane),
                       ("sign BIT plane  (8 elem/byte)", bit_plane)):
        print(f"  {label:<34}{len(buf):>11,}{len(c1.compress(buf)):>11,}"
              f"{entropy(buf):>13.2f}b")
    print(f"\n  bit-packed sign compresses to "
          f"{len(c1.compress(bit_plane))/len(bit_plane)*100:.0f}% of raw -- essentially not at all")

    bits = np.unpackbits(d.view(np.uint8).reshape(-1, 2), axis=1, bitorder="little")
    sixteen = sum(len(c1.compress(np.packbits(bits[:, k]).tobytes())) for k in range(16))
    two = sum(len(c1.compress(p)) for p in
              (d.view(np.uint8).reshape(-1, 2)[:, 0].tobytes(), byte_plane))
    print(f"\n  16 bit planes, each compressed : {sixteen:>11,} bytes")
    print(f"   2 byte planes, each compressed : {two:>11,} bytes  "
          f"({(sixteen-two)/two*100:+.0f}%)")
    print("\n  EXCEPTION: bit-packing a *single* flag costs 1/8 byte per element")
    print("  regardless of compressibility -- which is why PFor's bitmap wins.")


if __name__ == "__main__":
    main()
