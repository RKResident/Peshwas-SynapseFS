"""zstd level is non-monotone here, and other compressors do not pay off.

Reproduces ARCHITECTURE.md 4.1.2.

Levels 1-2 use zstd's fast/dfast match-finders; 3+ switch to greedy/lazy, which
hunt for better matches that do not exist in run-heavy data and emit more
literals. So level 1 is both *smaller* and ~2x faster than level 3.

Also compares lzma/bz2/brotli/zlib. lzma wins on size and loses decisively on
decompression speed, which is what the FUSE read path is graded on.
"""

from __future__ import annotations

import argparse
import bz2
import lzma
import time
import zlib

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import shuffle
from tools.experiments._common import add_common_args, header, load_streams


def build(streams):
    out = []
    for t, b, w in streams:
        out.append(shuffle((t - b).astype(t.dtype).tobytes(), w))
    return out


def bench(label, blobs, comp, decomp, raw):
    t0 = time.perf_counter(); packed = [comp(x) for x in blobs]
    ce = time.perf_counter() - t0
    t0 = time.perf_counter(); [decomp(x) for x in packed]
    de = time.perf_counter() - t0
    n = sum(len(x) for x in packed)
    print(f"  {label:<22}{n:>12,}{n/raw*100:>9.2f}%{raw/1048576/ce:>10.0f}MB/s"
          f"{raw/1048576/de:>11.0f}MB/s")


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--gap", type=int, default=3)
    args = ap.parse_args()

    streams = load_streams(args.checkpoints, args.newest - args.gap, args.newest)
    blobs = build(streams)
    raw = sum(len(x) for x in blobs)
    header(f"gap {args.gap}, {raw/1048576:.2f} MiB after subtract + byte shuffle")

    print(f"  {'codec':<22}{'bytes':>12}{'ratio':>9}{'compress':>12}{'decompress':>13}")
    dz = zstd.ZstdDecompressor()
    for lvl in (1, 2, 3, 4, 6, 9, 12):
        c = zstd.ZstdCompressor(level=lvl)
        bench(f"zstd L{lvl}", blobs, c.compress, dz.decompress, raw)
    print()
    bench("zlib L6", blobs, lambda x: zlib.compress(x, 6), zlib.decompress, raw)
    bench("bz2 L9", blobs, lambda x: bz2.compress(x, 9), bz2.decompress, raw)
    bench("lzma preset6", blobs, lambda x: lzma.compress(x, preset=6), lzma.decompress, raw)
    try:
        import brotli
        bench("brotli q5", blobs, lambda x: brotli.compress(x, quality=5),
              brotli.decompress, raw)
    except ImportError:
        print("  brotli q5              (not installed -- pip install brotli)")
    print("\n  Levels 1-2 use fast/dfast; 3+ use greedy/lazy. On run-heavy data")
    print("  the lazy strategies emit more literals, so level 1 is smaller AND faster.")


if __name__ == "__main__":
    main()
