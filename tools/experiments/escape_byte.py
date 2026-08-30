"""The escape-byte varint against PFor's bitmap, on bf16.

Both encode the same observation -- most residuals fit in one byte, a few do
not -- and they differ in two independent choices that are worth separating:

  how the exception is FLAGGED   an 8-bit inline marker (0xFF) or a 1-bit
                                 entry in a packed bitmap
  how the bytes are LAID OUT     one interleaved stream, or homogeneous
                                 planes compressed separately

The escape byte was measured and rejected on fp16 for a reason that no longer
holds: 72% of deltas there exceeded 255, so the marker cost 0.72 bytes per
element and a 1-bit flag cost 0.125 regardless. bf16 has three fewer mantissa
bits, the median delta falls from 510 to 27, and 86.5% of zigzag deltas now fit
in a byte -- so the marker fires rarely and the comparison has to be redone.

Raw size is reported next to the compressed size on purpose. The two schemes
land within 0.01 bytes/element of each other before compression, so anything
that separates them is a statement about how well zstd models the result, not
about how many bytes the encoding needs.

Encoding: zigzag first, so negatives are small (under plain subtraction a
negative delta is a huge uint16 and would escape every time). Then values
0..254 are one byte; 255 and above are 0xFF followed by the value as two
little-endian bytes.
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL, shuffle
from synapsefs.safetensors_io import SafetensorsFile

C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)
ESCAPE = 255


def zigzag16(delta_bits: np.ndarray) -> np.ndarray:
    v = delta_bits.view(np.int16).astype(np.int32)
    return ((v << 1) ^ (v >> 31)).astype(np.uint16)


def escape_stream(v: np.ndarray) -> bytes:
    """One interleaved byte stream: 1 byte if small, else 0xFF + 2 bytes."""
    small = v < ESCAPE
    lengths = np.where(small, 1, 3).astype(np.int64)
    offsets = np.empty(v.size, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])
    out = np.empty(int(lengths.sum()), dtype=np.uint8)
    out[offsets[small]] = v[small].astype(np.uint8)
    big = offsets[~small]
    wide = v[~small]
    out[big] = ESCAPE
    out[big + 1] = (wide & 0xFF).astype(np.uint8)
    out[big + 2] = (wide >> 8).astype(np.uint8)
    return out.tobytes()


def marker_planes(v: np.ndarray):
    """8-bit marker kept inline, wide values moved to their own shuffled plane."""
    small = v < ESCAPE
    lane = np.where(small, v, ESCAPE).astype(np.uint8)
    return lane.tobytes(), shuffle(v[~small].astype(np.uint16).tobytes(), 2)


def bitmap_planes(v: np.ndarray):
    """PFor: 1-bit flag, an 8-bit plane, and a shuffled 16-bit plane."""
    small = v < 256
    return (np.packbits(small).tobytes(),
            v[small].astype(np.uint8).tobytes(),
            shuffle(v[~small].astype(np.uint16).tobytes(), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="tools/tools/benchmark-bf16")
    ap.add_argument("--newest", type=int, default=25)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--dtype", default="BF16")
    args = ap.parse_args()

    for gap in args.gaps:
        T, B, raw = [], [], 0
        lo, hi = args.newest - gap, args.newest
        with SafetensorsFile(f"{args.checkpoints}/epoch{lo:02d}.safetensors") as b, \
             SafetensorsFile(f"{args.checkpoints}/epoch{hi:02d}.safetensors") as t:
            for name in t.names():
                s = t.spec(name)
                if s.dtype != args.dtype:
                    continue
                T.append(t.rows(name, 0, s.num_rows).ravel().copy())
                B.append(b.rows(name, 0, s.num_rows).ravel().copy())
                raw += T[-1].nbytes
        d = (np.concatenate(T) - np.concatenate(B)).astype(np.uint16)
        zz = zigzag16(d)
        n = d.size
        esc = (zz >= ESCAPE).mean()

        print(f"gap {gap}   {raw/2**20:.2f} MiB   "
              f"{esc*100:.1f}% of zigzag deltas escape\n")
        print(f"  {'scheme':<40} {'raw B/elem':>11} {'stored':>12} {'ratio':>8}")

        rows = [
            ("byte shuffle, no zigzag  (current)",
             (shuffle(d.tobytes(), 2),)),
            ("zigzag + byte shuffle",
             (shuffle(zz.tobytes(), 2),)),
            ("zigzag + escape byte, one stream",
             (escape_stream(zz),)),
            ("zigzag + escape byte, wide plane split",
             marker_planes(zz)),
            ("zigzag + PFor bitmap",
             bitmap_planes(zz)),
        ]
        for label, parts in rows:
            rawb = sum(len(p) for p in parts) / n
            size = sum(len(C.compress(p)) for p in parts)
            print(f"  {label:<40} {rawb:11.3f} {size:12,} {size/raw*100:7.2f}%")
        print()


if __name__ == "__main__":
    main()
