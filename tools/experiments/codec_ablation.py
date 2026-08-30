"""The transform matrix: which codec steps actually pay.

Reproduces ARCHITECTURE.md 4.1 and its "deliberately not in this pipeline"
table. Every transform here is exactly reversible; the question is only which
one compresses best.

Two operators (modular subtraction, XOR) crossed with the layout schemes, at
each of the gaps the star topology actually produces. Gap matters: the monotone
key wins at gap 1 and loses from gap 2 onward, which is why a single-pair
measurement is not enough.
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import (DEFAULT_LEVEL, dtype_spec, shuffle,
                                   to_monotone_key, zigzag)
from tools.experiments._common import add_common_args, header, load_pair


def bit_shuffle(buf: bytes, w: int) -> bytes:
    a = np.frombuffer(buf, np.uint8).reshape(-1, w)
    return np.packbits(np.unpackbits(a, axis=1, bitorder="little").T.copy(),
                       axis=1, bitorder="little").tobytes()


def two_planes(d):
    a = d.view(np.uint8).reshape(-1, 2)
    return a[:, 0].tobytes(), a[:, 1].tobytes()


def pfor(v, threshold=256):
    """Patched Frame Of Reference: a bit-packed flag plus byte-aligned planes.

    The escape-byte idea done right. An 8-bit marker costs 0.72 bytes/element
    at gap 3 (72% of deltas exceed 255); a 1-bit flag costs 0.125 regardless.
    """
    small = v < threshold
    return (np.packbits(small).tobytes(),
            v[small].astype(np.uint8).tobytes(),
            shuffle(v[~small].astype(np.uint16).tobytes(), 2))


SCHEMES = [
    ("plain zstd",              lambda d: (d.tobytes(),)),
    ("byte shuffle  (current)", lambda d: (shuffle(d.tobytes(), 2),)),
    ("2 byte planes, separate", two_planes),
    ("bit shuffle",             lambda d: (bit_shuffle(d.tobytes(), 2),)),
    ("PFor bitmap+split",       lambda d: pfor(d)),
    ("zigzag + byte shuffle",   lambda d: (shuffle(zigzag(d).tobytes(), 2),)),
    ("zigzag + PFor",           lambda d: pfor(zigzag(d))),
]


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 3],
                    help="commit distances to measure (the star produces 1-3)")
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    ap.add_argument("--dtype", default="F16", choices=["F16", "BF16"],
                    help="which corpus. Both are two bytes, but bf16 spends 8 "
                         "bits on the exponent and 7 on the mantissa where fp16 "
                         "spends 5 and 10, so the byte shuffle cuts in a "
                         "different place and every scheme here can change rank.")
    args = ap.parse_args()
    c = zstd.ZstdCompressor(level=args.level)

    header(f"transform matrix, zstd L{args.level}",
           "* marks the winner in each row.  Lower is better.")

    for gap in args.gaps:
        t, b, raw = load_pair(args.checkpoints, args.newest - gap, args.newest,
                              dtype=args.dtype)
        _, kind = dtype_spec(args.dtype)
        operators = [
            ("SUBTRACT", (t - b).astype(np.uint16)),
            ("XOR", (t ^ b).astype(np.uint16)),
            ("key-SUB", (to_monotone_key(t, kind) - to_monotone_key(b, kind)).astype(np.uint16)),
        ]
        print(f"gap {gap}   ({raw/1048576:.2f} MiB)")
        print(f"  {'scheme':<26}" + "".join(f"{n:>12}" for n, _ in operators))
        for label, fn in SCHEMES:
            vals = [sum(len(c.compress(p)) for p in fn(d)) / raw * 100
                    for _, d in operators]
            best = min(vals)
            print(f"  {label:<26}"
                  + "".join(f"{v:>11.2f}%" + ("*" if v == best else " ") for v in vals))
        print()


if __name__ == "__main__":
    main()
