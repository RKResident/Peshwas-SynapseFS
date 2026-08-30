"""Subtract the whole 16-bit word, or each IEEE field separately?

The codec computes `target_bits - base_bits` mod 2**16 over the entire fp16
word: sign, exponent and mantissa in one subtraction. The obvious objection is
that these are three different quantities glued together, and that differencing
them separately ought to respect their structure.

The argument against is not obvious and is the reason to measure rather than
reason. A whole-word subtraction on IEEE floats is a *logarithmic-scale
distance*: consecutive representable floats differ by exactly 1 in the integer
encoding, and that stays true ACROSS an exponent boundary. So 1.9995 and 2.0000
sit a few ULPs apart in whole-word terms even though their exponent fields
differ by one and their mantissa fields differ by nearly the full range. Borrow
propagation from mantissa into exponent is not an accident being tolerated, it
is the mechanism that makes the encoding continuous.

Split the fields and that continuity is gone: every weight that crosses a power
of two turns one small delta into two large ones. How often that happens is an
empirical question about drift magnitude, which is what this measures.

The sign bit is the opposite case. fp16 is sign-magnitude, so a weight crossing
zero moves ~0x8000 in whole-word terms while barely moving in value. There a
separate XOR costs one bit instead of a near-random high byte.

Schemes are compared at equal width where possible: `packed` keeps 16 bits per
element so no padding is paid, and only the arithmetic changes.
"""

from __future__ import annotations

import argparse

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL
from synapsefs.safetensors_io import SafetensorsFile

C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)
SIGN, EXP, MANT = 15, 10, 0
EXP_BITS, MANT_BITS = 5, 10


def shuffle_bytes(a: np.ndarray) -> bytes:
    """The codec's byte shuffle, for any element width."""
    raw = a.tobytes()
    w = a.dtype.itemsize
    m = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w)
    return np.concatenate([np.ascontiguousarray(m[:, i]) for i in range(w)]).tobytes()


def z(b: bytes) -> int:
    return len(C.compress(b))


def fields(bits: np.ndarray):
    s = (bits >> SIGN) & 1
    e = (bits >> EXP) & ((1 << EXP_BITS) - 1)
    m = bits & ((1 << MANT_BITS) - 1)
    return s, e, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tools/tools/benchmark/epoch02.safetensors")
    ap.add_argument("--target", default="tools/tools/benchmark/epoch03.safetensors")
    ap.add_argument("--tensor", default="features.28.weight")
    args = ap.parse_args()

    with SafetensorsFile(args.base) as b, SafetensorsFile(args.target) as t:
        spec = t.spec(args.tensor)
        tb = t.rows(args.tensor, 0, spec.num_rows).ravel().copy()
        bb = b.rows(args.tensor, 0, spec.num_rows).ravel().copy()
    raw = tb.nbytes
    st, et, mt = fields(tb)
    sb, eb, mb = fields(bb)

    print(f"{args.tensor}  {tb.size:,} elements  ({raw/2**20:.1f} MiB)\n")
    print(f"  sign differs      : {(st != sb).mean()*100:6.2f}% of weights")
    print(f"  exponent differs  : {(et != eb).mean()*100:6.2f}%   "
          f"<- every one of these is a power-of-two crossing")
    print(f"  mantissa differs  : {(mt != mb).mean()*100:6.2f}%\n")

    results = []

    whole = (tb - bb).astype(np.uint16)
    results.append(("whole-word subtract (current)", z(shuffle_bytes(whole)), raw))

    # Same 16 bits, but each field differenced in its own modulus.
    ds = (st ^ sb).astype(np.uint16)
    de = ((et - eb) & ((1 << EXP_BITS) - 1)).astype(np.uint16)
    dm = ((mt - mb) & ((1 << MANT_BITS) - 1)).astype(np.uint16)
    packed = (ds << SIGN) | (de << EXP) | dm
    results.append(("per-field, repacked to 16b", z(shuffle_bytes(packed)), raw))

    # Same arithmetic, but each field gets its own byte-aligned stream.
    planes = (z(np.packbits(ds.astype(np.uint8)).tobytes())
              + z(de.astype(np.uint8).tobytes())
              + z(shuffle_bytes(dm.astype(np.uint16))))
    results.append(("per-field, separate planes", planes, raw))

    # Sign split off, magnitude (exp+mantissa) still differenced as one word.
    mag_t = (tb & 0x7FFF).astype(np.uint16)
    mag_b = (bb & 0x7FFF).astype(np.uint16)
    hybrid = (z(np.packbits(ds.astype(np.uint8)).tobytes())
              + z(shuffle_bytes((mag_t - mag_b).astype(np.uint16))))
    results.append(("sign XOR + whole magnitude", hybrid, raw))

    # XOR of the whole word, for reference.
    results.append(("whole-word XOR", z(shuffle_bytes((tb ^ bb).astype(np.uint16))), raw))

    print(f"  {'scheme':<32} {'stored':>12} {'ratio':>9} {'vs current':>11}")
    ref = results[0][1]
    for name, size, base in results:
        print(f"  {name:<32} {size:12,} {size/base*100:8.2f}% "
              f"{(size-ref)/base*100:+10.2f}pp")

    # Where does the per-field scheme actually lose? Compare the magnitude of
    # the two mantissa deltas on the weights that crossed an exponent.
    crossed = et != eb
    w_small = np.abs(whole.view(np.int16).astype(np.int32))
    print(f"\n  median |delta| on weights that CROSSED an exponent boundary:")
    print(f"    whole-word     {np.median(w_small[crossed]):8.0f}")
    print(f"    per-field mant {np.median(np.minimum(dm[crossed], 1024 - dm[crossed])):8.0f}"
          f"   (of a 1024-wide field)")
    print(f"  on weights that did not cross:")
    print(f"    whole-word     {np.median(w_small[~crossed]):8.0f}")
    print(f"    per-field mant {np.median(np.minimum(dm[~crossed], 1024 - dm[~crossed])):8.0f}")


if __name__ == "__main__":
    main()
