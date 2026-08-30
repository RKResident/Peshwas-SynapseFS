"""XORcmp's encoder applied across checkpoints instead of across elements.

`elf_codec.py` measured Elf+ as specified: the stream is one tensor's weights
in memory order, and XORcmp diffs each value against its NEIGHBOUR. That axis
has no structure -- adjacent weights are independent -- so it expanded the
data. The mechanism was never given a fair test.

This is the fair test. The XOR is taken along the EPOCH axis, per parameter:

    x[i] = bits_t[i] XOR bits_{t-1}[i]

which is where our data is smooth, and XORcmp's flag scheme is used purely as
the entropy coder for that stream: bucketed leading-zero count, a short or
long centre-bit field, and a reuse flag when the current value's window
matches the previous one's.

Why it might win: two fp16 numbers that are close agree in sign, exponent and
the top mantissa bits, so their XOR has a long run of leading zeros. That is
exactly the redundancy Gorilla and Chimp were built to exploit, and it is
structure our shuffle+zstd path only reaches indirectly.

Why it might lose: the centre field still has to carry every differing
mantissa bit, and `codec_headroom.py` measured those at 8.000 bits of entropy
-- incompressible. A coder that must emit them verbatim, plus a per-value
header, cannot beat one that emits them verbatim with no header.

Costs are computed exactly, bit for bit, by simulating the encoder. `--verify`
round-trips a slice through a real encoder and decoder to prove the accounting
is not fiction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL
from synapsefs.safetensors_io import SafetensorsFile

# Elf's ladders are dense where XOR leading-zero counts pile up. For 16-bit
# values the whole range is 0..16, so several spacings are worth trying rather
# than assuming the paper's shape rescales.
BUCKET_SETS = {
    "elf-scaled": (0, 3, 5, 6, 7, 8, 9, 10),
    "high-dense": (0, 4, 6, 8, 9, 10, 11, 12),
    "uniform": (0, 2, 4, 6, 8, 10, 12, 14),
}


def lead_trail(x: np.ndarray, width: int):
    """Leading and trailing zero counts of every nonzero entry."""
    lead = np.full(x.shape, width, dtype=np.int32)
    trail = np.full(x.shape, width, dtype=np.int32)
    nz = x != 0
    v = x[nz].astype(np.uint32)
    bl = np.zeros(v.shape, dtype=np.int32)
    tmp = v.copy()
    while tmp.any():
        bl += (tmp > 0)
        tmp >>= 1
    lead[nz] = width - bl
    low = (v & (~v + 1))
    tz = np.zeros(v.shape, dtype=np.int32)
    tmp = low.copy()
    while (tmp > 1).any():
        tz += (tmp > 1)
        tmp = np.where(tmp > 1, tmp >> 1, tmp)
    trail[nz] = tz
    return lead, trail


def cost_bits(x: np.ndarray, width: int, buckets, exact_lead: bool) -> int:
    """Exact XORcmp bit cost for this stream. Sequential: the reuse flag
    depends on the previous value's window, so this cannot be vectorised."""
    lead, trail = lead_trail(x, width)
    blist = list(buckets)
    lead_q = np.zeros_like(lead)
    idx = np.zeros_like(lead)
    if exact_lead:
        lead_q = lead.copy()
    else:
        for i, b in enumerate(blist):
            hit = lead >= b
            lead_q[hit] = b
            idx[hit] = i
    lead_w = 4 if exact_lead else 3
    cnt_w = 5 if width > 16 else 4

    total = 0
    prev_lead, prev_trail = 0, 0
    xl = x.tolist(); ll = lead_q.tolist(); tl = trail.tolist()
    for xi, li, ti in zip(xl, ll, tl):
        if xi == 0:
            total += 2
            continue
        if li == prev_lead and ti >= prev_trail:
            total += 2 + (width - prev_lead - prev_trail)
        else:
            centre = width - li - ti
            total += 2 + lead_w + cnt_w + centre
            prev_lead, prev_trail = li, ti
    return total


def encode(x: np.ndarray, width: int, buckets) -> bytes:
    """Real encoder, for round-trip proof."""
    out = bytearray(); acc = n = 0

    def put(v, b):
        nonlocal acc, n
        acc = (acc << b) | (v & ((1 << b) - 1)); n += b
        while n >= 8:
            n -= 8; out.append((acc >> n) & 0xFF); acc &= (1 << n) - 1

    lead, trail = lead_trail(x, width)
    blist = list(buckets)
    prev_lead, prev_trail = 0, 0
    for xi, li, ti in zip(x.tolist(), lead.tolist(), trail.tolist()):
        if xi == 0:
            put(0b01, 2); continue
        q = 0; qi = 0
        for i, b in enumerate(blist):
            if li >= b:
                q, qi = b, i
        if q == prev_lead and ti >= prev_trail:
            put(0b00, 2); put(xi >> prev_trail, width - prev_lead - prev_trail)
        else:
            centre = width - q - ti
            put(0b10, 2); put(qi, 3); put(centre - 1, 4); put(xi >> ti, centre)
            prev_lead, prev_trail = q, ti
    if n:
        out.append((acc << (8 - n)) & 0xFF)
    return bytes(out)


def decode(blob: bytes, count: int, width: int, buckets) -> np.ndarray:
    pos = 0

    def get(b):
        nonlocal pos
        v = 0
        for _ in range(b):
            v = (v << 1) | ((blob[pos >> 3] >> (7 - (pos & 7))) & 1); pos += 1
        return v

    out = np.zeros(count, dtype=np.uint16 if width == 16 else np.uint32)
    prev_lead, prev_trail = 0, 0
    for i in range(count):
        flag = get(2)
        if flag == 0b01:
            continue
        if flag == 0b00:
            out[i] = get(width - prev_lead - prev_trail) << prev_trail
        else:
            qi = get(3); centre = get(4) + 1
            q = buckets[qi]; t = width - q - centre
            out[i] = get(centre) << t
            prev_lead, prev_trail = q, t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=Path("tools/tools/benchmark"))
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--tensor", default="features.31.weight")
    ap.add_argument("--limit", type=int, default=4_000_000)
    ap.add_argument("--verify", type=int, default=200_000)
    args = ap.parse_args()

    fmt = str(args.checkpoints / "epoch{:02d}.safetensors")
    with SafetensorsFile(fmt.format(args.base)) as b, \
         SafetensorsFile(fmt.format(args.target)) as t:
        spec = t.spec(args.tensor)
        tb = t.rows(args.tensor, 0, spec.num_rows).ravel()[:args.limit].copy()
        bb = b.rows(args.tensor, 0, spec.num_rows).ravel()[:args.limit].copy()

    raw_bits = tb.size * 16
    xor = (tb ^ bb).astype(np.uint16)
    sub = (tb - bb).astype(np.uint16)
    C = zstd.ZstdCompressor(level=DEFAULT_LEVEL)

    def shuffled(d):
        by = d.view(np.uint8).reshape(-1, 2)
        return np.concatenate([np.ascontiguousarray(by[:, 0]),
                               np.ascontiguousarray(by[:, 1])]).tobytes()

    print(f"{args.tensor}, epoch {args.base}->{args.target}, "
          f"{tb.size:,} values ({raw_bits/8/2**20:.1f} MiB)\n")

    if args.verify:
        s = xor[:args.verify]
        blob = encode(s, 16, BUCKET_SETS["elf-scaled"])
        back = decode(blob, s.size, 16, BUCKET_SETS["elf-scaled"])
        assert np.array_equal(back, s), "XORcmp round-trip FAILED"
        print(f"  round-trip verified on {args.verify:,} values\n")

    print(f"  {'scheme':<40} {'ratio':>8} {'bits/value':>11}")
    for name, bk in BUCKET_SETS.items():
        c = cost_bits(xor, 16, bk, exact_lead=False)
        print(f"  {'XORcmp on cross-epoch XOR [' + name + ']':<40} "
              f"{c/raw_bits*100:7.2f}% {c/tb.size:11.2f}")
    c = cost_bits(xor, 16, (), exact_lead=True)
    print(f"  {'XORcmp, exact 4-bit lead':<40} {c/raw_bits*100:7.2f}% "
          f"{c/tb.size:11.2f}")
    cs = cost_bits(sub, 16, BUCKET_SETS['elf-scaled'], exact_lead=False)
    print(f"  {'XORcmp on SUBTRACTED delta':<40} {cs/raw_bits*100:7.2f}% "
          f"{cs/tb.size:11.2f}")

    print()
    for label, blob in (("XOR + shuffle + zstd", shuffled(xor)),
                        ("SUBTRACT + shuffle + zstd  (ours)", shuffled(sub))):
        n = len(C.compress(blob))
        print(f"  {label:<40} {n*8/raw_bits*100:7.2f}% {n*8/tb.size:11.2f}")

    # Does a general-purpose coder find anything left in the XORcmp bitstream?
    bs = encode(xor, 16, BUCKET_SETS["elf-scaled"])
    n = len(C.compress(bs))
    print(f"  {'XORcmp bitstream, then zstd':<40} {n*8/raw_bits*100:7.2f}% "
          f"{n*8/tb.size:11.2f}")


if __name__ == "__main__":
    main()
