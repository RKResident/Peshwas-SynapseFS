"""Elf+ (Li et al.) implemented and measured against our codec.

Elf compresses a stream of floats in two stages. The ERASER rewrites each
value v as a nearby v' whose mantissa ends in many zero bits, exploiting the
fact that a decimal number like 20.15 needs far fewer bits than a double
provides; the trailing zeros then make the XOR of consecutive values cheap.
Recovery is exact because the number of significant decimal digits, beta*, is
stored alongside -- rounding v' back to that many digits returns v.

The bet is that the data is DECIMAL in origin. That is true of sensor
readings and false of neural network weights, which come out of gradient
descent with every mantissa bit meaningful. This module measures what that
costs rather than assuming it.

It matters where Elf could plug in. Elf is a single-stream compressor: it has
no notion of a base checkpoint, so it cannot compete with the delta path,
which gets its compression from cross-checkpoint redundancy Elf cannot see.
The honest comparison is against RAW_SHUFFLE_ZSTD on a FULL (anchor) commit,
which is the one place we compress a checkpoint on its own.

A synthetic decimal series is included as a positive control. Reporting "Elf
did not help our data" is worthless without first showing this implementation
reproduces Elf's advantage on the data it was designed for.

Every measurement here is verified by round-tripping: `--verify` decodes the
bitstream and asserts bit-identical recovery before any ratio is printed.
"""

from __future__ import annotations

import argparse
import math
import zlib
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path

import numpy as np
import zstandard as zstd

getcontext().prec = 40

# Elf quantises the leading-zero count into 8 buckets so it costs 3 bits
# instead of 6. The buckets are exponential because XOR leading-zero counts
# cluster high -- consecutive similar values agree in the sign and exponent.
LEAD_BUCKETS = {64: (0, 8, 12, 16, 18, 20, 22, 24),
                32: (0, 6, 10, 12, 14, 16, 18, 20),
                # Not in the paper: fp16 is what our checkpoints actually hold,
                # and Elf is specified only for double and single. The buckets
                # are the 32-bit ladder halved, keeping the same shape -- dense
                # near the top, where XOR leading-zero counts pile up.
                16: (0, 3, 5, 6, 7, 8, 9, 10)}
SPEC = {64: (52, 1023, np.uint64, np.float64),
        32: (23, 127, np.uint32, np.float32),
        16: (10, 15, np.uint16, np.float16)}


class BitWriter:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, bits: int) -> None:
        if bits <= 0:
            return
        self.acc = (self.acc << bits) | (value & ((1 << bits) - 1))
        self.n += bits
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def finish(self) -> bytes:
        if self.n:
            self.out.append((self.acc << (8 - self.n)) & 0xFF)
            self.acc = self.n = 0
        return bytes(self.out)

    @property
    def bits(self) -> int:
        return len(self.out) * 8 + self.n


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, bits: int) -> int:
        v = 0
        for _ in range(bits):
            byte = self.data[self.pos >> 3]
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v


def decimal_info(text: str) -> tuple[int, int, int]:
    """(alpha, beta, sp) from a shortest round-trip decimal rendering.

    alpha = digits after the point, beta = significant digits, sp = the
    decimal exponent of the leading digit (floor(log10|v|)). The paper computes
    these numerically for speed; here correctness is the point and the ratio is
    identical either way, so this uses the exact decimal rendering.
    """
    d = Decimal(text).normalize()
    _, digits, exp = d.as_tuple()
    beta = len(digits)
    return (-exp if exp < 0 else 0), beta, beta - 1 + exp


def _round_to(value: Decimal, alpha: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-alpha), rounding=ROUND_HALF_UP)


class Elf:
    """Eraser + Restorer for one float width."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.mant, self.bias, self.uint, self.flt = SPEC[width]
        self.buckets = LEAD_BUCKETS[width]

    def to_bits(self, v) -> int:
        return int(np.array([v], dtype=self.flt).view(self.uint)[0])

    def to_float(self, b: int):
        return np.array([b], dtype=self.uint).view(self.flt)[0]

    def render(self, v) -> str:
        return repr(float(v)) if self.width == 64 else str(self.flt(v))

    @property
    def max_beta(self) -> int:
        """Beyond this the Eraser cannot pay for the 4-bit beta* it stores.

        The paper uses 16 for double. fp16 carries only ~3.3 decimal digits,
        so its beta* never reaches 16 and the paper's cutoff would never
        fire -- the cutoff has to scale with the format or it stops being a
        cutoff at all.
        """
        return {64: 16, 32: 9, 16: 5}[self.width]

    def erase(self, v) -> tuple[int, int | None, bool]:
        """Return (bits_of_v', beta*, corrected). beta* None means no erasure."""
        bits = self.to_bits(v)
        if not np.isfinite(v) or v == 0:
            return bits, None, False
        alpha, beta, _ = decimal_info(self.render(v))
        if beta >= self.max_beta or alpha == 0:
            return bits, None, False

        raw_exp = (bits >> self.mant) & ((1 << (self.width - self.mant - 1)) - 1)
        if raw_exp == 0:                      # subnormal: treat as e = 1
            raw_exp = 1
        g = math.ceil(alpha * math.log2(10)) + raw_exp - self.bias
        if self.mant - g <= 4:                # too few bits erased to pay for beta*
            return bits, None, False

        # The paper's g is a closed form; a rounding boundary can leave it one
        # bit short. Widening until the value round-trips keeps the codec
        # lossless by construction rather than by trusting the formula.
        for extra in range(0, 4):
            keep = g + extra
            if keep >= self.mant:
                return bits, None, False
            erased = bits & ~((1 << (self.mant - keep)) - 1)
            cand = self.to_float(erased)
            if cand == 0 or not np.isfinite(cand):
                continue
            if self.restore(int(erased), beta) == self.to_bits(v):
                if self.mant - keep <= 4:
                    return bits, None, False
                return int(erased), beta, extra > 0
        return bits, None, False

    def restore(self, bits: int, beta: int) -> int:
        vprime = self.to_float(bits)
        if vprime == 0 or not np.isfinite(vprime):
            return bits
        _, _, sp = decimal_info(self.render(vprime))
        if beta == 0:
            return self.to_bits(self.flt(10.0 ** (sp + 1)))
        alpha = beta - (sp + 1)
        d = _round_to(Decimal(self.render(vprime)), alpha)
        return self.to_bits(self.flt(float(d)))

    # ---- XORcmp -------------------------------------------------------

    def _bucket(self, lead: int) -> tuple[int, int]:
        idx = 0
        for i, b in enumerate(self.buckets):
            if b <= lead:
                idx = i
        return idx, self.buckets[idx]

    def compress(self, values) -> tuple[bytes, dict]:
        w = BitWriter()
        W, stats = self.width, {"erased": 0, "beta_reused": 0, "corrected": 0,
                                "n": len(values), "erase_bits": 0}
        prev = prev_lead = prev_trail = None
        prev_beta = -1
        for v in values:
            bits, beta, corrected = self.erase(v)
            before = w.bits
            if beta is None:
                w.write(0b10, 2)
            elif beta == prev_beta:
                w.write(0b0, 1)
                stats["beta_reused"] += 1
            else:
                w.write(0b11, 2)
                w.write(beta, 4)
            if beta is not None:
                stats["erased"] += 1
                prev_beta = beta
                stats["corrected"] += corrected
            stats["erase_bits"] += w.bits - before

            if prev is None:
                trail = W if bits == 0 else (bits & -bits).bit_length() - 1
                w.write(trail, 7)
                w.write(bits >> trail, W - trail)
                prev_lead, prev_trail = 0, trail
            else:
                x = bits ^ prev
                if x == 0:
                    w.write(0b01, 2)
                else:
                    lead = W - x.bit_length()
                    trail = (x & -x).bit_length() - 1
                    idx, lq = self._bucket(lead)
                    if lq == prev_lead and trail >= prev_trail:
                        w.write(0b00, 2)
                        w.write(x >> prev_trail, W - prev_lead - prev_trail)
                    else:
                        centre = W - lq - trail
                        if centre <= 16:
                            w.write(0b10, 2)
                            w.write(idx, 3)
                            w.write(centre - 1, 4)
                        else:
                            w.write(0b11, 2)
                            w.write(idx, 3)
                            w.write(centre - 1, 6)
                        w.write(x >> trail, centre)
                        prev_lead, prev_trail = lq, trail
            prev = bits
        return w.finish(), stats

    def decompress(self, blob: bytes, n: int) -> np.ndarray:
        r = BitReader(blob)
        W = self.width
        out = np.empty(n, dtype=self.uint)
        prev = prev_lead = prev_trail = None
        prev_beta = -1
        for i in range(n):
            if r.read(1) == 0:
                beta = prev_beta
            elif r.read(1) == 0:
                beta = None
            else:
                beta = r.read(4)
            if beta is not None:
                prev_beta = beta

            if prev is None:
                trail = r.read(7)
                bits = r.read(W - trail) << trail
                prev_lead, prev_trail = 0, trail
            else:
                flag = r.read(2)
                if flag == 0b01:
                    bits = prev
                elif flag == 0b00:
                    x = r.read(W - prev_lead - prev_trail) << prev_trail
                    bits = prev ^ x
                else:
                    idx = r.read(3)
                    centre = r.read(4 if flag == 0b10 else 6) + 1
                    lq = self.buckets[idx]
                    trail = W - lq - centre
                    x = r.read(centre) << trail
                    bits = prev ^ x
                    prev_lead, prev_trail = lq, trail
            prev = bits
            out[i] = self.restore(bits, beta) if beta is not None else bits
        return out


def synthetic_series(n: int, rng) -> np.ndarray:
    """A sensor-like decimal series: two decimal places, smooth drift."""
    walk = np.cumsum(rng.normal(0, 0.05, n)) + 20.0
    return np.round(walk, 2)


def measure(label: str, values: np.ndarray, width: int, verify: bool) -> None:
    elf = Elf(width)
    blob, st = elf.compress(values)
    if verify:
        back = elf.decompress(blob, len(values))
        truth = values.astype(SPEC[width][3]).view(SPEC[width][2])
        if not np.array_equal(back, truth):
            bad = int((back != truth).sum())
            raise SystemExit(f"{label}: NOT LOSSLESS -- {bad}/{len(values)} differ")

    raw = values.astype(SPEC[width][3]).tobytes()
    zst = len(zstd.ZstdCompressor(level=1).compress(raw))
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width // 8)
    shuf = np.concatenate([np.ascontiguousarray(a[:, i]) for i in range(width // 8)])
    shz = len(zstd.ZstdCompressor(level=1).compress(shuf.tobytes()))
    gz = len(zlib.compress(raw, 6))
    n = st["n"]
    print(f"  {label:<26} {len(blob)/len(raw)*100:7.2f}% {shz/len(raw)*100:8.2f}% "
          f"{zst/len(raw)*100:7.2f}% {gz/len(raw)*100:7.2f}% "
          f"{st['erased']/n*100:8.1f}% {st['erase_bits']/n:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="tools/tools/benchmark-mlp/epoch20.safetensors")
    ap.add_argument("--tensor", default="features.3.weight")
    ap.add_argument("--n", type=int, default=60000,
                    help="values per series; the Decimal path is slow by design")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify = not args.no_verify
    rng = np.random.default_rng(0)

    from synapsefs.safetensors_io import SafetensorsFile
    with SafetensorsFile(args.checkpoint) as f:
        spec = f.spec(args.tensor)
        w16 = f.rows(args.tensor, 0, spec.num_rows).ravel()[:args.n]
    weights = w16.view(np.float16).astype(np.float64)

    print(f"{args.n} values per series; 'Elf+' vs our shuffle+zstd and two baselines")
    print("erased% is how often the Eraser fired; flag bits/value is what Elf+ "
          "spends\non the beta* prefix code.\n")
    print(f"  {'series':<26} {'Elf+':>7} {'shuf+zstd':>8} {'zstd':>7} {'gzip':>7} "
          f"{'erased%':>8} {'flag/v':>7}")
    measure("sensor decimals (control)", synthetic_series(args.n, rng), 64, verify)
    measure("NN weights f64 (upcast)", weights, 64, verify)
    measure("NN weights f32 (upcast)", weights.astype(np.float32), 32, verify)
    measure("NN weights f16 (native)", weights.astype(np.float16), 16, verify)


if __name__ == "__main__":
    main()
