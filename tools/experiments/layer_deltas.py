"""Per-layer delta statistics across every consecutive epoch pair.

Reproduces the layer-wise table in `docs/ARCHITECTURE.md` §4.1.

The codec compresses a *residual*, so the only thing that decides how well a
checkpoint packs is how far each tensor moved since the base. That is not a
single number: a BatchNorm `running_var` buffer and a 1792x1792 convolution
kernel drift by different amounts, for different reasons, and their trends
over training go in opposite directions. This script measures each tensor
separately over all 24 consecutive pairs so those effects stay visible.

Units are ULPs -- the raw integer distance between the two bit patterns, which
is exactly what the codec sees. `<128` is the fraction that fits in one byte,
which is the fraction PFor would encode in its narrow plane.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL, dtype_spec, shuffle
from synapsefs.safetensors_io import SafetensorsFile

DEFAULT_CHECKPOINTS = Path("tools/tools/benchmark")

def classify(name: str, nelem: int) -> str:
    if name.endswith("num_batches_tracked"):
        return "bn.counter"
    if name.endswith("running_mean"):
        return "bn.running_mean"
    if name.endswith("running_var"):
        return "bn.running_var"
    if name.endswith(".bias"):
        return "bias"
    # A BN scale is 1-D; a conv kernel is 4-D. Only the element count is
    # available here, and the 1-D tensors are the small ones.
    return "conv.weight" if nelem > 4096 else "bn.weight"


def ratio(delta_bits: np.ndarray, width: int) -> float:
    stream = shuffle(delta_bits.tobytes(), width)
    return len(zstd.ZstdCompressor(level=DEFAULT_LEVEL).compress(stream)) / len(stream)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no-compress", action="store_true",
                    help="skip the zstd pass (much faster, drops the ratio column)")
    args = ap.parse_args()

    # name -> list over pairs of (median|d|, zero_frac, sub128_frac, ratio)
    per_tensor: dict[str, list] = defaultdict(list)
    per_pair: list = []
    order: list[str] = []
    sizes: dict[str, int] = {}

    prev = None
    for epoch in range(1, args.epochs + 1):
        path = args.checkpoints / f"epoch{epoch:02d}.safetensors"
        cur = {}
        with SafetensorsFile(path) as f:
            for name in f.names():
                spec = f.spec(name)
                if spec.dtype != "F16":
                    continue
                cur[name] = f.rows(name, 0, spec.num_rows).ravel().copy()
        if prev is not None:
            tot_n = tot_zero = tot_sub = 0
            mags = []
            raw = comp = 0
            for name, t in cur.items():
                b = prev[name]
                d = (t - b).astype(np.uint16)          # exact mod 2**16
                sd = d.view(np.int16).astype(np.int32)  # wrapped signed distance
                mag = np.abs(sd)
                r = None if args.no_compress else ratio(d, 2)
                per_tensor[name].append(
                    (float(np.median(mag)), float((mag == 0).mean()),
                     float((mag < 128).mean()), r))
                if name not in sizes:
                    order.append(name)
                    sizes[name] = t.nbytes
                tot_n += mag.size
                tot_zero += int((mag == 0).sum())
                tot_sub += int((mag < 128).sum())
                mags.append(mag)
                if r is not None:
                    raw += d.nbytes
                    comp += r * d.nbytes
            allmag = np.concatenate(mags)
            per_pair.append((epoch - 1, epoch, float(np.median(allmag)),
                             float(np.mean(allmag)), float(np.percentile(allmag, 99)),
                             tot_zero / tot_n, tot_sub / tot_n,
                             None if args.no_compress else comp / raw))
        prev = cur
        print(f"  ...epoch {epoch:02d}", end="\r", flush=True)
    print(" " * 30, end="\r")

    print("PER-EPOCH-PAIR TOTALS  (all F16 tensors pooled, magnitudes in ULPs)")
    print()
    print(f"  {'pair':>9}  {'med|d|':>7}  {'mean|d|':>8}  {'p99|d|':>7}  "
          f"{'zero':>7}  {'<128':>7}  {'ratio':>7}")
    for lo, hi, med, mean, p99, z, s, r in per_pair:
        rs = "-" if r is None else f"{r*100:6.2f}%"
        print(f"  {lo:02d}->{hi:02d}    {med:7.0f}  {mean:8.1f}  {p99:7.0f}  "
              f"{z*100:6.2f}%  {s*100:6.2f}%  {rs:>7}")

    print()
    print("PER-TENSOR, AVERAGED OVER ALL 24 PAIRS")
    print()
    print(f"  {'tensor':<32} {'class':<16} {'MiB':>6}  {'med|d|':>7}  "
          f"{'zero':>7}  {'<128':>7}  {'ratio':>7}  {'trend':>7}")
    for name in order:
        rows = per_tensor[name]
        med = np.mean([r[0] for r in rows])
        z = np.mean([r[1] for r in rows])
        s = np.mean([r[2] for r in rows])
        rr = [r[3] for r in rows]
        rs = "-" if rr[0] is None else f"{np.mean(rr)*100:6.2f}%"
        # last five pairs vs first five: does the tensor settle down?
        trend = np.mean([r[0] for r in rows[-5:]]) - np.mean([r[0] for r in rows[:5]])
        print(f"  {name:<32} {classify(name, sizes[name]//2):<16} "
              f"{sizes[name]/2**20:6.2f}  {med:7.1f}  {z*100:6.2f}%  "
              f"{s*100:6.2f}%  {rs:>7}  {trend:+7.1f}")

    print()
    print("BY CLASS  (bytes-weighted where it matters)")
    print()
    agg: dict[str, list] = defaultdict(list)
    for name in order:
        agg[classify(name, sizes[name] // 2)].append(name)
    print(f"  {'class':<18} {'tensors':>7} {'MiB':>8}  {'med|d|':>7}  "
          f"{'zero':>7}  {'<128':>7}  {'ratio':>7}")
    for cls, names in sorted(agg.items(), key=lambda kv: -sum(sizes[n] for n in kv[1])):
        nbytes = sum(sizes[n] for n in names)
        med = np.mean([r[0] for n in names for r in per_tensor[n]])
        z = np.mean([r[1] for n in names for r in per_tensor[n]])
        s = np.mean([r[2] for n in names for r in per_tensor[n]])
        rr = [r[3] for n in names for r in per_tensor[n]]
        rs = "-" if rr[0] is None else f"{np.mean(rr)*100:6.2f}%"
        print(f"  {cls:<18} {len(names):7d} {nbytes/2**20:8.2f}  {med:7.1f}  "
              f"{z*100:6.2f}%  {s*100:6.2f}%  {rs:>7}")


if __name__ == "__main__":
    main()
