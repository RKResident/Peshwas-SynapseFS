"""Star vs bead-chain: storage against reconstruction cost.

Reproduces ARCHITECTURE.md 4.3.

Both topologies store every Nth commit in full. They differ in what the
residuals in between diff against:

    star   A(full) <- B-A, C-A, D-A <- E(full)     reconstruct: 1 decode
    chain  A(full) <- B-A, C-B, D-C <- E(full)     reconstruct: up to N-1

Chain deltas are smaller because each diffs against its immediate predecessor.
Star reconstruction is cheaper because it is always one hop. This measures both
on real checkpoints rather than arguing about them.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import DEFAULT_LEVEL, shuffle, plain_stream
from synapsefs.safetensors_io import SafetensorsFile
from tools.experiments._common import add_common_args, header, path_for


def encode(target: np.ndarray, base: np.ndarray | None, c) -> bytes:
    if base is None:
        return c.compress(shuffle(target.tobytes(), 2))
    d = (target - base).astype(target.dtype)
    return c.compress(shuffle(d.tobytes(), 2))


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--interval", type=int, default=4, help="REBASE_INTERVAL")
    ap.add_argument("--tensors", nargs="+", default=[
        "features.0.weight", "features.10.weight", "features.17.weight",
        "features.24.weight", "features.31.weight"])
    args = ap.parse_args()
    c = zstd.ZstdCompressor(level=DEFAULT_LEVEL)
    dz = zstd.ZstdDecompressor()
    epochs = list(range(args.first, args.newest + 1))

    header(f"epochs {epochs[0]}..{epochs[-1]}, rebase every {args.interval}",
           f"zstd L{DEFAULT_LEVEL}, subtract + byte shuffle")

    grand = {"star": 0, "chain": 0, "raw": 0, "uncompressed": 0}
    print(f"  {'tensor':<24}{'raw':>12}{'star':>12}{'chain':>12}{'chain saves':>13}")
    for name in args.tensors:
        W = {}
        for e in epochs:
            with SafetensorsFile(path_for(args.checkpoints, e)) as f:
                s = f.spec(name)
                W[e] = f.rows(name, 0, s.num_rows).ravel().copy()
        nbytes = W[epochs[0]].nbytes
        star = chain = raw = 0
        for i, e in enumerate(epochs):
            is_hub = (i % args.interval) == 0
            raw += len(encode(W[e], None, c))
            if is_hub:
                star += len(encode(W[e], None, c))
                chain += len(encode(W[e], None, c))
            else:
                hub = epochs[i - (i % args.interval)]
                star += len(encode(W[e], W[hub], c))
                chain += len(encode(W[e], W[epochs[i-1]], c))
        tot = nbytes * len(epochs)
        for k, v in (("star", star), ("chain", chain), ("raw", raw),
                     ("uncompressed", tot)):
            grand[k] += v
        print(f"  {name:<24}{raw/tot*100:>11.2f}%{star/tot*100:>11.2f}%"
              f"{chain/tot*100:>11.2f}%{(star-chain)/tot*100:>12.2f}pp")

    tot = grand["uncompressed"]   # same denominator as the rows above
    print(f"\n  {'ALL TENSORS':<24}{grand['raw']/tot*100:>11.2f}%"
          f"{grand['star']/tot*100:>11.2f}%{grand['chain']/tot*100:>11.2f}%"
          f"{(grand['star']-grand['chain'])/tot*100:>12.2f}pp")

    # --- reconstruction cost -------------------------------------------------
    print("\n  Reconstruction: decodes needed for the WORST commit in a group")
    name = args.tensors[-1]
    with SafetensorsFile(path_for(args.checkpoints, epochs[0])) as f:
        s = f.spec(name)
    W = {}
    for e in epochs[:args.interval]:
        with SafetensorsFile(path_for(args.checkpoints, e)) as f:
            W[e] = f.rows(name, 0, s.num_rows).ravel().copy()
    hub = epochs[0]
    blobs_star = [encode(W[epochs[args.interval-1]], W[hub], c)]
    blobs_chain = [encode(W[epochs[i]], W[epochs[i-1]], c)
                   for i in range(1, args.interval)]

    def decode_time(blobs, reps=5):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            for b in blobs:
                plain_stream("delta-shuffle-zstd", b, decompressor=dz)
            best = min(best, time.perf_counter() - t0)
        return best

    ts, tc = decode_time(blobs_star), decode_time(blobs_chain)
    print(f"    {name}  ({s.nbytes/2**20:.1f} MiB)")
    print(f"    star   {len(blobs_star)} decode   {ts*1000:>7.1f} ms")
    print(f"    chain  {len(blobs_chain)} decodes  {tc*1000:>7.1f} ms   ({tc/ts:.2f}x)")


if __name__ == "__main__":
    main()
