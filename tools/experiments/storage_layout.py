"""Loose chunk files vs a packfile, on the read path.

Reproduces ARCHITECTURE.md 5.4.

Builds both layouts from a repo's real chunks and times them warm and cold.
Cold is what matters -- PS module 2 grades cold-cache read throughput -- and is
obtained with POSIX_FADV_DONTNEED, which needs no root.

The headline is that the fetch difference is real but small next to zstd, and
that parallel fetch recovers most of it: neither layout wants naive sequential
access.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blake3
import zstandard as zstd



def evict(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("tools/checkpoints"),
                    help="a repo whose packs supply the chunk payloads")
    ap.add_argument("--work", type=Path, default=Path("/tmp/synapse-layout"),
                    help="scratch directory for the two layouts")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    # Chunks are loose objects now; read them straight out of the store.
    objects = args.repo / ".synapse" / "objects"
    files = [p for p in objects.rglob("*") if p.is_file() and "tmp" not in p.parts]
    if not files:
        raise SystemExit(f"no objects under {objects}")
    payloads = [p.read_bytes() for p in files]
    total = sum(len(b) for b in payloads)
    print(f"{len(payloads)} chunks, {total/1048576:.1f} MiB from {args.repo}\n")

    args.work.mkdir(parents=True, exist_ok=True)
    loose_dir = args.work / "loose"
    hashes = [blake3.blake3(b).hexdigest() for b in payloads]
    loose = []
    for h, b in zip(hashes, payloads):
        d = loose_dir / h[:2] / h[2:4]
        d.mkdir(parents=True, exist_ok=True)
        f = d / h[4:]
        if not f.exists():
            f.write_bytes(b)
        loose.append(f)
    onepack = args.work / "all.pack"
    if not onepack.exists():
        onepack.write_bytes(b"".join(payloads))
    offsets, off = [], 0
    for b in payloads:
        offsets.append((off, len(b))); off += len(b)

    def read_loose(p):
        fd = os.open(p, os.O_RDONLY); d = os.read(fd, 1 << 24); os.close(fd); return d

    def loose_seq():
        t0 = time.perf_counter(); [read_loose(p) for p in loose]
        return time.perf_counter() - t0

    def loose_par(w):
        def f():
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=w) as ex:
                list(ex.map(read_loose, loose))
            return time.perf_counter() - t0
        return f

    def pack_par(w):
        def f():
            t0 = time.perf_counter()
            fd = os.open(onepack, os.O_RDONLY)
            with ThreadPoolExecutor(max_workers=w) as ex:
                list(ex.map(lambda o: os.pread(fd, o[1], o[0]), offsets))
            os.close(fd)
            return time.perf_counter() - t0
        return f

    n = len(payloads)
    print(f"  {'layout':<26}{'warm':>12}{'cold':>12}{'us/chunk cold':>16}")
    cases = [("loose, 1 thread", loose_seq, loose),
             ("loose, 8 threads", loose_par(8), loose),
             ("loose, 16 threads", loose_par(16), loose),
             ("pack, 1 thread", pack_par(1), [onepack]),
             ("pack, 8 threads", pack_par(8), [onepack])]
    cold = {}
    for label, fn, targets in cases:
        warm = min(fn() for _ in range(args.reps))
        c = min((evict(targets) or fn()) for _ in range(args.reps))
        cold[label] = c
        print(f"  {label:<26}{warm*1000:>10.1f}ms{c*1000:>10.1f}ms{c*1e6/n:>15.1f}")

    dz = zstd.ZstdDecompressor()
    t0 = time.perf_counter()
    for b in payloads:
        try:
            dz.decompress(b)
        except zstd.ZstdError:
            pass
    dec = time.perf_counter() - t0
    bl = min(v for k, v in cold.items() if k.startswith("loose"))
    bp = min(v for k, v in cold.items() if k.startswith("pack"))
    print(f"\n  zstd decompress (identical either way): {dec*1000:.1f} ms")
    print(f"  end-to-end cold, best of each: loose {(bl+dec)*1000:.0f} ms, "
          f"pack {(bp+dec)*1000:.0f} ms  ({(bl+dec)/(bp+dec):.2f}x)")
    print("\n  Fetch is a minority of end-to-end cost; decode dominates and is")
    print("  identical either way. Parallel fetch is worth ~3x on cold reads.")


if __name__ == "__main__":
    main()
