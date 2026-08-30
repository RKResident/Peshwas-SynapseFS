"""Export one chunk per layer with every intermediate stage of the codec.

Produces, for each weight-bearing layer, the five representations a chunk
passes through, so the pipeline can be inspected or re-implemented against
real data rather than a description of it:

  1_stored.zst                 the object exactly as it sits in the store
  2_uncompressed_shuffled.bin  zstd removed; still byte-shuffled
  3_delta_unshuffled_u16.bin   un-shuffled residual, target - base mod 2**width
  4_base_weights.bin           the base rows this chunk was diffed against,
                               AFTER the alignment gather -- what the decoder
                               actually adds to
  5_target_weights.bin         the reconstructed rows, bit-identical to the
                               original checkpoint

Stage 4 is the one worth being careful about. `base_tensor_manifest` names the
base, but the bytes the encoder subtracted are the base rows gathered through
`base_row_permutation`, not the base rows in their own order. Exporting the
ungathered rows would give a file that looks plausible and does not satisfy
4 + 3 == 5. Every layer is checked against that identity before anything is
written.

HEAD is a FULL commit -- the star topology stores every Nth checkpoint whole --
so it has no base and no residual. The newest commit that has deltas is its
parent, and that is what this exports; `commit` in the metadata says which.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zstandard as zstd

from synapsefs.codec.chunk import dtype_spec, is_delta, plain_stream, unshuffle
from synapsefs.graph import CommitCheckpoint, get_json
from synapsefs.store.objectstore import ObjectStore

NP_OF = {2: np.uint16, 4: np.uint32, 8: np.uint64}
MIN_ELEMENTS = 3000


def newest_delta_commit(store, head: str, after_full: bool) -> tuple[str, str]:
    """Walk back to a commit that stores residuals.

    `after_full` additionally requires the parent to be a FULL commit, which
    picks the first residual after a rebase anchor. That is the gap-1 case:
    the base is the immediately preceding checkpoint, so the drift is one
    epoch rather than several and the chunk is the cheapest the star topology
    ever produces. Commits deeper into a rebase interval diff against the same
    anchor across a wider gap and compress measurably worse, so which one is
    exported changes the numbers.
    """
    h = head
    while True:
        c = get_json(store, h)
        parents = c.get("parents") or []
        if not parents:
            raise SystemExit("no commit in this lineage stores deltas")
        if not c.get("full"):
            if not after_full or get_json(store, parents[0]).get("full"):
                return h, c.get("message", "")
        h = parents[0]


def export(ckpt: CommitCheckpoint, store, name: str, out: Path,
           min_elements: int) -> dict:
    """Write one layer's chunk group. Returns its metadata."""
    mh = ckpt._tensor_manifests[name]
    man = get_json(store, mh)
    width, _ = dtype_spec(man["dtype"])
    shape = tuple(man["shape"])
    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    base_hash = man.get("base_tensor_manifest")

    # Take consecutive chunks until the element budget is met. Most layers hit
    # it with one; the small buffers never do and take everything they have.
    picked, elems = [], 0
    for ch in man["chunks"]:
        picked.append(ch)
        elems += (ch["row_end"] - ch["row_start"] + 1) * row_elems
        if elems >= min_elements:
            break

    D = zstd.ZstdDecompressor()
    stored, plain, delta, base, target = b"", b"", [], [], []
    for ch in picked:
        lo, hi = ch["row_start"], ch["row_end"] + 1
        blob = store.get(ch["object"])
        p = plain_stream(ch["encoding"], blob, decompressor=D)
        stored += blob
        plain += p
        delta.append(np.frombuffer(unshuffle(p, width), dtype=NP_OF[width]))
        target.append(ckpt.rows(name, lo, hi).ravel())
        if is_delta(ch["encoding"]) and base_hash:
            base.append(ckpt._aligned_base(man, base_hash, lo, hi).ravel())

    delta_a = np.concatenate(delta)
    target_a = np.concatenate(target)
    base_a = np.concatenate(base) if base else None

    # 4 + 3 == 5, in native width. If this does not hold the export is wrong.
    if base_a is not None:
        got = (base_a + delta_a).astype(NP_OF[width])
        if not np.array_equal(got, target_a):
            raise SystemExit(f"{name}: base + delta != target -- export is wrong")

    out.mkdir(parents=True, exist_ok=True)
    (out / "1_stored.zst").write_bytes(stored)
    (out / "2_uncompressed_shuffled.bin").write_bytes(plain)
    (out / "3_delta_unshuffled_u16.bin").write_bytes(delta_a.tobytes())
    (out / "5_target_weights.bin").write_bytes(target_a.tobytes())
    if base_a is not None:
        (out / "4_base_weights.bin").write_bytes(base_a.tobytes())

    meta = {
        "tensor": name,
        "dtype": man["dtype"],
        "shape": list(shape),
        "width_bytes": width,
        "chunks_exported": len(picked),
        "chunks_in_tensor": len(man["chunks"]),
        "rows": [picked[0]["row_start"], picked[-1]["row_end"]],
        "elements": int(target_a.size),
        "encoding": picked[0]["encoding"],
        "is_delta": bool(base_a is not None),
        "base_tensor_manifest": base_hash,
        "row_permuted": man.get("base_row_permutation") is not None,
        "stored_bytes": len(stored),
        "plain_bytes": len(plain),
        "ratio_pct": round(len(stored) / len(plain) * 100, 2) if plain else None,
        "verified_base_plus_delta_equals_target": base_a is not None,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("tools/tools/benchmark/.synapse"))
    ap.add_argument("--out", type=Path, default=Path("chunk_export/per_layer"))
    ap.add_argument("--min-elements", type=int, default=MIN_ELEMENTS)
    ap.add_argument("--commit", help="export this commit instead of searching")
    ap.add_argument("--any-delta", action="store_true",
                    help="take the newest residual commit rather than requiring "
                         "its parent to be a FULL anchor")
    args = ap.parse_args()

    store = ObjectStore(args.repo / "objects")
    head = (args.repo / "refs" / "heads" / "main").read_text().strip()
    if args.commit:
        commit = args.commit
        message = get_json(store, commit).get("message", "")
    else:
        commit, message = newest_delta_commit(store, head, not args.any_delta)
    parent = get_json(store, commit)["parents"][0]
    ckpt = CommitCheckpoint(store, commit)

    man = get_json(store, get_json(store, commit)["checkpoint_manifest"])
    names = list(man["tensors"])
    big = [n for n in names if n.endswith(".weight")
           and len(get_json(store, man["tensors"][n])["shape"]) >= 2]
    small = [n for n in names if n not in big]

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    print(f"{'layer':<24} {'elements':>12} {'stored':>11} {'plain':>11} {'ratio':>7}")
    for n in big:
        m = export(ckpt, store, n, args.out / n, args.min_elements)
        index.append(m)
        print(f"{n:<24} {m['elements']:>12,} {m['stored_bytes']:>11,} "
              f"{m['plain_bytes']:>11,} {m['ratio_pct']:>6.2f}%")

    # The 61 buffers that cannot reach the budget on their own, concatenated so
    # the export still covers every tensor in the commit.
    comb = args.out / "_small_tensors_combined"
    comb.mkdir(parents=True, exist_ok=True)
    parts, total = [], 0
    for n in small:
        sub = comb / n
        m = export(ckpt, store, n, sub, 1)
        parts.append(m)
        total += m["elements"]
    (comb / "metadata.json").write_text(json.dumps(
        {"note": "every tensor too small to reach the element budget alone; "
                 "each subdirectory holds the whole tensor",
         "tensors": len(parts), "total_elements": total,
         "members": parts}, indent=2))
    print(f"\n{len(parts)} small tensors bundled, {total:,} elements total")

    (args.out / "metadata.json").write_text(json.dumps({
        "model": "PlainCNN 92.42M params, CIFAR-100, Adam lr=1e-4, fp16",
        "repo": str(args.repo),
        "commit": commit,
        "message": message,
        "head": head,
        "base_commit": parent,
        "base_message": get_json(store, parent).get("message", ""),
        "base_is_full": bool(get_json(store, parent).get("full")),
        "why_not_head": "HEAD is a FULL commit (star topology stores every Nth "
                        "checkpoint whole), so it has no base and no residual. "
                        "This commit is the newest residual whose base is a FULL "
                        "anchor -- gap 1, the cheapest chunk the topology "
                        "produces.",
        "files": {
            "1_stored.zst": "the chunk exactly as stored in the object store",
            "2_uncompressed_shuffled.bin": "zstd removed; still byte-shuffled",
            "3_delta_unshuffled_u16.bin":
                "un-shuffled residual, uint16 LE, target - base mod 2**16",
            "4_base_weights.bin":
                "base rows AFTER the alignment gather -- what the decoder adds to",
            "5_target_weights.bin": "reconstructed rows, bit-identical to the original",
        },
        "identity": "(4 + 3) mod 2**width == 5, checked before writing",
        "layers": index,
    }, indent=2))
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
