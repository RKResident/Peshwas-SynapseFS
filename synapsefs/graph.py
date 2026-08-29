"""The commit object graph: writing a checkpoint's objects, and reading one back.

Sits above `codec`, `pack`, and `store`, and is the layer `commit` and
(later) `checkout` / the FUSE mount both talk to. It owns two directions:

**Writing.** A committed checkpoint becomes four kinds of loose object plus a
pack of chunks (FORMAT.md section 4): a verbatim `header`, one
`tensor-manifest` per tensor, one `checkpoint-manifest`, and one `commit`.
Chunks live in the pack; everything else is small, few, and stays loose.

**Reading.** `CommitCheckpoint` turns a commit hash back into something that
behaves like a checkpoint file, implementing FORMAT.md section 10's
reconstruction path. This is what makes commit-against-commit possible at all:
`encode_checkpoint` needs to read the base's rows, and the base is normally not
a file on disk -- it is a residual chain in a pack.

The key design point is that `CommitCheckpoint` exposes exactly the three
methods `encode_checkpoint` uses on a base -- `names()`, `spec()`, `rows()` --
which are the same three `SafetensorsFile` exposes. Neither side needs to know
which it is holding. That is what lets one code path serve "diff against a file
on disk" and "diff against commit 4d8e2f" without a branch, and it is also the
interface `checkout` and the FUSE read path will use.

Nothing here materializes a whole checkpoint. `rows()` decodes only the chunks
overlapping the requested range, recursing into base tensor-manifests for only
those rows. The PS forbids pre-materializing on mount and benchmarks reads from
a cold cache, so the diff path is built the same way from the start rather than
being rewritten later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import blake3
import numpy as np

from synapsefs.codec.checkpoint import apply_col_perm
from synapsefs.codec.chunk import decode_chunk, dtype_spec, is_delta
from synapsefs.errors import IntegrityError, ObjectNotFoundError
from synapsefs.safetensors_io import TensorSpec
from synapsefs.store.objectstore import ObjectStore

# FORMAT.md 12A. Commits form a **star**, not a chain: every residual commit
# diffs directly against its group's full checkpoint, and every Nth commit
# becomes a new full checkpoint (a new hub).
#
# Reconstruction is therefore always one residual decode on top of one full
# checkpoint, whatever N is -- N bounds how far the group's data is allowed to
# drift from its hub, not how deep a walk gets. Measured at depth 3 on a
# 512x512 fp16 tensor, the star costs ~9% more storage at typical fine-tune
# drift and reconstructs 2.19x faster; the gap is wider still for the partial
# reads the FUSE path actually issues, which under a chain pull chunks at
# every level.
#
# A placeholder, not a measured optimum -- deliberately one constant so it can
# be swapped, and eventually replaced by a dynamic trigger keyed on
# residual-ratio degradation.
REBASE_INTERVAL = 4

__all__ = [
    "REBASE_INTERVAL",
    "commits_since_full",
    "nearest_full_ancestor",
    "canonical_json",
    "put_json",
    "get_json",
    "CheckpointObjects",
    "write_checkpoint_objects",
    "write_commit_object",
    "walk_first_parent",
    "checkpoint_sizes",
    "CommitCheckpoint",
]


def canonical_json(obj: Any) -> bytes:
    """Serialize exactly one way, always.

    TeamInstructions section A lists this as an invariant, and the reason is
    that these bytes get hashed: `{"a":1,"b":2}` and `{"b": 2, "a": 1}` are the
    same object and must not be two different object hashes, or dedup silently
    stops working across any two writers that disagree on key order or
    whitespace. Python's dict ordering is insertion order, which is exactly the
    kind of thing that varies between code paths.
    """
    return json.dumps(
        obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")


def put_json(store: ObjectStore, obj: Any) -> str:
    return store.put(canonical_json(obj))


def get_json(store: ObjectStore, object_hash: str) -> Any:
    return json.loads(store.get(object_hash).decode("utf-8"))


@dataclass(frozen=True)
class CheckpointObjects:
    """Hashes written for one checkpoint, before the commit object itself."""

    checkpoint_manifest: str
    header: str
    tensor_manifests: Dict[str, str]
    reused_tensor_manifests: int
    """How many tensor-manifests already existed and were referenced rather
    than rewritten -- FORMAT.md section 4.5's reuse rule. Reported so the
    saving is visible instead of invisible."""


def write_checkpoint_objects(
    store: ObjectStore,
    *,
    header_bytes: bytes,
    manifests: Dict[str, dict],
    topology_config_hash: Optional[str],
    reused_manifests: Optional[Dict[str, str]] = None,
) -> CheckpointObjects:
    """Write the `header`, `tensor-manifest`s and `checkpoint-manifest`.

    FORMAT.md 4.5's reuse rule falls out of content addressing rather than
    needing to be implemented: a tensor whose manifest is byte-identical to a
    previous commit's hashes to the same object, so `store.put` is a no-op and
    the same hash is referenced again. `reused_tensor_manifests` counts those,
    which is the only reason `has` is checked first.
    """
    header_hash = store.put(header_bytes)

    tensor_manifests: Dict[str, str] = {}
    reused = 0
    for name, manifest in manifests.items():
        blob = canonical_json(manifest)
        # `store.put` already dedups; hashing here as well is purely so the
        # saving can be *counted*. Same algorithm as ObjectStore.put, which is
        # a duplication worth noting -- if that ever changes, this counter
        # silently reports zero reuse rather than breaking.
        object_hash = blake3.blake3(blob).hexdigest()
        if store.has(object_hash):
            reused += 1
        tensor_manifests[name] = store.put(blob)

    # FORMAT.md 4.5: unchanged tensors point straight at the base's
    # tensor-manifest. No object is written for them and they appear here
    # exactly as they appeared in the base commit.
    for name, manifest_hash in (reused_manifests or {}).items():
        tensor_manifests[name] = manifest_hash
        reused += 1

    checkpoint_manifest = {
        "header_object": header_hash,
        "tensors": tensor_manifests,
        "topology_config_hash": topology_config_hash,
    }
    return CheckpointObjects(
        checkpoint_manifest=put_json(store, checkpoint_manifest),
        header=header_hash,
        tensor_manifests=tensor_manifests,
        reused_tensor_manifests=reused,
    )


def write_commit_object(
    store: ObjectStore,
    *,
    checkpoint_manifest: str,
    parents: List[str],
    message: str,
    timestamp: str,
    full: bool,
    checkpoint_name: str,
) -> str:
    """Write a `commit` object (FORMAT.md 4.6) and return its hash.

    `full` is an extension to that section's schema: it records whether this
    commit stored its checkpoint in full rather than as a residual. Without it,
    deciding whether the re-basing interval (FORMAT.md 12A, every 4th commit)
    is due would mean loading every tensor-manifest of every ancestor to see
    whether they all have a null base -- an O(tensors x depth) walk to answer a
    one-bit question. See FORMAT.md 4.6 for the field.

    `checkpoint_name` is the second extension: the basename of the file that
    was committed. `checkout` without `--out` has to write the checkpoint back
    into the working tree "under the filename recorded in the commit"
    (CLI.md ~4), and there is nowhere else to record it -- the safetensors
    header names tensors, not the file. Storing the *basename* only is
    deliberate: the committer's absolute path is their business, and a commit
    that could steer a later checkout into writing outside the repo root would
    be a path-traversal bug waiting to happen.
    """
    return put_json(
        store,
        {
            "checkpoint_manifest": checkpoint_manifest,
            "parents": list(parents),
            "timestamp": timestamp,
            "message": message,
            "full": full,
            "checkpoint_name": checkpoint_name,
        },
    )


def commits_since_full(store: ObjectStore, commit_hash: Optional[str]) -> int:
    """How many commits have accumulated since the nearest full checkpoint.

    0 means `commit_hash` is itself full (or there is no commit at all). This
    is the number FORMAT.md 12A's `N` bounds: it is *not* a reconstruction
    depth, because every residual commit diffs directly against its baseline
    rather than against its predecessor. It measures how far the star has
    spread, i.e. how much drift has piled up since the last anchor.

    Follows first parents only. A merge commit's other parents are a separate
    lineage whose spread does not bound this one's.
    """
    depth = 0
    current = commit_hash
    while current is not None:
        commit = get_json(store, current)
        if commit.get("full", False) or not commit.get("parents"):
            return depth
        depth += 1
        current = commit["parents"][0]
    return depth


def nearest_full_ancestor(store: ObjectStore, commit_hash: Optional[str]) -> Optional[str]:
    """The commit that `commit_hash`'s lineage is anchored on -- itself if it
    is full, otherwise the nearest full commit reachable through first parents.

    This is the star's hub. Every residual commit diffs against it directly, so
    reconstructing any commit is one residual decode on top of one full
    checkpoint, regardless of how many commits sit between them.

    Returns None only when there is no commit at all (unborn HEAD). A lineage
    always terminates at a root, and a root is always full, so a walk that
    starts from a real commit always finds an anchor.
    """
    current = commit_hash
    while current is not None:
        commit = get_json(store, current)
        if commit.get("full", False) or not commit.get("parents"):
            return current
        current = commit["parents"][0]
    return None


def walk_first_parent(
    store: ObjectStore, commit_hash: Optional[str], *, limit: Optional[int] = None
) -> List[Tuple[str, dict]]:
    """`(hash, commit object)` from `commit_hash` back to the root, newest first.

    First-parent only, which is what CLI.md ~6 makes `log`'s default: a merge's
    second parent is a different lineage, and splicing it into a linear listing
    would interleave two histories with no way to tell them apart. `--graph`
    renders the extra parents from each commit's own `parents` list instead, so
    the walk itself never needs a second mode.

    `limit` stops the walk early rather than truncating afterwards -- `log -n 3`
    on a thousand-commit repo should read three objects, not a thousand.
    """
    out: List[Tuple[str, dict]] = []
    current = commit_hash
    while current is not None and (limit is None or len(out) < limit):
        commit = get_json(store, current)
        out.append((current, commit))
        parents = commit.get("parents") or []
        current = parents[0] if parents else None
    return out


def checkpoint_sizes(store: ObjectStore, commit_hash: str) -> dict:
    """`{"original_bytes", "stored_bytes", "tensors", "chunks"}` for one commit.

    Neither number is recorded in the commit object, deliberately: both are
    *derived* from objects it already points at, and baking a cached summary
    into an immutable content-addressed object would fork the commit identity
    on a semantically-null repack.

    `stored_bytes` counts each distinct chunk **once**, even when several
    tensors or commits reference it -- the honest reading of "what this commit
    costs on disk", since a deduped chunk was paid for by whoever wrote it
    first. It comes from `stat()` now that chunks are loose files; there is no
    index to consult.
    """
    commit = get_json(store, commit_hash)
    manifest = get_json(store, commit["checkpoint_manifest"])

    original = 0
    stored = 0
    seen: set = set()
    for manifest_hash in manifest["tensors"].values():
        tensor = get_json(store, manifest_hash)
        width, _kind = dtype_spec(tensor["dtype"])
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        original += width * count

        for chunk in tensor["chunks"]:
            object_hex = chunk["object"]
            if object_hex in seen:
                continue
            seen.add(object_hex)
            path = store.path_for(object_hex)
            if path.is_file():
                stored += path.stat().st_size

    return {
        "original_bytes": original,
        "stored_bytes": stored,
        "tensors": len(manifest["tensors"]),
        "chunks": len(seen),
    }


class CommitCheckpoint:
    """A committed checkpoint, readable as though it were a `.safetensors` file.

    Implements `names()`, `spec()` and `rows()` with the same meaning
    `SafetensorsFile` gives them, so it can be handed to
    `encode_checkpoint(base_source=...)` interchangeably with a real file.

    `rows()` follows FORMAT.md section 10: select only the chunks overlapping
    the requested rows, fetch each from the pack set, and -- when a chunk is a
    residual -- recurse into the base tensor-manifest for only those same rows.
    Nothing larger than the requested span is ever decoded.
    """

    def __init__(self, store: ObjectStore, commit_hash: str):
        self.store = store
        self.commit_hash = commit_hash

        self.commit = get_json(store, commit_hash)
        self.manifest = get_json(store, self.commit["checkpoint_manifest"])
        self._tensor_manifests: Dict[str, str] = self.manifest["tensors"]
        self._cache: Dict[str, dict] = {}

    # -- the SafetensorsFile-shaped surface --------------------------------

    def names(self) -> List[str]:
        return list(self._tensor_manifests)

    def refs(self):
        """`{name: TensorSpec}`. The alignment solver probes for this to learn
        the base's shapes; `TensorSpec` carries `.shape`, which is all it
        reads."""
        return {name: self.spec(name) for name in self.names()}

    def tensor_manifest_hashes(self) -> Dict[str, str]:
        """tensor name -> its tensor-manifest hash in this commit.

        Pass to `encode_checkpoint(base_manifests=...)`: a residual chunk is
        only readable if its manifest records which manifest it was diffed
        against, and this is where those hashes come from.
        """
        return dict(self._tensor_manifests)

    def spec(self, name: str) -> TensorSpec:
        manifest = self._manifest_for(name)
        shape = tuple(manifest["shape"])
        width, _kind = dtype_spec(manifest["dtype"])
        num_rows = shape[0] if shape else 1
        row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        return TensorSpec(
            name=name,
            dtype=manifest["dtype"],
            shape=shape,
            width=width,
            num_rows=num_rows,
            row_elems=row_elems,
            nbytes=width * int(np.prod(shape)) if shape else width,
        )

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        spec = self.spec(name)
        if not (0 <= start <= stop <= spec.num_rows):
            raise IntegrityError(
                f"{name}: row range [{start}, {stop}) out of bounds for "
                f"{spec.num_rows} rows"
            )
        flat = self._rows_from_manifest(self._tensor_manifests[name], start, stop)
        return flat.reshape(stop - start, spec.row_elems)

    def gather_rows(self, name: str, indices) -> np.ndarray:
        """Rows at arbitrary `indices`, in the order given.

        The read primitive a **row permutation** needs. `rows(start, stop)`
        cannot serve it: under a permutation `p`, target rows `[lo, hi)` are
        built from base rows `p[lo:hi]`, which is a scattered set, not a range.

        Each chunk is decoded at most once and its rows scattered into place,
        rather than decoding the whole tensor -- so a gather touching one
        chunk costs one chunk. Worst case (indices spread over every chunk) it
        degrades to a full tensor decode, which is inherent to permutation and
        not something a chunk layout can avoid.
        """
        spec = self.spec(name)
        idx = np.asarray(indices, dtype=np.intp)
        if idx.size and (idx.min() < 0 or idx.max() >= spec.num_rows):
            raise IntegrityError(
                f"{name}: gather index out of bounds for {spec.num_rows} rows"
            )
        return self._gather_from_manifest(
            self._tensor_manifests[name], idx, spec.row_elems
        )

    def as_float(self, name: str) -> np.ndarray:
        """The whole tensor as an owned 2-D float32 array.

        For the alignment solver, which scores whole layers against each other
        and cannot work on raw bit patterns. bf16 is widened by shifting into
        the high half of a float32 -- exactly the bits the truncation that
        produced it discarded -- because numpy has no bf16 dtype.

        This is the one method here that materialises a whole tensor. That is
        the alignment engine's memory model, not this class's: it streams one
        *layer pair* at a time (PS module 1a requires out-of-core alignment),
        which bounds the cost to the largest single tensor rather than the
        checkpoint.
        """
        spec = self.spec(name)
        raw = self.rows(name, 0, spec.num_rows).reshape(-1)
        if spec.dtype == "BF16":
            wide = raw.astype(np.uint32)
            np.left_shift(wide, 16, out=wide)
            out = wide.view(np.float32)
        elif spec.dtype == "F16":
            out = raw.view(np.float16).astype(np.float32)
        elif spec.dtype == "F32":
            out = raw.view(np.float32)
        else:
            out = raw.astype(np.float32)
        return np.ascontiguousarray(out).reshape(spec.num_rows, spec.row_elems)

    def _gather_from_manifest(
        self, manifest_hash: str, idx: np.ndarray, row_elems: int
    ) -> np.ndarray:
        manifest = self._load(manifest_hash)
        dtype = manifest["dtype"]
        width, _ = dtype_spec(dtype)
        out = np.empty(
            (len(idx), row_elems), dtype={2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        )
        remaining = len(idx)
        for chunk in manifest["chunks"]:
            lo = chunk["row_start"]
            hi = chunk["row_end"] + 1
            mask = (idx >= lo) & (idx < hi)
            if not mask.any():
                continue
            block = self._rows_from_manifest(manifest_hash, lo, hi).reshape(
                hi - lo, row_elems
            )
            out[mask] = block[idx[mask] - lo]
            remaining -= int(mask.sum())
        if remaining:
            raise IntegrityError(
                f"gather covered only {len(idx) - remaining} of {len(idx)} rows; "
                f"the manifest's chunks do not tile the tensor"
            )
        return out

    @property
    def header_bytes(self) -> bytes:
        """The verbatim source header, for byte-exact reconstruction."""
        return self.store.get(self.manifest["header_object"])

    # -- reconstruction ----------------------------------------------------

    def _manifest_for(self, name: str) -> dict:
        try:
            return self._load(self._tensor_manifests[name])
        except KeyError:
            raise IntegrityError(
                f"commit {self.commit_hash[:8]} has no tensor {name!r}"
            ) from None

    def _load(self, object_hash: str) -> dict:
        cached = self._cache.get(object_hash)
        if cached is None:
            cached = get_json(self.store, object_hash)
            self._cache[object_hash] = cached
        return cached

    def _permutation(self, object_hash: Optional[str]) -> "Optional[np.ndarray]":
        """Read a stored permutation object: packed int32 LE, no header."""
        if object_hash is None:
            return None
        blob = self.store.get(object_hash)
        if len(blob) % 4:
            raise IntegrityError(
                f"permutation {object_hash[:16]}: {len(blob)} bytes is not a "
                f"whole number of int32"
            )
        perm = np.frombuffer(blob, dtype="<i4")
        # A non-bijection would silently drop or duplicate rows rather than
        # fail, so it is checked on the way in.
        if not np.array_equal(np.sort(perm), np.arange(perm.size)):
            raise IntegrityError(
                f"permutation {object_hash[:16]} is not a bijection"
            )
        return perm

    def _aligned_base(
        self, manifest: dict, base_hash: str, lo: int, hi: int
    ) -> np.ndarray:
        """The base rows this chunk was diffed against.

        **Applies the same gather the encoder applied -- never the inverse.**
        `base_row_permutation[i]` is the base index that target index `i` came
        from, so target rows `[lo, hi)` need base rows `perm[lo:hi]`. Inverting
        here produces a valid bijection of the right length that reconstructs
        scrambled weights, and no structural check would notice; only the
        tensor `content_hash` (verify --content) catches it.
        """
        row_perm = self._permutation(manifest.get("base_row_permutation"))
        col_perm = self._permutation(manifest.get("base_col_permutation"))

        if row_perm is None:
            block = self._rows_from_manifest(base_hash, lo, hi)
            row_elems = block.size // max(1, hi - lo)
            block = block.reshape(hi - lo, row_elems)
        else:
            base_manifest = self._load(base_hash)
            shape = tuple(base_manifest["shape"])
            row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            block = self._gather_from_manifest(
                base_hash, row_perm[lo:hi].astype(np.intp), row_elems
            )

        if col_perm is not None:
            block = apply_col_perm(
                block, col_perm, manifest.get("col_block_size", 1)
            )
        return block

    def _rows_from_manifest(self, manifest_hash: str, start: int, stop: int) -> np.ndarray:
        """Decode rows `[start, stop)` of one tensor-manifest, recursing into
        its base for residual chunks.

        Chunks are the unit of decoding, so the covered span is generally wider
        than the request; it is sliced back down at the end. Chunks that do not
        overlap are never fetched, which is the property the FUSE read path
        depends on.
        """
        manifest = self._load(manifest_hash)
        dtype = manifest["dtype"]
        base_hash = manifest.get("base_tensor_manifest")

        pieces: List[np.ndarray] = []
        covered_start: Optional[int] = None
        covered_stop = 0

        for chunk in manifest["chunks"]:
            lo = chunk["row_start"]
            hi = chunk["row_end"] + 1          # FORMAT.md 7.2 stores it inclusive
            if hi <= start or lo >= stop:
                continue

            try:
                payload = self.store.get(chunk["object"])
            except ObjectNotFoundError:
                raise IntegrityError(
                    f"chunk {chunk['object'][:16]} referenced by a manifest is "
                    f"not in the object store"
                ) from None

            base_rows = None
            if is_delta(chunk["encoding"]):
                if base_hash is None:
                    raise IntegrityError(
                        f"chunk {chunk['object'][:16]} is a residual but its "
                        f"tensor-manifest has no base"
                    )
                base_rows = self._aligned_base(manifest, base_hash, lo, hi)

            pieces.append(decode_chunk(chunk["encoding"], payload, base_rows, dtype=dtype))
            if covered_start is None:
                covered_start = lo
            covered_stop = hi

        if covered_start is None:
            width, _ = dtype_spec(dtype)
            return np.frombuffer(b"", dtype={2: np.uint16, 4: np.uint32, 8: np.uint64}[width])

        joined = np.concatenate(pieces)
        row_elems = len(joined) // (covered_stop - covered_start)
        lo_off = (start - covered_start) * row_elems
        hi_off = (stop - covered_start) * row_elems
        return joined[lo_off:hi_off]
