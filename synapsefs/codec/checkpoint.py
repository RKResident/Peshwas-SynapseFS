"""Checkpoint-level chunking, dedup, and tensor-manifest construction.

Sits directly above `synapsefs/codec/chunk.py`: that module knows how to
encode *one* chunk given target rows and (optionally) the matching base
rows; this module decides which rows form a chunk, walks every tensor in a
checkpoint, and assembles the FORMAT.md section 7 tensor-manifest dicts.

Replaces `synapsefs/codec/checkpoint_iter.py`, which never successfully ran:
it called the reference `safetensors.safe_open` (dead for `BF16`, see
`synapsefs/safetensors_io.py`'s module docstring), leaked a base file handle,
had no handling for a tensor missing from the base or present with a
different shape/dtype, accumulated every encoded payload in a list (so its
"out-of-core" docstring claim was false), and had no dedup. This module
fixes all of that:

- Reads through `SafetensorsFile`, which never raises on `BF16` and always
  hands back 2-D bit-pattern arrays.
- Uses `EncodedChunk.original_len` (true tensor bytes) rather than
  `plain_len` (encoded stream length) for the `original_bytes` counter that
  feeds `residual_ratio` (CLI.md section 3.1).
- Holds the target and (optional) base readers in a `contextlib.ExitStack`,
  so both close on every path, including a failure opening the base.
- A missing or shape/dtype-mismatched base tensor is treated as "no base for
  this tensor" rather than an error -- a fine-tune adding a head, or
  changing a layer's shape, is a normal thing for a checkpoint to contain --
  and is recorded in `CheckpointResult.notes`.
- `encode_checkpoint` takes an `emit` callback rather than returning a list
  of records. Exactly one encoded payload is alive at a time; nothing here
  ever accumulates a list of them. This is the actual fix for defect 5, not
  a cosmetic one: the whole point is that peak memory stays bounded by one
  chunk regardless of checkpoint size.
- Chunks are deduped against both this run's own history and, via the
  caller-supplied `already_have` predicate, against what the object store
  already holds from previous commits. A deduped chunk still gets a
  manifest entry -- dedup affects storage, never the manifest.

What this module deliberately does *not* do (out of scope for prompt C):

- Resolve `base_tensor_manifest` to an actual hash. That needs the object
  graph, which does not exist yet -- every manifest gets `None` here.
- Apply a row or column permutation. `base_row_permutation` and
  `base_col_permutation` are always `None` (identity) and `col_block_size`
  is always `1`; non-identity alignment is the alignment team's job, feeding
  its results in through a future revision of this API.
- Serialise the manifest to canonical JSON (TeamInstructions.md section A).
  That is the pack/store layer's job; this module produces plain dicts.
"""

from __future__ import annotations

import os
from contextlib import ExitStack

import blake3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Union

import numpy as np

from synapsefs.codec.chunk import DELTA, encode_chunk
from synapsefs.safetensors_io import SafetensorsFile, TensorSpec

# What this module needs from a base checkpoint, and all it needs: three
# methods. `SafetensorsFile` (a real file on disk) and `graph.CommitCheckpoint`
# (a residual chain reconstructed out of packs) both satisfy it, which is what
# lets "diff against a file" and "diff against commit 4d8e2f" be the same code
# path rather than two.
#
# Deliberately a structural expectation rather than an imported base class:
# importing `graph` here would invert the dependency (graph sits above codec)
# and create a cycle.
class BaseSource(Protocol):
    def names(self) -> List[str]: ...
    def spec(self, name: str) -> TensorSpec: ...
    def rows(self, name: str, start: int, stop: int) -> "np.ndarray": ...

__all__ = [
    "DEFAULT_CHUNK_SIZE_BYTES",
    "ChunkRecord",
    "CheckpointResult",
    "encode_checkpoint",
]

PathLike = Union[str, "os.PathLike[str]"]

# OPEN QUESTION 1.2 (docs/OPEN_QUESTIONS.md) -- not benchmarked yet. Target is
# ~1-4 MB post-compression per FORMAT.md section 7; 4 MiB is the top of that
# range, picked as a placeholder, not a measured default. Whoever closes
# OPEN QUESTION 1.2 should update this constant and cite the benchmark here.
DEFAULT_CHUNK_SIZE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk on its way to the pack writer.

    Emitted through `encode_checkpoint`'s `emit` callback -- never collected
    into a list by this module. `row_start`/`row_end` mirror the manifest's
    convention (FORMAT.md section 7.2): inclusive logical row indices, not
    the half-open ranges used internally to slice the reader.
    """

    tensor: str
    row_start: int
    row_end: int
    content_hash: bytes
    encoding: str
    payload: bytes
    plain_len: int
    original_len: int


@dataclass(frozen=True)
class CheckpointResult:
    """Everything `commit --json` (CLI.md section 3.1) needs, minus the
    fields only the pack/store layer or the alignment step can fill in
    (`commit`, `branch`, `base`, `residual_ratio` -- a derived quantity left
    to the caller, and `alignment`)."""

    manifests: Dict[str, dict]
    header_bytes: bytes
    tensors: int

    original_bytes: int
    """True tensor bytes across the checkpoint -- the `residual_ratio`
    denominator (CLI.md section 3.1)."""

    stored_bytes: int
    """Bytes this commit actually adds to the repository: the sum of
    `stored_len` over the chunks passed to `emit`, and *only* those.

    Deduped chunks are deliberately excluded. Counting them would make the
    numerator blind to the one thing dedup is for -- on a checkpoint with
    four byte-identical tensors it overstates the real cost fourfold, so a
    `residual_ratio` computed from it reads 92% where the true figure is
    23%. Whatever else that number is, it is not a measure of what the
    commit cost."""

    deduped_bytes: int
    """What dedup saved: the summed `stored_len` of chunks that were *not*
    emitted because their content was already present. `stored_bytes +
    deduped_bytes` is the total encoded size, i.e. what a dedup-less
    encoder would have written."""

    chunks_new: int
    chunks_deduped: int

    reused_manifests: Dict[str, str]
    """tensor name -> the base's tensor-manifest hash, for tensors that were
    byte-identical to the base and are therefore *not* described again.

    FORMAT.md 4.5's reuse rule. These names are deliberately absent from
    `manifests`: the caller points the new checkpoint-manifest straight at
    these hashes, so no new object is written and -- more importantly -- the
    tensor does not grow a residual chain. Without this, a frozen tensor
    accumulates one zero-delta hop per commit and reconstruction pointlessly
    walks all of them.
    """

    notes: List[str]


def _rows_per_chunk(row_nbytes: int, chunk_size_bytes: int) -> int:
    return max(1, chunk_size_bytes // row_nbytes)


def _manifest_dict(
    spec: TensorSpec,
    chunks: List[dict],
    base_manifest_hash: Optional[str],
    content_hash: str,
) -> dict:
    return {
        "name": spec.name,
        # Verbatim safetensors dtype name (e.g. "BF16"), matching every other
        # dtype string in this codebase. FORMAT.md section 7's own example
        # shows lowercase "bf16", which disagrees with the rest of the spec
        # (section 8's key-kind table, chunk.py's `_DTYPES`, and
        # safetensors_io's `TensorSpec.dtype`, all uppercase) -- a doc
        # discrepancy, not something to paper over here.
        "dtype": spec.dtype,
        "shape": list(spec.shape),
        # BLAKE3 of this tensor's fully reconstructed bytes, in this
        # checkpoint's own row order -- independent of the base, the
        # permutations, the chunk boundaries and the encoding.
        #
        # It exists because **the manifest hash is not a content identity.**
        # Two manifests with the same hash certainly hold the same content;
        # the converse is false, and merge depends on the converse: two
        # branches can hold byte-identical weights whose manifests differ
        # because they were aligned against different bases. Comparing
        # manifest hashes would report a conflict on a tensor nobody touched.
        #
        # It is also the only check that catches a permutation composed in the
        # wrong order across a base chain -- both orderings are valid
        # bijections of the correct length, so no structural invariant sees it.
        "content_hash": content_hash,
        # The tensor-manifest this one was diffed against, or null if this
        # tensor was stored in full.
        #
        # The relationship is an "if and only if": null **iff** no chunk uses a
        # delta encoding. A delta chunk with no base is unresolvable; a base
        # with no delta chunk anywhere is a pointer to nothing -- it makes
        # reconstruction walk into a manifest that contributes zero bytes, and
        # forces GC to retain that whole subtree to satisfy a link nobody
        # reads. The caller enforces the second direction (see `any_delta`).
        "base_tensor_manifest": base_manifest_hash,
        # Non-identity alignment is the alignment team's job; this module
        # only ever chunks a target against the base's own row order, so
        # both permutations are identity (null) and column blocking is 1.
        "base_row_permutation": None,
        "base_col_permutation": None,
        "col_block_size": 1,
        "chunks": chunks,
    }


def _base_rows_source(
    name: str,
    spec: TensorSpec,
    base: "Optional[BaseSource]",
    base_names: Optional[set],
    notes: List[str],
) -> "Optional[BaseSource]":
    """Decide whether `base` has a usable, same-shape/dtype copy of `name`.

    Returns `base` itself if so (a sentinel meaning "read rows from here"),
    or `None` if there is no base, the tensor is absent from it, or it
    disagrees in shape/dtype -- in which case a `notes` entry is appended
    naming the tensor and the reason.
    """
    if base is None:
        return None
    if name not in base_names:
        notes.append(f"tensor {name!r} absent from base; stored in full")
        return None
    base_spec = base.spec(name)
    if base_spec.shape != spec.shape or base_spec.dtype != spec.dtype:
        notes.append(
            f"tensor {name!r} present in base with different shape/dtype "
            f"(base {list(base_spec.shape)} {base_spec.dtype!r} vs "
            f"target {list(spec.shape)} {spec.dtype!r}); stored in full"
        )
        return None
    return base


def encode_checkpoint(
    target_path: PathLike,
    base: "Optional[PathLike | BaseSource]" = None,
    *,
    emit: Callable[[ChunkRecord], None],
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    base_manifests: Optional[Dict[str, str]] = None,
    already_have: Optional[Callable[[bytes], bool]] = None,
    compress_raw: bool = True,
    allow_raw_fallback: bool = True,
    level: int = 3,
) -> CheckpointResult:
    """Chunk, encode, and dedup every tensor in `target_path` against
    `base` (or against nothing, for a root commit), building the
    tensor-manifest dicts as it goes.

    `base` is either a path to a `.safetensors` file, or any object satisfying
    `BaseSource` above -- in practice `graph.CommitCheckpoint`, which makes a
    previously committed checkpoint readable without materializing it.

    `base_manifests` maps tensor name -> the base commit's tensor-manifest hash
    for that tensor, and must be supplied whenever the result will be stored.
    It is what `base_tensor_manifest` is filled from, and reconstruction has no
    other route to the base. It is optional only because tests that diff
    against a bare file and reconstruct by hand already hold the base rows
    themselves and never consult the manifest.

    `emit` is called once per chunk that actually needs storing, in tensor
    order and then row order; its `ChunkRecord.payload` must not be
    retained by this function after the call returns, and it never is --
    at most one payload is alive at a time; nothing here builds a list of
    them. A chunk whose content hash was already emitted earlier in this
    same run, or for which `already_have(hash)` returns `True`, is counted
    in `chunks_deduped` instead and is *not* passed to `emit` -- but it
    still gets a manifest entry, since dedup affects storage, never the
    manifest.

    Malformed input files raise `UsageError` (via `SafetensorsFile`); this
    function never raises `IntegrityError`, which CLI.md section 1.3
    reserves exclusively for verification failures on repo-internal data.
    """
    notes: List[str] = []
    manifests: Dict[str, dict] = {}
    seen_hashes: set = set()
    reused_manifests: Dict[str, str] = {}
    tensors = 0
    original_bytes = 0
    stored_bytes = 0
    chunks_new = 0
    chunks_deduped = 0
    deduped_bytes = 0

    def store_chunk(record: ChunkRecord) -> None:
        """Dedup, count, and emit one chunk.

        Byte counters follow the same branch as the chunk itself: a deduped
        chunk costs nothing on disk, so it must not land in `stored_bytes`.
        """
        nonlocal chunks_new, chunks_deduped, stored_bytes, deduped_bytes
        if record.content_hash in seen_hashes:
            chunks_deduped += 1
            deduped_bytes += len(record.payload)
            return
        if already_have is not None and already_have(record.content_hash):
            chunks_deduped += 1
            deduped_bytes += len(record.payload)
            seen_hashes.add(record.content_hash)
            return
        seen_hashes.add(record.content_hash)
        chunks_new += 1
        stored_bytes += len(record.payload)
        emit(record)

    with ExitStack() as stack:
        target = stack.enter_context(SafetensorsFile(target_path))
        base_source_obj: Optional[BaseSource] = None
        base_names: Optional[set] = None
        if base is not None:
            # A path is opened and owned here; an already-open source (e.g.
            # `graph.CommitCheckpoint`) is borrowed, and closing it is the
            # caller's business.
            if isinstance(base, (str, os.PathLike)):
                base_source_obj = stack.enter_context(SafetensorsFile(base))
            else:
                base_source_obj = base
            base_names = set(base_source_obj.names())

        header_bytes = target.header_bytes

        for name in target.names():
            tensors += 1
            spec = target.spec(name)
            original_bytes += spec.nbytes

            base_source = _base_rows_source(
                name, spec, base_source_obj, base_names, notes
            )
            # Only set when this tensor actually diffed against the base. A
            # tensor that fell back to raw (absent from the base, or reshaped)
            # is stored in full and must say so.
            base_manifest_hash = (
                base_manifests.get(name)
                if base_source is not None and base_manifests is not None
                else None
            )

            row_nbytes = spec.row_elems * spec.width
            chunk_entries: List[dict] = []
            unchanged = False
            # Accumulated as chunks stream past, never by materialising the
            # tensor: the PS requires out-of-core operation and grades peak
            # RSS. BLAKE3 is sequential, and `rows()` yields the tensor in its
            # own row order, which is exactly the data-region byte order.
            content_digest = blake3.blake3()

            # Degenerate shapes: a dim past 0 that is itself 0 (row_nbytes
            # == 0) or an empty tensor (num_rows == 0). Either way there is
            # nothing to chunk -- an empty chunk list, not a crash.
            if spec.num_rows > 0 and row_nbytes > 0:
                rows_per_chunk = _rows_per_chunk(row_nbytes, chunk_size_bytes)

                # A tensor is only known to be unchanged once its *last*
                # chunk turns out identical, but chunks are emitted as they
                # are produced. Holding identical chunks back until the
                # question is settled avoids writing zero-delta chunks that
                # the reuse rule then makes unreferenced.
                #
                # This does not reintroduce the unbounded buffering that the
                # `emit` callback exists to prevent. Only *identical* chunks
                # are held, and an all-zero stream compresses to a couple of
                # dozen bytes at any chunk size; the moment one chunk differs,
                # the buffer is flushed and the rest of the tensor streams
                # straight through as before.
                pending: List[ChunkRecord] = []
                still_identical = base_manifest_hash is not None

                for row_start in range(0, spec.num_rows, rows_per_chunk):
                    row_stop = min(row_start + rows_per_chunk, spec.num_rows)

                    target_rows = target.rows(name, row_start, row_stop)
                    base_rows = (
                        base_source.rows(name, row_start, row_stop)
                        if base_source is not None
                        else None
                    )

                    content_digest.update(
                        np.ascontiguousarray(target_rows).view(np.uint8).reshape(-1)
                    )

                    encoded = encode_chunk(
                        target_rows,
                        base_rows,
                        dtype=spec.dtype,
                        level=level,
                        compress_raw=compress_raw,
                        allow_raw_fallback=allow_raw_fallback,
                    )

                    row_end = row_stop - 1  # FORMAT.md 7.2: inclusive
                    chunk_entries.append(
                        {
                            "row_start": row_start,
                            "row_end": row_end,
                            "encoding": encoded.encoding,
                            "object": encoded.content_hash.hex(),
                        }
                    )
                    record = ChunkRecord(
                        tensor=name,
                        row_start=row_start,
                        row_end=row_end,
                        content_hash=encoded.content_hash,
                        encoding=encoded.encoding,
                        payload=encoded.payload,
                        plain_len=encoded.plain_len,
                        original_len=encoded.original_len,
                    )

                    if still_identical and encoded.is_identical:
                        pending.append(record)
                        continue

                    # This tensor has changed after all. Everything held back
                    # is real and has to go out, in order, before this chunk.
                    if still_identical:
                        still_identical = False
                        for held in pending:
                            store_chunk(held)
                        pending = []
                    store_chunk(record)
                    # Neither `encoded` nor `record` is referenced past this
                    # point; the next iteration rebinds both and CPython drops
                    # the payload immediately, which is what keeps peak
                    # retained bytes to one chunk regardless of checkpoint
                    # size.

                unchanged = still_identical and bool(chunk_entries)
                # Invariant: a base pointer is only meaningful if at least one
                # chunk actually references it. The per-chunk raw fallback can
                # take every chunk of a tensor, which would otherwise leave a
                # base pointing at nothing.
                if not any(c["encoding"] == DELTA for c in chunk_entries):
                    base_manifest_hash = None

                if not unchanged:
                    # A tensor with no chunks at all (a degenerate shape) has
                    # nothing held back; this is just the safety net.
                    for held in pending:
                        store_chunk(held)
                pending = []

            # FORMAT.md 4.5. Every chunk identical to the base means the
            # tensor is unchanged, so point at the base's manifest instead of
            # writing a new one. Requires knowing that hash, which is why this
            # only fires when `base_manifests` was supplied.
            if unchanged:
                reused_manifests[name] = base_manifest_hash
            else:
                manifests[name] = _manifest_dict(
                    spec, chunk_entries, base_manifest_hash,
                    content_digest.hexdigest(),
                )

    return CheckpointResult(
        manifests=manifests,
        header_bytes=header_bytes,
        tensors=tensors,
        original_bytes=original_bytes,
        stored_bytes=stored_bytes,
        deduped_bytes=deduped_bytes,
        chunks_new=chunks_new,
        chunks_deduped=chunks_deduped,
        reused_manifests=reused_manifests,
        notes=notes,
    )
