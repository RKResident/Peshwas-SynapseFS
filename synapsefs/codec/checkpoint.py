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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from synapsefs.codec.chunk import encode_chunk
from synapsefs.safetensors_io import SafetensorsFile, TensorSpec

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
    notes: List[str]


def _rows_per_chunk(row_nbytes: int, chunk_size_bytes: int) -> int:
    return max(1, chunk_size_bytes // row_nbytes)


def _manifest_dict(spec: TensorSpec, chunks: List[dict]) -> dict:
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
        # Resolving this needs the object graph, which does not exist yet
        # (that's a future prompt's job). Always null for now.
        "base_tensor_manifest": None,
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
    base: Optional[SafetensorsFile],
    base_names: Optional[set],
    notes: List[str],
) -> Optional[SafetensorsFile]:
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
    base_path: Optional[PathLike] = None,
    *,
    emit: Callable[[ChunkRecord], None],
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    already_have: Optional[Callable[[bytes], bool]] = None,
    compress_raw: bool = True,
    allow_raw_fallback: bool = True,
    level: int = 3,
) -> CheckpointResult:
    """Chunk, encode, and dedup every tensor in `target_path` against
    `base_path` (or against nothing, for a root commit), building the
    tensor-manifest dicts as it goes.

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
    tensors = 0
    original_bytes = 0
    stored_bytes = 0
    chunks_new = 0
    chunks_deduped = 0
    deduped_bytes = 0

    with ExitStack() as stack:
        target = stack.enter_context(SafetensorsFile(target_path))
        base: Optional[SafetensorsFile] = None
        base_names: Optional[set] = None
        if base_path is not None:
            base = stack.enter_context(SafetensorsFile(base_path))
            base_names = set(base.names())

        header_bytes = target.header_bytes

        for name in target.names():
            tensors += 1
            spec = target.spec(name)
            original_bytes += spec.nbytes

            base_source = _base_rows_source(name, spec, base, base_names, notes)

            row_nbytes = spec.row_elems * spec.width
            chunk_entries: List[dict] = []

            # Degenerate shapes: a dim past 0 that is itself 0 (row_nbytes
            # == 0) or an empty tensor (num_rows == 0). Either way there is
            # nothing to chunk -- an empty chunk list, not a crash.
            if spec.num_rows > 0 and row_nbytes > 0:
                rows_per_chunk = _rows_per_chunk(row_nbytes, chunk_size_bytes)

                for row_start in range(0, spec.num_rows, rows_per_chunk):
                    row_stop = min(row_start + rows_per_chunk, spec.num_rows)

                    target_rows = target.rows(name, row_start, row_stop)
                    base_rows = (
                        base_source.rows(name, row_start, row_stop)
                        if base_source is not None
                        else None
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

                    # Byte counters follow the same branch as the chunk
                    # itself: a deduped chunk costs nothing on disk, so it
                    # must not land in `stored_bytes`.
                    if encoded.content_hash in seen_hashes:
                        chunks_deduped += 1
                        deduped_bytes += encoded.stored_len
                    elif already_have is not None and already_have(encoded.content_hash):
                        chunks_deduped += 1
                        deduped_bytes += encoded.stored_len
                        seen_hashes.add(encoded.content_hash)
                    else:
                        seen_hashes.add(encoded.content_hash)
                        chunks_new += 1
                        stored_bytes += encoded.stored_len
                        emit(
                            ChunkRecord(
                                tensor=name,
                                row_start=row_start,
                                row_end=row_end,
                                content_hash=encoded.content_hash,
                                encoding=encoded.encoding,
                                payload=encoded.payload,
                                plain_len=encoded.plain_len,
                                original_len=encoded.original_len,
                            )
                        )
                    # `encoded` (and its payload) is not referenced past this
                    # point; the next loop iteration rebinds the name, and
                    # CPython drops the refcount to zero immediately -- this
                    # is what keeps peak retained payload bytes to at most
                    # one chunk's worth, regardless of checkpoint size.

            manifests[name] = _manifest_dict(spec, chunk_entries)

    return CheckpointResult(
        manifests=manifests,
        header_bytes=header_bytes,
        tensors=tensors,
        original_bytes=original_bytes,
        stored_bytes=stored_bytes,
        deduped_bytes=deduped_bytes,
        chunks_new=chunks_new,
        chunks_deduped=chunks_deduped,
        notes=notes,
    )
