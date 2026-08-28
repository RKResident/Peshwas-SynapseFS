"""Reconstruction: a commit back into a real `.safetensors` file, byte-exact.

This is the read-side mirror of `codec/checkpoint.py`. That module takes a
file and turns it into (header object, tensor-manifests, chunks); this one
takes those back and writes the file again. CLI.md section 4 states the
requirement precisely: *"The output of `checkout --out` must be
byte-identical to reading the same commit through the mount."* Byte-identical
is the bar, not numerically-close -- so nothing here reformats, re-serializes
or normalizes anything.

Two properties make byte-exactness achievable rather than aspirational:

1. The header is stored **verbatim** as its own object, prefix and all. We
   never re-serialize the JSON, so key order, whitespace, `__metadata__` and
   any alignment padding the producer chose all survive untouched. A
   re-serialized header would be semantically identical and byte-different,
   which is exactly the failure this avoids.
2. The residual codec is *lossless integer* arithmetic (FORMAT.md section 8),
   so a decoded chunk is bit-equal to what was encoded -- not equal to within
   an epsilon. `compare_sources` below exists to let a human confirm that on
   their own checkpoints rather than take it on faith.

Nothing here materializes a whole checkpoint in memory. Tensors are streamed
row-batch by row-batch straight into the output file, so peak RSS is bounded
by `batch_bytes` regardless of model size -- the same constraint the FUSE read
path works under, and one the PS grades.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np

from synapsefs.codec.chunk import FLOAT, SINT, dtype_spec, to_monotone_key
from synapsefs.errors import IntegrityError
from synapsefs.safetensors_io import TensorSpec
from synapsefs.store.atomic import atomic_writer

__all__ = [
    "DEFAULT_BATCH_BYTES",
    "CheckpointSource",
    "header_layout",
    "materialize",
    "TensorComparison",
    "compare_sources",
]

# How much of one tensor to hold at a time. 8 MiB is two chunks' worth at the
# codec's 4 MiB default, so a batch never straddles more chunk boundaries than
# it has to while still keeping the write() count low. Not a measured optimum.
DEFAULT_BATCH_BYTES = 8 * 1024 * 1024

_HEADER_LEN_STRUCT = struct.Struct("<Q")


class CheckpointSource(Protocol):
    """The three-method surface `SafetensorsFile` and `CommitCheckpoint` share.

    Everything in this module is written against this protocol, never against
    either concrete class, which is what lets `compare_sources` diff a commit
    against a file, a commit against a commit, or a file against a file with
    one implementation. See `graph.CommitCheckpoint`'s docstring for why the
    two sides were built to the same shape in the first place.
    """

    def names(self) -> List[str]: ...
    def spec(self, name: str) -> TensorSpec: ...
    def rows(self, name: str, start: int, stop: int) -> np.ndarray: ...


# -- header layout ---------------------------------------------------------


def header_layout(header_bytes: bytes) -> List[Tuple[str, int, int]]:
    """`(name, begin, end)` for every tensor, sorted by position in the data
    section -- i.e. the order the bytes must be written back in.

    The safetensors header is a JSON object whose *key order need not match
    its `data_offsets` order*. Writing tensors in `names()` order would
    therefore produce a file whose header says one thing and whose bytes say
    another: still parseable, still numerically correct, and not
    byte-identical. Sorting by `begin` here is what makes the output
    reproducible rather than dependent on dict iteration order.

    `__metadata__` is skipped: it is the one reserved key that carries no
    tensor and has no `data_offsets`.
    """
    (header_len,) = _HEADER_LEN_STRUCT.unpack_from(header_bytes, 0)
    doc = json.loads(header_bytes[8 : 8 + header_len].decode("utf-8"))

    layout: List[Tuple[str, int, int]] = []
    for name, entry in doc.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        layout.append((name, int(begin), int(end)))
    layout.sort(key=lambda item: item[1])
    return layout


def _check_tiling(layout: List[Tuple[str, int, int]]) -> int:
    """Confirm the tensors tile the data section with no gaps, and return its
    total length.

    A gap would be bytes that exist in the source file, are described by no
    tensor, and are therefore *not stored anywhere* by the codec -- it only
    ever ingests tensor rows. Reconstructing such a file byte-exactly is
    impossible, so this refuses loudly instead of silently emitting zeros in
    the hole and calling the result identical.

    Raises IntegrityError, which is correct here rather than an over-reach of
    the reserved verification code: a checkpoint-manifest whose header
    describes bytes its tensor-manifests cannot produce *is* an inconsistent
    object graph.
    """
    cursor = 0
    for name, begin, end in layout:
        if begin != cursor:
            raise IntegrityError(
                f"header data section is not contiguous: {name!r} starts at "
                f"{begin}, expected {cursor}; this checkpoint cannot be "
                f"reconstructed byte-exactly"
            )
        if end < begin:
            raise IntegrityError(f"{name!r}: data_offsets end {end} < begin {begin}")
        cursor = end
    return cursor


# -- streaming a tensor ----------------------------------------------------


def _rows_per_batch(spec: TensorSpec, batch_bytes: int) -> int:
    row_bytes = spec.row_elems * spec.width
    if row_bytes <= 0:
        return 1
    return max(1, batch_bytes // row_bytes)


def iter_tensor_bytes(
    source: CheckpointSource, spec: TensorSpec, *, batch_bytes: int
) -> Iterator[np.ndarray]:
    """Yield `spec`'s raw little-endian bytes in batches, in row order.

    `rows()` hands back a 2-D *unsigned integer* view of the raw bit patterns
    (never a float array), so viewing it as uint8 is a reinterpretation, not a
    conversion -- no rounding happens anywhere on this path. `ascontiguousarray`
    is a no-op for the arrays either source actually returns; it is here so
    that a future source returning a strided view cannot silently write
    garbage.
    """
    step = _rows_per_batch(spec, batch_bytes)
    for start in range(0, spec.num_rows, step):
        stop = min(start + step, spec.num_rows)
        block = np.ascontiguousarray(source.rows(spec.name, start, stop))
        yield block.view(np.uint8).reshape(-1)


# -- materialize -----------------------------------------------------------


def materialize(
    source: CheckpointSource,
    header_bytes: bytes,
    out_path: Path,
    *,
    tmp_dir: Optional[Path] = None,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> dict:
    """Write `source` out as a `.safetensors` file at `out_path`.

    Goes through `atomic_writer`, so a crash or a decode failure partway
    through leaves either no file or the previous one -- never a truncated
    checkpoint that a later `checkout` would happily hand to torch. Note the
    default `tmp_dir`: the staging file must sit on the *destination's*
    filesystem, not the repo's, because `--out /tmp/x.safetensors` on a
    different device would make the final rename fail with EXDEV.

    Returns a stats dict for the CLI to render.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tmp_dir) if tmp_dir is not None else out_path.parent

    layout = header_layout(header_bytes)
    data_bytes = _check_tiling(layout)

    written = 0
    with atomic_writer(out_path, tmp_dir=staging) as handle:
        handle.write(header_bytes)
        for name, begin, end in layout:
            spec = source.spec(name)
            declared = end - begin
            if declared != spec.nbytes:
                raise IntegrityError(
                    f"{name!r}: header claims {declared} bytes, tensor-manifest "
                    f"describes {spec.nbytes}"
                )
            produced = 0
            for block in iter_tensor_bytes(source, spec, batch_bytes=batch_bytes):
                handle.write(block)
                produced += block.nbytes
            if produced != declared:
                # Reaching here means the manifest's chunk rows do not cover
                # the tensor. Better to fail than to emit a short file whose
                # header promises more.
                raise IntegrityError(
                    f"{name!r}: reconstructed {produced} bytes, header declares "
                    f"{declared}"
                )
            written += produced

    return {
        "path": str(out_path),
        "tensors": len(layout),
        "header_bytes": len(header_bytes),
        "data_bytes": data_bytes,
        "total_bytes": len(header_bytes) + written,
    }


# -- comparison ------------------------------------------------------------


@dataclass(frozen=True)
class TensorComparison:
    """How one tensor in two checkpoints relates.

    `max_ulp_diff` is the interesting column for a lossless-codec check and
    deserves an explanation. It reuses `to_monotone_key` from the codec: that
    map turns a float's bit pattern into an integer that sorts in the same
    order as the float, so the *integer* distance between two keys is exactly
    the number of representable values between them -- the ULP distance. It is
    the scale-free way to say "these differ by one bit in the mantissa"
    without picking an epsilon that means different things at 1e-8 and 1e+8.

    For a correct reconstruction every one of these is zero, and that is the
    point: the codec's claim is bit-equality, so a report of "max abs diff
    1e-7" would already be a bug rather than a rounding artifact.
    """

    name: str
    status: str
    """`identical`, `differs`, `dtype_mismatch`, `shape_mismatch`, `missing_left`
    or `missing_right`."""
    elements: int
    mismatched: int
    max_abs_diff: float
    mean_abs_diff: float
    max_ulp_diff: int

    @property
    def ok(self) -> bool:
        return self.status == "identical"


def _as_float(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Reinterpret raw bit patterns as float64 for reporting only.

    bf16 has no numpy dtype, so it is widened to fp32 the way the hardware
    does: a bf16 *is* the top 16 bits of an fp32 with the same value, so a
    16-bit left shift is exact, not an approximation.
    """
    flat = np.ascontiguousarray(bits).reshape(-1)
    if dtype == "BF16":
        return (flat.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    if dtype == "F16":
        return flat.view(np.float16).astype(np.float64)
    if dtype == "F32":
        return flat.view(np.float32).astype(np.float64)
    if dtype == "I64":
        return flat.view(np.int64).astype(np.float64)
    raise IntegrityError(f"unsupported dtype for comparison: {dtype!r}")


def _ulp_gap(left: np.ndarray, right: np.ndarray, dtype: str) -> int:
    """Largest ULP distance between two batches of raw bit patterns.

    Skipped for 8-byte dtypes: their monotone keys span the full uint64 range,
    so the subtraction has nowhere to widen into. Integers have no meaningful
    ULP anyway -- `max_abs_diff` already says everything for I64.
    """
    width, kind = dtype_spec(dtype)
    if width > 4 or kind not in (FLOAT, SINT):
        return 0
    lk = to_monotone_key(np.ascontiguousarray(left).reshape(-1), kind).astype(np.int64)
    rk = to_monotone_key(np.ascontiguousarray(right).reshape(-1), kind).astype(np.int64)
    gap = np.abs(lk - rk)
    return int(gap.max()) if gap.size else 0


def _compare_one(
    left: CheckpointSource,
    right: CheckpointSource,
    name: str,
    *,
    batch_bytes: int,
) -> TensorComparison:
    ls, rs = left.spec(name), right.spec(name)
    if ls.dtype != rs.dtype:
        return TensorComparison(name, "dtype_mismatch", 0, 0, 0.0, 0.0, 0)
    if ls.shape != rs.shape:
        return TensorComparison(name, "shape_mismatch", 0, 0, 0.0, 0.0, 0)

    elements = ls.num_rows * ls.row_elems
    mismatched = 0
    max_abs = 0.0
    abs_sum = 0.0
    max_ulp = 0

    step = _rows_per_batch(ls, batch_bytes)
    for start in range(0, ls.num_rows, step):
        stop = min(start + step, ls.num_rows)
        lb = left.rows(name, start, stop)
        rb = right.rows(name, start, stop)
        if lb.shape != rb.shape:
            return TensorComparison(name, "shape_mismatch", elements, 0, 0.0, 0.0, 0)

        differing = lb != rb
        # Fast path: bit-equal batches are the expected case, and computing
        # float statistics over them is pure waste on a multi-GiB checkpoint.
        if not differing.any():
            continue

        mismatched += int(differing.sum())
        delta = np.abs(_as_float(lb, ls.dtype) - _as_float(rb, rs.dtype))
        finite = delta[np.isfinite(delta)]
        if finite.size:
            max_abs = max(max_abs, float(finite.max()))
            abs_sum += float(finite.sum())
        max_ulp = max(max_ulp, _ulp_gap(lb, rb, ls.dtype))

    return TensorComparison(
        name=name,
        status="identical" if mismatched == 0 else "differs",
        elements=elements,
        mismatched=mismatched,
        max_abs_diff=max_abs,
        mean_abs_diff=(abs_sum / mismatched) if mismatched else 0.0,
        max_ulp_diff=max_ulp,
    )


def compare_sources(
    left: CheckpointSource,
    right: CheckpointSource,
    *,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> List[TensorComparison]:
    """Compare every tensor of two checkpoints, streaming.

    Tensor sets are unioned rather than intersected, so a tensor present on
    only one side is reported as `missing_left`/`missing_right` instead of
    quietly disappearing from the report -- "identical except for the ones I
    didn't look at" is the one answer this must never give.
    """
    left_names, right_names = set(left.names()), set(right.names())
    results: List[TensorComparison] = []

    for name in sorted(left_names | right_names):
        if name not in left_names:
            results.append(TensorComparison(name, "missing_left", 0, 0, 0.0, 0.0, 0))
        elif name not in right_names:
            results.append(TensorComparison(name, "missing_right", 0, 0, 0.0, 0.0, 0))
        else:
            results.append(_compare_one(left, right, name, batch_bytes=batch_bytes))
    return results
