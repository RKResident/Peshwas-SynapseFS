"""Virtual .safetensors file reconstruction for FUSE reads.

Maps arbitrary byte offset and size requests to the verbatim header and
tensor row ranges without pre-materialising any files on disk (PS module 2h,
TeamInstructions section B).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from synapsefs.fuse.cache import ChunkCache
from synapsefs.graph import CommitCheckpoint


@dataclass(frozen=True)
class TensorSegment:
    """Byte span and dimension metadata for one tensor in the safetensors file."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    width: int
    num_rows: int
    row_elems: int
    row_nbytes: int
    abs_begin: int
    abs_end: int


class VirtualSafetensorsFile:
    """Read-only virtual representation of a committed safetensors checkpoint.

    Serves arbitrary `read(offset, size)` byte requests by mapping byte spans
    to header bytes or on-demand decoded tensor row slices.
    """

    def __init__(
        self,
        checkpoint: CommitCheckpoint,
        cache: Optional[ChunkCache] = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.cache = cache
        self._span_cache: Dict[str, List[Tuple[int, int]]] = {}
        self.header_bytes = checkpoint.header_bytes
        self._parse_header()

    def _parse_header(self) -> None:
        if len(self.header_bytes) < 8:
            raise ValueError("header_bytes too short")
        (header_len,) = struct.unpack("<Q", self.header_bytes[:8])
        self.data_start = 8 + header_len
        json_bytes = self.header_bytes[8:self.data_start]
        header = json.loads(json_bytes.decode("utf-8"))

        self.segments: List[TensorSegment] = []
        max_end = self.data_start

        for name, spec in header.items():
            if name == "__metadata__":
                continue
            offsets = spec["data_offsets"]
            abs_begin = self.data_start + offsets[0]
            abs_end = self.data_start + offsets[1]
            if abs_end > max_end:
                max_end = abs_end

            shape = tuple(spec["shape"])
            num_rows = shape[0] if shape else 1
            row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            total_elems = int(np.prod(shape)) if shape else 1
            width = (offsets[1] - offsets[0]) // total_elems if total_elems else 1
            row_nbytes = row_elems * width

            self.segments.append(
                TensorSegment(
                    name=name,
                    dtype=spec["dtype"],
                    shape=shape,
                    width=width,
                    num_rows=num_rows,
                    row_elems=row_elems,
                    row_nbytes=row_nbytes,
                    abs_begin=abs_begin,
                    abs_end=abs_end,
                )
            )

        # Sort segments by file offset for fast lookup
        self.segments.sort(key=lambda s: s.abs_begin)
        self.total_size = max_end

    def read(self, offset: int, size: int) -> bytes:
        """Serve a slice of the safetensors file at [offset, offset + size).

        Decodes only the tensor row chunks that intersect the requested range.
        """
        return self._read(offset, size, self._get_rows)

    def try_read_cached(self, offset: int, size: int) -> "bytes | None":
        """The same slice, but only if every chunk it needs is already decoded.

        Lets the FUSE layer answer a cache hit without a thread hop. Dispatching
        to a worker costs ~4x the request itself -- measured on a mount whose
        read() returned a preallocated buffer, 504 MiB/s through
        `trio.to_thread.run_sync` against 1959 MiB/s inline -- and a hit is a
        memoryview slice, so paying that is pure loss. A miss still has to go to
        a thread: decoding on the trio loop would stall every other request for
        the duration.

        Returns None rather than decoding, so the caller can fall back.
        """
        return self._read(offset, size, self._get_rows_cached)

    def _read(self, offset: int, size: int, rows_fn) -> "bytes | None":
        if offset < 0 or size <= 0 or offset >= self.total_size:
            return b""

        end_offset = min(offset + size, self.total_size)
        actual_size = end_offset - offset
        buf = bytearray(actual_size)

        # 1. Header slice
        if offset < self.data_start:
            h_start = offset
            h_end = min(end_offset, self.data_start)
            h_len = h_end - h_start
            buf[0:h_len] = self.header_bytes[h_start:h_end]

        # 2. Tensor data slices
        req_data_start = max(offset, self.data_start)
        if req_data_start < end_offset:
            for seg in self.segments:
                if seg.abs_end <= req_data_start:
                    continue
                if seg.abs_begin >= end_offset:
                    break

                # Intersect [req_data_start, end_offset) with [seg.abs_begin, seg.abs_end)
                int_start = max(req_data_start, seg.abs_begin)
                int_end = min(end_offset, seg.abs_end)
                if int_end <= int_start:
                    continue

                rel_start = int_start - seg.abs_begin
                rel_end = int_end - seg.abs_begin

                # Convert relative bytes to row range
                start_row = rel_start // seg.row_nbytes
                end_row = min((rel_end + seg.row_nbytes - 1) // seg.row_nbytes, seg.num_rows)
                if end_row <= start_row:
                    end_row = start_row + 1

                rows_data = rows_fn(seg.name, start_row, end_row)
                if rows_data is None:
                    return None          # cached-only probe: let the caller decode
                raw_view = memoryview(rows_data).cast("B")

                byte_lo = rel_start - start_row * seg.row_nbytes
                byte_hi = byte_lo + (int_end - int_start)
                dest_lo = int_start - offset
                dest_hi = dest_lo + (int_end - int_start)

                buf[dest_lo:dest_hi] = raw_view[byte_lo:byte_hi]

        return bytes(buf)

    def _spans(self, name: str) -> List[Tuple[int, int]]:
        spans = self._span_cache.get(name)
        if spans is None:
            spans = self._span_cache[name] = self.checkpoint.chunk_spans(name)
        return spans

    def _chunk_rows(self, name: str, lo: int, hi: int) -> np.ndarray:
        """One whole chunk, cached under its own row span.

        `get_or_compute` rather than get/miss/put: concurrent readers of the
        same chunk otherwise all miss and all decode it. See its docstring.
        """
        if self.cache is None:
            return self.checkpoint.rows(name, lo, hi)
        key = (self.checkpoint.commit_hash, name, lo, hi)
        return self.cache.get_or_compute(
            key, lambda: self.checkpoint.rows(name, lo, hi))

    def _get_rows_cached(self, name: str, start_row: int,
                         end_row: int) -> "np.ndarray | None":
        """`_get_rows` restricted to chunks already in the cache. None on a miss.

        Deliberately does not fall back to `checkpoint.rows` for an uncovered
        span the way `_get_rows` does: that path decodes, which is exactly what
        this probe exists to avoid.
        """
        if self.cache is None:
            return None
        spans = self._spans(name)
        covering = [(lo, hi) for lo, hi in spans if hi > start_row and lo < end_row]
        if not covering:
            return None
        commit = self.checkpoint.commit_hash
        pieces = []
        for lo, hi in covering:
            hit = self.cache.get((commit, name, lo, hi))
            if hit is None:
                return None
            pieces.append(hit)
        block = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        base = covering[0][0]
        return block[start_row - base:end_row - base]

    def _get_rows(self, name: str, start_row: int, end_row: int) -> np.ndarray:
        """Rows `[start_row, end_row)`, decoding each chunk at most once.

        The cache is keyed on the CHUNK, not on the request. Keying it on the
        requested range looks equivalent and is not: a sequential reader whose
        block is smaller than a chunk produces a fresh key every call, so it
        never hits, and every call decodes the entire 4 MiB chunk again to
        return 128 KiB of it. Measured on the 92M benchmark that was 8,762 MiB
        of object reads to serve a 177 MiB file -- 31.6x amplification, and
        124x slower than reading the file in one call.

        Snapping to chunk boundaries makes the cache do what it was for: the
        first block of a chunk pays for the decode and the next thirty-one are
        slices of a cache hit.
        """
        spans = self._spans(name)
        covering = [(lo, hi) for lo, hi in spans if hi > start_row and lo < end_row]
        if not covering:
            return self.checkpoint.rows(name, start_row, end_row)

        pieces = [self._chunk_rows(name, lo, hi) for lo, hi in covering]
        block = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        base = covering[0][0]
        return block[start_row - base:end_row - base]

