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

                rows_data = self._get_rows(seg.name, start_row, end_row)
                raw_view = memoryview(rows_data).cast("B")

                byte_lo = rel_start - start_row * seg.row_nbytes
                byte_hi = byte_lo + (int_end - int_start)
                dest_lo = int_start - offset
                dest_hi = dest_lo + (int_end - int_start)

                buf[dest_lo:dest_hi] = raw_view[byte_lo:byte_hi]

        return bytes(buf)

    def _get_rows(self, name: str, start_row: int, end_row: int) -> np.ndarray:
        """Fetch row slice, using ChunkCache if available."""
        if self.cache is None:
            return self.checkpoint.rows(name, start_row, end_row)

        cache_key = (self.checkpoint.commit_hash, name, start_row, end_row)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        rows = self.checkpoint.rows(name, start_row, end_row)
        self.cache.put(cache_key, rows, rows.nbytes)
        return rows

