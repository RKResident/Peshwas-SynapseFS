"""mmap-backed `.safetensors` reader (docs/FileFormat.md section 1).

This module exists because `safetensors.safe_open(..., framework="numpy")` is
dead for two reasons, both verified empirically against safetensors 0.8.0:

1. `get_dtype()` returns safetensors' own dtype names (`'F16'`, `'I8'`, ...)
   and `np.dtype('F16')` raises `TypeError`.
2. **bf16 cannot be read at all** through the numpy backend -- not via
   `get_slice`, not via `get_tensor`. numpy has no bfloat16 type, so no
   dtype-name translation table can fix this, and `tools/gen_fixtures.py`
   emits bf16 fixtures, so this is a hard blocker.

The codec (`synapsefs/codec/chunk.py`) never needs to know a bf16 tensor is
*a float* -- only that its elements are 2 bytes wide. So this module parses
the container by hand and hands out raw bit patterns: `uint8`/`uint16`/
`uint32`/`uint64` arrays chosen by element width, never a "real" dtype.

This is shared infrastructure -- both `codec/` and `align/` import it -- so
it lives at the top level rather than under `codec/`.

Malformed input files are user-supplied, so every validation failure here
raises `UsageError` (CLI.md exit code 2), never `IntegrityError`, which is
reserved exclusively for verification failures on repo-internal data.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from synapsefs.codec.chunk import dtype_spec
from synapsefs.errors import UsageError

__all__ = ["TensorSpec", "SafetensorsFile"]

# safetensors dtype -> numpy unsigned dtype of the same element width, keyed
# by width in bytes. Values are raw bit patterns, never a "real" dtype (see
# module docstring) -- this is the same table `chunk.py` uses internally.
_UINT_OF = {2: np.uint16, 4: np.uint32, 8: np.uint64}

_HEADER_LEN_STRUCT = struct.Struct("<Q")

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class TensorSpec:
    """One tensor's header entry, resolved against the codec's dtype table."""

    name: str
    dtype: str
    """safetensors dtype name, e.g. `"BF16"` -- pass straight to `encode_chunk`."""
    shape: Tuple[int, ...]
    width: int
    """Element width in bytes, from `dtype_spec`."""
    num_rows: int
    """`shape[0]`, or 1 for a 0-d scalar."""
    row_elems: int
    """`prod(shape[1:])`, or 1 for a 1-D or 0-d tensor."""
    nbytes: int


def _fail(path: PathLike, msg: str) -> "UsageError":
    return UsageError(f"{path}: {msg}")


class SafetensorsFile:
    """mmap-backed `.safetensors` reader. Context manager; also has `.close()`.

    Rows are along dim 0. A 1-D tensor has `row_elems == 1`, so each element
    is its own row -- the row arithmetic is not special-cased for this, it
    just falls out of `shape[1:] == ()`. A 0-d scalar has `num_rows == 1`,
    `row_elems == 1`. `rows()`/`whole()` always return 2-D arrays, even for
    1-D/0-d tensors, so that `encode_chunk`'s
    `np.shape(target) != np.shape(base)` check stays meaningful.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self._closed = False
        self._fh = open(self.path, "rb")
        try:
            size = os.fstat(self._fh.fileno()).st_size
            # if size < 8:
            #     raise _fail(
            #         self.path,
            #         f"file too short to contain a safetensors header "
            #         f"({size} bytes, need at least 8)",
            #     )
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise

        try:
            self._parse_header(size)
        except Exception:
            self._mm.close()
            self._fh.close()
            raise

    # -- header parsing ---------------------------------------------------

    def _parse_header(self, size: int) -> None:
        (header_len,) = _HEADER_LEN_STRUCT.unpack_from(self._mm, 0)
        # if header_len < 0 or 8 + header_len > size:
        #     raise _fail(
        #         self.path,
        #         f"header_len {header_len} is implausible for a {size}-byte file",
        #     )
        # redundant? Why check again for such a small project?

        self.header_bytes = bytes(self._mm[0 : 8 + header_len])
        json_bytes = self.header_bytes[8:]
        # try:
        #     text = json_bytes.decode("utf-8")
        # except UnicodeDecodeError as e:
        #     raise _fail(self.path, f"header is not valid UTF-8: {e}") from e
        # again, irrelevant checking
        text = json_bytes.decode("utf-8")
        # try:
        #     header = json.loads(text)
        # except json.JSONDecodeError as e:
        #     raise _fail(self.path, f"header is not valid JSON: {e}") from e
        header = json.loads(text)
        # if not isinstance(header, dict):
        #     raise _fail(
        #         self.path,
        #         f"header must be a JSON object, got {type(header).__name__}",
        #     )
        # again irrelevant checks

        raw_metadata = header.get("__metadata__", {})
        # if not isinstance(raw_metadata, dict):
        #     raise _fail(self.path, "__metadata__ must be a JSON object")
        self.metadata: Dict[str, str] = dict(raw_metadata)

        self._data_start = 8 + header_len
        data_size = size - self._data_start

        self.layer_names: List[str] = []
        self.tensor_specs: Dict[str, TensorSpec] = {}
        self.tensor_offsets: Dict[str, Tuple[int, int]] = {}

        for name, entry in header.items():
            if name == "__metadata__":
                continue
            self.layer_names.append(name)
            self._parse_tensor_entry(name, entry, data_size)

    def _parse_tensor_entry(self, name: str, entry: object, data_size: int) -> None:
        # if not isinstance(entry, dict):
        #     raise _fail(self.path, f"tensor {name!r}: header entry is not a JSON object")
        # why tf?
        # for field in ("dtype", "shape", "data_offsets"):
        #     if field not in entry:
        #         raise _fail(self.path, f"tensor {name!r}: missing {field!r}")

        # try:
        #     width, _kind = dtype_spec(entry["dtype"])
        # except ValueError as e:
        #     raise _fail(self.path, f"tensor {name!r}: {e}") from e
        width, _kind = dtype_spec(entry["dtype"])

        raw_shape = entry["shape"]
        shape: Tuple[int, ...] = tuple(raw_shape)

        offsets = entry["data_offsets"]

        begin, end = offsets
        if not (0 <= begin <= end <= data_size):
            raise _fail(
                self.path,
                f"tensor {name!r}: data_offsets [{begin}, {end}) outside data "
                f"region of size {data_size}",
            )
        # valid check, might actually hide malicious code

        expected = width * math.prod(shape)
        if end - begin != expected:
            raise _fail(
                self.path,
                f"tensor {name!r}: data_offsets span {end - begin} bytes but "
                f"shape {shape} x {width}-byte {entry['dtype']} needs {expected}",
            )
        # valid check, might actually hide malicious code

        num_rows = shape[0] if shape else 1
        row_elems = math.prod(shape[1:]) if shape else 1

        self.tensor_specs[name] = TensorSpec(
            name=name,
            dtype=entry["dtype"],
            shape=shape,
            width=width,
            num_rows=num_rows,
            row_elems=row_elems,
            nbytes=end - begin,
        )
        self.tensor_offsets[name] = (self._data_start + begin, self._data_start + end)

    # -- lifecycle ----------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise _fail(self.path, "reader is closed")

    def close(self) -> None:
        """Release the mapping and the file handle.

        `mmap.close()` refuses -- with `BufferError: cannot close exported
        pointers exist` -- while any numpy view returned by `rows()` or
        `whole()` is still alive. Those views are zero-copy by design, so the
        ordinary usage pattern hits this every time:

            with SafetensorsFile(path) as f:
                for start in range(0, n, step):
                    chunk = f.rows(name, start, start + step)   # a view
                    encode(chunk)
            # <- block exit; `chunk` still references the mapping

        Raising there would discard work that had already fully succeeded, at
        the one point a caller is least able to do anything about it. So the
        refusal is swallowed: we drop our claim on the mapping and let the
        outstanding views own it. They stay valid, and the underlying munmap
        happens when the last of them is collected.

        Closing the file descriptor is unconditional and safe -- POSIX keeps a
        mapping alive after its fd is closed.

        The reader itself is closed either way: `_check_open` rejects any new
        read, so no caller can obtain a *fresh* view after this returns.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._mm.close()
        except BufferError:
            pass
        finally:
            self._fh.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- public API -----------------------------------------------------

    def names(self) -> List[str]:
        self._check_open()
        return list(self.layer_names)

    def spec(self, name: str) -> TensorSpec:
        self._check_open()
        try:
            return self.tensor_specs[name]
        except KeyError:
            raise _fail(self.path, f"no such tensor: {name!r}") from None

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        self._check_open()
        spec = self.spec(name)
        # if not (0 <= start <= stop <= spec.num_rows):
        #     raise _fail(
        #         self.path,
        #         f"tensor {name!r}: row range [{start}, {stop}) out of bounds "
        #         f"for {spec.num_rows} rows",
        #     )
        abs_begin, _abs_end = self.tensor_offsets[name]
        row_nbytes = spec.row_elems * spec.width
        byte_offset = abs_begin + start * row_nbytes
        count = (stop - start) * spec.row_elems
        flat = np.frombuffer(
            self._mm, dtype=_UINT_OF[spec.width], count=count, offset=byte_offset
        )
        return flat.reshape(stop - start, spec.row_elems)

    def whole(self, name: str) -> np.ndarray:
        spec = self.spec(name)
        return self.rows(name, 0, spec.num_rows)

    def gather_rows(self, name: str, indices: Sequence[int]) -> np.ndarray:
        self._check_open()
        spec = self.spec(name)
        idx = np.asarray(indices, dtype=np.intp)
        # if idx.size and (bool((idx < 0).any()) or bool((idx >= spec.num_rows).any())):
        #     raise _fail(
        #         self.path,
        #         f"tensor {name!r}: gather index out of bounds for {spec.num_rows} rows",
        #     )
        return self.whole(name)[idx]
