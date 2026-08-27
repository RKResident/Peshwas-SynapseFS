"""mmap'd checkpoint reader and the two-layer sliding window.

Nothing here copies tensor data except as_float(). The file is mapped once and
every accessor returns a numpy view into that mapping, so the cost of holding a
3 GB checkpoint open is page cache the kernel can reclaim, not resident heap.

16-bit dtypes come out as raw uint16 bits. BF16 has no numpy dtype at all, and
F16 is easier to keep honest as opaque bits -- the codec team keys() them
anyway. Widening happens in as_float and nowhere else: BF16 by shifting the
bits into the high half of a uint32, which is exactly what the truncation that
produced them threw away.

A numpy view exports the mapping's buffer, so CPython refuses to unmap it
while one is alive: close() cannot dangle a pointer, but it also cannot free
anything until the last view is dropped. Holding views past close() is an RSS
leak, not a crash. Fresh calls after close() raise instead.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from math import prod
from typing import Iterator, Sequence

import numpy as np

from .Error import MalformedCheckpoint
from .IR import TensorRef

ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "BOOL": 1,
}

STORAGE_DTYPE = {
    "F64": np.float64, "F32": np.float32, "F16": np.uint16, "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}

MAX_HEADER = 100 << 20


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def cols(self) -> int:
        return prod(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def count(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def ref(self) -> TensorRef:
        return TensorRef(self.name, self.shape, self.dtype)


class SafetensorsReader:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._fh = open(self.path, "rb")
        try:
            self._map = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as e:
            self._fh.close()
            raise MalformedCheckpoint(f"{self.path}: cannot map ({e})") from e

        size = len(self._map)
        if size < 8:
            self.close()
            raise MalformedCheckpoint(f"{self.path}: {size} bytes, too short for a header")

        header_len = int.from_bytes(self._map[0:8], "little")
        if header_len == 0 or header_len > MAX_HEADER or 8 + header_len > size:
            self.close()
            raise MalformedCheckpoint(
                f"{self.path}: header_len {header_len} does not fit in {size} bytes"
            )

        self.data_start = 8 + header_len
        self.data_len = size - self.data_start
        self._header_span = (0, self.data_start)

        try:
            blob = json.loads(bytes(self._map[8:self.data_start]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self.close()
            raise MalformedCheckpoint(f"{self.path}: header is not UTF-8 JSON ({e})") from e
        if not isinstance(blob, dict):
            self.close()
            raise MalformedCheckpoint(f"{self.path}: header JSON is not an object")

        self.metadata: dict[str, str] = blob.pop("__metadata__", {}) or {}
        self._entries: dict[str, TensorEntry] = {}
        try:
            for name, spec in blob.items():
                self._entries[name] = self._entry_from(name, spec)
        except MalformedCheckpoint:
            self.close()
            raise

    def _entry_from(self, name: str, spec: object) -> TensorEntry:
        if not isinstance(spec, dict):
            raise MalformedCheckpoint(f"{name}: header entry is not an object")
        try:
            dtype = spec["dtype"]
            shape = tuple(int(d) for d in spec["shape"])
            begin, end = (int(x) for x in spec["data_offsets"])
        except (KeyError, TypeError, ValueError) as e:
            raise MalformedCheckpoint(f"{name}: malformed header entry ({e})") from e

        if dtype not in ITEMSIZE:
            raise MalformedCheckpoint(f"{name}: unknown dtype '{dtype}'")
        if any(d < 0 for d in shape):
            raise MalformedCheckpoint(f"{name}: negative dimension in {shape}")
        if begin < 0 or end < begin:
            raise MalformedCheckpoint(f"{name}: bad data_offsets [{begin}, {end})")
        if end > self.data_len:
            raise MalformedCheckpoint(
                f"{name}: data_offsets end {end} past data region ({self.data_len} bytes)"
            )

        expected = (prod(shape) if shape else 1) * ITEMSIZE[dtype]
        if end - begin != expected:
            raise MalformedCheckpoint(
                f"{name}: data_offsets span {end - begin} bytes, shape {shape} "
                f"{dtype} needs {expected}"
            )
        return TensorEntry(name, dtype, shape, begin, end)

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    @property
    def header_bytes(self) -> memoryview:
        """[0, 8+header_len) verbatim -- what storage replays for byte-exactness."""
        lo, hi = self._header_span
        return self._map[lo:hi]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def entry(self, name: str) -> TensorEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor '{name}'") from None

    def ref(self, name: str) -> TensorRef:
        return self.entry(name).ref

    def refs(self) -> dict[str, TensorRef]:
        return {n: e.ref for n, e in self._entries.items()}

    def tensor(self, name: str) -> np.ndarray:
        """Storage-dtype view. 16-bit dtypes arrive as uint16 bits."""
        if self._map is None:
            raise ValueError(f"{self.path}: reader is closed")
        e = self.entry(name)
        flat = np.frombuffer(
            self._map, dtype=STORAGE_DTYPE[e.dtype], count=e.count,
            offset=self.data_start + e.begin,
        )
        return flat.reshape(e.shape) if e.shape else flat.reshape(())

    def matrix(self, name: str) -> np.ndarray:
        """Same view, folded to the logical 2D [rows, cols] the solver works in."""
        e = self.entry(name)
        return self.tensor(name).reshape(e.rows, e.cols)

    def as_float(self, name: str) -> np.ndarray:
        """The only copy in this module. Returns an owned 2D float32 array."""
        e = self.entry(name)
        raw = self.tensor(name)
        if e.dtype == "BF16":
            wide = raw.astype(np.uint32)
            np.left_shift(wide, 16, out=wide)
            out = wide.view(np.float32)
        elif e.dtype == "F16":
            out = raw.view(np.float16).astype(np.float32)
        else:
            out = raw.astype(np.float32)
        return out.reshape(e.rows, e.cols)

    def close(self) -> None:
        m = getattr(self, "_map", None)
        if m is not None:
            try:
                m.close()
            except BufferError:
                pass  # live views still export it; refcounting unmaps later
            self._map = None
        fh = getattr(self, "_fh", None)
        if fh is not None and not fh.closed:
            fh.close()

    def __enter__(self) -> SafetensorsReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LayerWindow:
    """Two adjacent layers of both checkpoints, and nothing else.

    Algorithm 1's objective for group l reads W_l (rows) and W_{l+1} (columns),
    so four float32 matrices must be resident at once. advance() slides forward
    and drops the trailing layer; the count never exceeds four. On the last
    layer next_name is None and only two are held.
    """

    SIDES = ("a", "b")

    def __init__(self, a: SafetensorsReader, b: SafetensorsReader,
                 names: Sequence[str]) -> None:
        self._names = list(names)
        if not self._names:
            raise ValueError("LayerWindow needs at least one tensor name")
        self._a, self._b = a, b
        for n in self._names:
            if n not in a:
                raise MalformedCheckpoint(f"'{n}' missing from {a.path}")
            if n not in b:
                raise MalformedCheckpoint(f"'{n}' missing from {b.path}")
            if a.ref(n).shape != b.ref(n).shape:
                raise MalformedCheckpoint(
                    f"'{n}': shape {a.ref(n).shape} vs {b.ref(n).shape}"
                )
        self._i = 0
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    @property
    def index(self) -> int:
        return self._i

    @property
    def name(self) -> str:
        return self._names[self._i]

    @property
    def next_name(self) -> str | None:
        nxt = self._i + 1
        return self._names[nxt] if nxt < len(self._names) else None

    @property
    def at_end(self) -> bool:
        return self.next_name is None

    def _live(self) -> set[tuple[str, str]]:
        live = {(s, self.name) for s in self.SIDES}
        if self.next_name is not None:
            live |= {(s, self.next_name) for s in self.SIDES}
        return live

    def _get(self, side: str, name: str | None) -> np.ndarray | None:
        if name is None:
            return None
        key = (side, name)
        hit = self._cache.get(key)
        if hit is None:
            reader = self._a if side == "a" else self._b
            hit = self._cache[key] = reader.as_float(name)
        return hit

    @property
    def a(self) -> np.ndarray:
        return self._get("a", self.name)

    @property
    def b(self) -> np.ndarray:
        return self._get("b", self.name)

    @property
    def a_next(self) -> np.ndarray | None:
        return self._get("a", self.next_name)

    @property
    def b_next(self) -> np.ndarray | None:
        return self._get("b", self.next_name)

    @property
    def resident(self) -> int:
        return len(self._cache)

    @property
    def resident_names(self) -> list[str]:
        return sorted(f"{s}:{n}" for s, n in self._cache)

    @property
    def resident_bytes(self) -> int:
        return sum(m.nbytes for m in self._cache.values())

    def advance(self) -> bool:
        if self.at_end:
            return False
        self._i += 1
        live = self._live()
        for key in [k for k in self._cache if k not in live]:
            del self._cache[key]
        return True

    def reset(self) -> None:
        self._i = 0
        self._cache.clear()

    def __iter__(self) -> Iterator[LayerWindow]:
        while True:
            yield self
            if not self.advance():
                return