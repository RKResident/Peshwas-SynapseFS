"""Bounded LRU chunk cache for the FUSE layer.

Tracks memory in bytes rather than object counts so that peak RSS remains
strictly bounded under sustained reads and concurrent access (TeamInstructions
section B, CLI.md --cache-size).
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Callable, Optional

#: Default cap on decoded chunks held in memory.
#:
#: The cache exists to bridge one specific gap: FUSE hands out 128 KiB reads
#: against 4 MiB chunks, so without it every block re-decodes the whole chunk
#: (and its delta base) to return 1/32nd of it. Bridging that needs only the
#: chunks currently in flight -- roughly one per concurrent reader -- not the
#: whole file.
#:
#: Measured on the 25-epoch 90M benchmark, 8 concurrent readers of distinct
#: commits, peak daemon RSS against cold aggregate throughput:
#:
#:     cache      0 MiB ->  21 MB/s,  290 MiB RSS
#:     cache     32 MiB -> 207 MB/s,  266 MiB RSS
#:     cache    128 MiB -> 203 MB/s,  361 MiB RSS
#:     cache    512 MiB -> 224 MB/s,  860 MiB RSS
#:
#: 32 MiB buys the entire ~10x and costs *less* RSS than running with no cache
#: at all, because the decode churn a cache miss causes is itself expensive.
#: Everything above it retains chunks no reader comes back to: 512 MiB paid
#: 594 MiB of RSS for throughput inside the run-to-run noise.
DEFAULT_CACHE_SIZE_BYTES = 32 * 1024 * 1024


class _Flight:
    """One decode in progress, and the slot its result lands in.

    The result is handed to waiters through this object rather than through the
    cache, because a small cache can evict the entry between the owner's `put`
    and a waiter waking up -- which would send every waiter off to recompute
    exactly what they queued to avoid.
    """

    __slots__ = ("event", "value", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None


class ChunkCache:
    """Thread-safe LRU cache bounded by total byte size."""

    def __init__(self, max_bytes: int = DEFAULT_CACHE_SIZE_BYTES) -> None:
        self.max_bytes = max(0, max_bytes)
        self.current_bytes = 0
        self._cache: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._lock = threading.Lock()
        #: key -> _Flight for decodes currently in progress. Bounded by the
        #: number of worker threads, so it never needs eviction of its own.
        self._inflight: dict[Any, "_Flight"] = {}

    def get(self, key: Any) -> Optional[Any]:
        """Fetch an item from cache and mark it most recently used."""
        with self._lock:
            if key not in self._cache:
                return None
            val, nbytes = self._cache[key]
            self._cache.move_to_end(key)
            return val

    def put(self, key: Any, val: Any, nbytes: int) -> None:
        """Store an item in cache, evicting oldest entries to stay under max_bytes."""
        if self.max_bytes <= 0:
            return

        with self._lock:
            # If replacing an existing key, subtract previous size
            if key in self._cache:
                _, old_bytes = self._cache.pop(key)
                self.current_bytes -= old_bytes

            # If the single item exceeds max_bytes, don't cache it
            if nbytes > self.max_bytes:
                return

            while self.current_bytes + nbytes > self.max_bytes and self._cache:
                _evicted_key, (_evicted_val, evicted_bytes) = self._cache.popitem(last=False)
                self.current_bytes -= evicted_bytes

            self._cache[key] = (val, nbytes)
            self.current_bytes += nbytes

    def get_or_compute(self, key: Any, factory: Callable[[], Any],
                       sizeof: Optional[Callable[[Any], int]] = None) -> Any:
        """Cached value for `key`, computing it at most once across threads.

        The plain check-miss-compute-put sequence has a window between the miss
        and the put where the value is being produced but is not yet visible.
        FUSE readahead fires several requests into the same chunk at once and
        the worker threads pick them up together, so every one of them misses
        and every one of them decodes the same chunk from the same bytes to the
        same answer. Measured on the 90M benchmark, one reader, a cache far
        larger than the working set so eviction could not be the cause:

            SYNAPSEFS_READ_THREADS=1    1.00x the minimum object bytes
            SYNAPSEFS_READ_THREADS=8    1.48x

        48% of the decode work at eight threads was that race. It costs peak
        RSS as well as time: each duplicate decode allocates its own
        decompressed stream, unshuffled array and base block, where sharing one
        result costs a single reference.

        The first caller to miss owns the decode; the rest wait on its
        `_Flight` and receive the same object. The cache lock is never held
        across `factory()`, and the owner wakes its waiters from a `finally`,
        so a decode that raises propagates to everyone instead of leaving them
        parked forever.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                return entry[0]
            flight = self._inflight.get(key)
            if flight is None:
                flight = self._inflight[key] = _Flight()
                owner = True
            else:
                owner = False

        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value

        try:
            value = factory()
        except BaseException as exc:
            flight.error = exc
            raise
        else:
            flight.value = value
            nbytes = sizeof(value) if sizeof is not None else getattr(value, "nbytes", 0)
            self.put(key, value, nbytes)
            return value
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            flight.event.set()

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

