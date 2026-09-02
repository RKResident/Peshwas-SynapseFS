"""Bounded LRU chunk cache for the FUSE layer.

Tracks memory in bytes rather than object counts so that peak RSS remains
strictly bounded under sustained reads and concurrent access (TeamInstructions
section B, CLI.md --cache-size).
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Optional

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


class ChunkCache:
    """Thread-safe LRU cache bounded by total byte size."""

    def __init__(self, max_bytes: int = DEFAULT_CACHE_SIZE_BYTES) -> None:
        self.max_bytes = max(0, max_bytes)
        self.current_bytes = 0
        self._cache: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._lock = threading.Lock()

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

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

