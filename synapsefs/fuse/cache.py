"""Bounded LRU chunk cache for the FUSE layer.

Tracks memory in bytes rather than object counts so that peak RSS remains
strictly bounded under sustained reads and concurrent access (TeamInstructions
section B, CLI.md --cache-size).
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Optional


class ChunkCache:
    """Thread-safe LRU cache bounded by total byte size."""

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
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

