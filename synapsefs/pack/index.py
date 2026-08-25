"""Pack index writer and mmap reader (docs/FORMAT.md section 6).

An index answers one question as cheaply as possible: *given a content hash,
where in the pack is that chunk?* Layout, from FORMAT.md 6.1::

    0          8      magic      b"SYNIDX\\0\\0"
    8          4      version    u32 LE
    12         4      count      u32 LE = N
    16         32     pack_hash  blake3 of the .pack this indexes
    48         1024   fanout     256 x u32 LE, cumulative
    1072       N x 32 hashes     raw 32-byte hashes, sorted ascending
    1072+32N   N x 8  offsets    u64 LE, payload offset into the .pack
    1072+40N   N x 4  stored_len u32 LE
    1072+44N   N x 4  plain_len  u32 LE
    1072+48N   N x 8  checksum   first 8 bytes of blake3(stored payload)
    EOF-32     32     trailer    blake3 of bytes [0, EOF-32)

Three design choices carry their weight here, and all three are about the read
side rather than the write side:

**Parallel arrays, not interleaved records.** A binary search touches only the
hash array, so ~8 probes pull in ~8 cache lines. Interleaved 56-byte records
would scatter those probes across the file and drag in offset, length and
checksum bytes the search never looks at.

**A 256-entry fanout table.** `fanout[b]` is the number of entries whose first
hash byte is `<= b`, cumulative, so `fanout[255] == count`. It turns the
search into a bounded one over the ~N/256 entries sharing a first byte,
knocking 8 comparisons off the front for free.

**Never a Python dict.** FORMAT.md 6.3 calls this out as an implementation
trap: a dict keyed by 64-char hex strings costs 50-80 MB resident at 280k
entries, against a graded 7% peak-RSS metric. Hashes stay as raw slices of the
mmap and are compared as bytes. Nothing here builds a per-entry Python object
except the single `PackEntry` a successful lookup returns.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import blake3

from synapsefs.errors import IntegrityError
from synapsefs.pack.pack import (
    CHECKSUM_SIZE,
    HASH_SIZE,
    TRAILER_SIZE,
    PackEntry,
)
from synapsefs.store.atomic import atomic_writer

__all__ = ["MAGIC", "VERSION", "HEADER_SIZE", "FANOUT_SIZE", "write_index", "PackIndex"]

MAGIC = b"SYNIDX\0\0"
VERSION = 1

HEADER_SIZE = 48          # magic + version + count + pack_hash
FANOUT_ENTRIES = 256
FANOUT_SIZE = FANOUT_ENTRIES * 4
ARRAYS_START = HEADER_SIZE + FANOUT_SIZE      # 1072
ENTRY_SIZE = HASH_SIZE + 8 + 4 + 4 + CHECKSUM_SIZE  # 56

_HEADER = struct.Struct("<8sII32s")


def _fanout(sorted_hashes: Sequence[bytes]) -> bytes:
    """Build the cumulative first-byte table.

    Cumulative is the part that is easy to get subtly wrong: `fanout[b]` counts
    every entry whose first byte is `<= b`, not `== b`. A lookup then reads
    `fanout[b-1] .. fanout[b]` as its search bounds, so an off-by-one here
    produces an index that finds most hashes and silently misses a few -- the
    worst possible failure shape.
    """
    counts = [0] * FANOUT_ENTRIES
    for h in sorted_hashes:
        counts[h[0]] += 1
    cumulative: List[int] = []
    running = 0
    for c in counts:
        running += c
        cumulative.append(running)
    return struct.pack(f"<{FANOUT_ENTRIES}I", *cumulative)


def write_index(
    index_path: Path,
    *,
    pack_hash: bytes,
    entries: Iterable[PackEntry],
    tmp_dir: Path,
) -> Path:
    """Write the `.idx` for a pack. Returns `index_path`.

    `entries` may arrive in any order -- typically the order the chunks were
    written, which is tensor order, not hash order. They are sorted here,
    because the whole read path depends on the hash array being ascending.

    Duplicate hashes are rejected rather than tolerated. `encode_checkpoint`
    already dedups, so a duplicate means something upstream is wrong, and a
    duplicated key in a binary-searched array is exactly the kind of fault
    that produces plausible-but-wrong lookups instead of an error.
    """
    entries = sorted(entries, key=lambda e: e.content_hash)
    hashes = [e.content_hash for e in entries]
    for i in range(1, len(hashes)):
        if hashes[i] == hashes[i - 1]:
            raise ValueError(f"duplicate chunk hash in index: {hashes[i].hex()}")

    count = len(entries)
    body = b"".join(
        [
            _HEADER.pack(MAGIC, VERSION, count, pack_hash),
            _fanout(hashes),
            b"".join(hashes),
            struct.pack(f"<{count}Q", *(e.offset for e in entries)),
            struct.pack(f"<{count}I", *(e.stored_len for e in entries)),
            struct.pack(f"<{count}I", *(e.plain_len for e in entries)),
            b"".join(e.checksum for e in entries),
        ]
    )
    with atomic_writer(Path(index_path), tmp_dir=Path(tmp_dir)) as f:
        f.write(body)
        f.write(blake3.blake3(body).digest())
    return Path(index_path)


class PackIndex:
    """mmap-backed, binary-searched pack index. Context manager; also `.close()`.

    Opening is cheap on purpose: it maps the file and reads a 48-byte header,
    and does **not** verify the trailer. Checking the trailer means reading the
    whole index, which belongs in `verify` (see `verify()` below), never in the
    path a page fault takes.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._closed = False
        self._fh = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise

        try:
            self._parse_header()
        except Exception:
            self._mm.close()
            self._fh.close()
            raise

    def _parse_header(self) -> None:
        size = len(self._mm)
        if size < ARRAYS_START + TRAILER_SIZE:
            raise IntegrityError(f"{self.path}: too small to be a pack index ({size} bytes)")
        magic, version, count, pack_hash = _HEADER.unpack(self._mm[:HEADER_SIZE])
        if magic != MAGIC:
            raise IntegrityError(f"{self.path}: bad index magic {magic!r}")
        if version != VERSION:
            raise IntegrityError(f"{self.path}: unsupported index version {version}")

        expected = ARRAYS_START + ENTRY_SIZE * count + TRAILER_SIZE
        if size != expected:
            raise IntegrityError(
                f"{self.path}: size {size} does not match {count} entries "
                f"(expected {expected})"
            )

        self.count = count
        self.pack_hash = pack_hash
        # A memoryview over the mmap, so slicing below is a view rather than a
        # copy. This is the "no per-entry Python objects" rule in practice.
        self._view = memoryview(self._mm)
        self._hashes = ARRAYS_START
        self._offsets = self._hashes + HASH_SIZE * count
        self._stored = self._offsets + 8 * count
        self._plain = self._stored + 4 * count
        self._checksums = self._plain + 4 * count

    # -- lookup ------------------------------------------------------------

    def _hash_at(self, i: int) -> bytes:
        start = self._hashes + i * HASH_SIZE
        return bytes(self._view[start : start + HASH_SIZE])

    def _fanout_at(self, b: int) -> int:
        start = HEADER_SIZE + b * 4
        return struct.unpack_from("<I", self._mm, start)[0]

    def lookup(self, content_hash: bytes) -> Optional[PackEntry]:
        """Find a chunk by content hash, or return None.

        FORMAT.md 6.2: narrow to the entries sharing a first byte using the
        fanout, then binary-search that span.
        """
        if len(content_hash) != HASH_SIZE:
            raise ValueError(f"content_hash must be {HASH_SIZE} bytes, got {len(content_hash)}")

        first = content_hash[0]
        lo = 0 if first == 0 else self._fanout_at(first - 1)
        hi = self._fanout_at(first)

        while lo < hi:
            mid = (lo + hi) // 2
            candidate = self._hash_at(mid)
            if candidate < content_hash:
                lo = mid + 1
            elif candidate > content_hash:
                hi = mid
            else:
                return self._entry_at(mid)
        return None

    def _entry_at(self, i: int) -> PackEntry:
        return PackEntry(
            content_hash=self._hash_at(i),
            offset=struct.unpack_from("<Q", self._mm, self._offsets + i * 8)[0],
            stored_len=struct.unpack_from("<I", self._mm, self._stored + i * 4)[0],
            plain_len=struct.unpack_from("<I", self._mm, self._plain + i * 4)[0],
            checksum=bytes(
                self._view[
                    self._checksums + i * CHECKSUM_SIZE :
                    self._checksums + (i + 1) * CHECKSUM_SIZE
                ]
            ),
        )

    def entries(self) -> List[PackEntry]:
        """Every entry, in hash order. For tests and for `verify`; the read hot
        path uses `lookup`, which builds exactly one object."""
        return [self._entry_at(i) for i in range(self.count)]

    def verify(self) -> None:
        """Recompute the trailer and check it. Raises IntegrityError on
        mismatch. Reads the whole file, so keep it out of the read path."""
        size = len(self._mm)
        body = self._view[: size - TRAILER_SIZE]
        stored = bytes(self._view[size - TRAILER_SIZE :])
        computed = blake3.blake3(body).digest()
        if computed != stored:
            raise IntegrityError(
                f"{self.path}: index trailer mismatch -- stored {stored.hex()[:16]}..., "
                f"computed {computed.hex()[:16]}..."
            )

    # -- lifecycle ---------------------------------------------------------

    def __len__(self) -> int:
        return self.count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._view.release()
        try:
            self._mm.close()
        except BufferError:
            pass
        finally:
            self._fh.close()

    def __enter__(self) -> "PackIndex":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
