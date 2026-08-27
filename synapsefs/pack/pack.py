"""Packfile writer and scanner (docs/FORMAT.md section 5).

A pack is an immutable container for chunk payloads. Chunks arrive one at a
time from `codec.checkpoint.encode_checkpoint`'s `emit` callback; this module
is the sink that catches them.

Layout, from FORMAT.md 5.1::

    0       8   magic       b"SYNPACK\\0"
    8       4   version     u32 LE
    12      4   flags       u32 LE, bit0 = dictionary present
    16      4   count       u32 LE, number of records
    20      32  dict_hash   blake3 of the dictionary object, or 32 zero bytes
    52      ..  records     count x record
    EOF-32  32  trailer     blake3 of bytes [0, EOF-32)

    record: 32 content_hash | 4 stored_len | 4 plain_len | N payload

Two things about that layout are worth understanding before reading the code,
because they explain most of it:

**The 40-byte record header is deliberately redundant with the index.** It
makes a pack self-describing, so a lost or corrupt `.idx` can be rebuilt by a
linear scan (`scan_pack` below) rather than by re-encoding the checkpoint. At
1 MB chunks it costs 0.004%. It is also what makes the index testable: an
index lookup can be checked against a linear scan, instead of against itself.

**`count` sits at offset 16, but a streaming writer cannot know it** until
every record has been written. `PackWriter` therefore writes a placeholder,
seeks back at close, and patches it -- which is why this module needs
`atomic_writer` (a seekable file) rather than `atomic_write` (a bytes blob).

This module deliberately knows nothing about `ChunkRecord`, encodings, or
zstd. It moves opaque payloads and their hashes. Anything it understood about
the codec would be a coupling with no benefit.
"""

from __future__ import annotations

import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import blake3

from synapsefs.errors import IntegrityError
from synapsefs.store.atomic import _fsync_dir, atomic_writer

__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_SIZE",
    "RECORD_HEADER_SIZE",
    "TRAILER_SIZE",
    "PackEntry",
    "PackWriter",
    "scan_pack",
]
"""
def has_chunk(hash: str) -> bool
def get_chunk_data_location(hash: str) -> str, int, int:
def pack_chunks(chunks: list[str]) -> bool:
"""

MAGIC = b"SYNPACK\0"
VERSION = 1
FLAG_DICTIONARY = 1

HEADER_SIZE = 52
RECORD_HEADER_SIZE = 40
TRAILER_SIZE = 32
HASH_SIZE = 32
CHECKSUM_SIZE = 8

_ZERO_HASH = b"\0" * HASH_SIZE
_HEADER = struct.Struct("<8sIII32s")     # magic, version, flags, count, dict_hash
_RECORD = struct.Struct("<32sII")        # content_hash, stored_len, plain_len

# Size of the read-back buffer used to compute the trailer. Arbitrary; large
# enough that syscall overhead vanishes, small enough to stay off the radar of
# the peak-RSS metric.
_HASH_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class PackEntry:
    """Everything the index needs to record about one chunk in a pack.

    Produced by `PackWriter` as it writes, and independently by `scan_pack`
    from the pack alone. Those two must agree -- that equality is the pack
    layer's central test.
    """

    content_hash: bytes
    """blake3 of the chunk's *uncompressed* content -- its identity for dedup."""

    offset: int
    """Byte offset of the **payload**, not of the record header.

    FORMAT.md 5.2: the header occupies `[offset - 40, offset)`. Pointing at
    the payload makes the FUSE read path a single `pread(fd, stored_len,
    offset)` with no arithmetic; recovery scans, which do want the headers,
    start at 52 and walk instead.
    """

    stored_len: int
    """Length of the payload as stored (i.e. compressed)."""

    plain_len: int
    """Length after decompression, so a reader can size its buffer up front."""

    checksum: bytes
    """First 8 bytes of blake3 of the **stored** payload bytes.

    Note this hashes different bytes than `content_hash` does, and answers a
    different question. `content_hash` asks "which chunk is this?" and must be
    stable across recompression. `checksum` asks "did these bytes rot on
    disk?" and is checked by `verify` without decompressing anything.
    """


class PackWriter:
    """Streaming pack writer. Use as a context manager::

        with PackWriter(pack_dir, tmp_dir=tmp) as writer:
            encode_checkpoint(target, base, emit=writer.add_record)
        writer.pack_path      # <pack_hash>.pack, named after its own content
        writer.entries        # for index.write_index()

    Nothing is visible on disk until the block exits: the pack is staged in
    `tmp_dir` and renamed into place atomically, so a crash mid-write leaves
    no half-pack for a later run to trip over.

    The final name is only known at close, because a pack is named for the
    hash of its own contents -- which is also why `pack_dir`, not a full path,
    is what you pass in.
    """

    def __init__(self, pack_dir: Path, *, tmp_dir: Path, dict_hash: Optional[bytes] = None):
        if dict_hash is not None and len(dict_hash) != HASH_SIZE:
            raise ValueError(f"dict_hash must be {HASH_SIZE} bytes, got {len(dict_hash)}")
        self.pack_dir = Path(pack_dir)
        self.tmp_dir = Path(tmp_dir)
        self.dict_hash = dict_hash

        self.entries: List[PackEntry] = []
        self.pack_hash: Optional[bytes] = None
        self.pack_path: Optional[Path] = None

        self._file = None
        self._cm = None
        self._offset = HEADER_SIZE
        self._closed = False
        # Unique, so two concurrent commits cannot stage onto each other.
        self._staging_name = f".incoming-{uuid.uuid4().hex}.pack"

    def __enter__(self) -> "PackWriter":
        self._cm = atomic_writer(self._staging_path(), tmp_dir=self.tmp_dir)
        self._file = self._cm.__enter__()
        # Placeholder header. `count` is patched at close; everything else is
        # already final, so only those 4 bytes ever get rewritten.
        self._file.write(self._header_bytes(count=0))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Let atomic_writer unlink the staging file; nothing lands.
            self._cm.__exit__(exc_type, exc, tb)
            return
        self._finish()
        self._cm.__exit__(None, None, None)
        self._rename_to_content_address()

    def add(self, content_hash: bytes, payload: bytes, plain_len: int) -> PackEntry:
        """Append one chunk. Returns the entry the index will need.

        `payload` is not retained after this returns -- it is written straight
        through to the file. That is what keeps a pack writer's memory flat
        regardless of how large the pack grows; only the ~56-byte `PackEntry`
        per chunk accumulates.
        """
        if self._closed:
            raise ValueError("pack writer is closed")
        if len(content_hash) != HASH_SIZE:
            raise ValueError(f"content_hash must be {HASH_SIZE} bytes, got {len(content_hash)}")

        stored_len = len(payload)
        self._file.write(_RECORD.pack(content_hash, stored_len, plain_len))
        payload_offset = self._offset + RECORD_HEADER_SIZE
        self._file.write(payload)
        self._offset = payload_offset + stored_len

        entry = PackEntry(
            content_hash=content_hash,
            offset=payload_offset,
            stored_len=stored_len,
            plain_len=plain_len,
            checksum=blake3.blake3(payload).digest()[:CHECKSUM_SIZE],
        )
        self.entries.append(entry)
        return entry

    def add_record(self, record) -> PackEntry:
        """Adapter for `codec.checkpoint.ChunkRecord`, usable directly as that
        module's `emit` callback.

        Duck-typed on the three attributes it reads rather than importing
        `ChunkRecord`, so `pack` stays independent of `codec`.
        """
        return self.add(record.content_hash, record.payload, record.plain_len)

    # -- internals ---------------------------------------------------------

    def _staging_path(self) -> Path:
        # atomic_writer needs *a* destination, but the real name depends on
        # content we have not written yet. Land it under a provisional name
        # and rename once the hash is known; both names are in pack_dir, so
        # the second rename is same-filesystem and atomic too.
        return self.pack_dir / self._staging_name

    def _header_bytes(self, count: int) -> bytes:
        flags = FLAG_DICTIONARY if self.dict_hash is not None else 0
        return _HEADER.pack(MAGIC, VERSION, flags, count, self.dict_hash or _ZERO_HASH)

    def _finish(self) -> None:
        self._closed = True
        # Patch the record count now that it is known.
        self._file.seek(0)
        self._file.write(self._header_bytes(count=len(self.entries)))
        self._file.seek(0, 2)

        # The trailer covers [0, EOF-32), which includes the header we just
        # patched. blake3 is sequential, so it cannot be accumulated during
        # writing and then have a header prepended -- the file has to be read
        # back. One extra sequential pass, almost certainly served from page
        # cache, and it has a genuine upside: it hashes the bytes that really
        # landed on disk rather than the bytes we believed we wrote.
        self._file.flush()
        self._file.seek(0)
        digest = blake3.blake3()
        while True:
            block = self._file.read(_HASH_BLOCK)
            if not block:
                break
            digest.update(block)
        self.pack_hash = digest.digest()
        self._file.seek(0, 2)
        self._file.write(self.pack_hash)

    def _rename_to_content_address(self) -> None:
        assert self.pack_hash is not None
        # A pack is named for the hash of its own contents, which is only
        # known once every record is written -- hence a second rename on top
        # of the one atomic_writer already did. Both names live in pack_dir,
        # so this rename is same-filesystem and atomic too; the directory
        # fsync is what makes it survive a power loss, same as the first.
        self.pack_path = self.pack_dir / f"{self.pack_hash.hex()}.pack"
        os.rename(self._staging_path(), self.pack_path)
        _fsync_dir(self.pack_dir)


def read_pack_header(path: Path) -> Tuple[int, int, int, bytes]:
    """Return `(version, flags, count, dict_hash)` from a pack's header.

    Raises IntegrityError if the magic or version is wrong -- a file that is
    not the pack it claims to be is a corruption-class failure (CLI.md 1.3
    exit 4), not a usage error.
    """
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise IntegrityError(f"{path}: truncated pack header ({len(raw)} bytes)")
    magic, version, flags, count, dict_hash = _HEADER.unpack(raw)
    if magic != MAGIC:
        raise IntegrityError(f"{path}: bad pack magic {magic!r}")
    if version != VERSION:
        raise IntegrityError(f"{path}: unsupported pack version {version}")
    return version, flags, count, dict_hash


def scan_pack(path: Path) -> List[PackEntry]:
    """Rebuild every `PackEntry` by walking the pack linearly.

    This is the recovery path for a lost or damaged `.idx` -- the reason the
    40-byte record headers exist at all. It is also the oracle the index tests
    check against, which is why it is written independently of `PackWriter`
    rather than sharing code with it: two implementations that agree are
    evidence, one implementation checked against itself is not.

    Raises IntegrityError on a truncated or malformed pack.
    """
    path = Path(path)
    _version, _flags, count, _dict_hash = read_pack_header(path)
    size = path.stat().st_size
    entries: List[PackEntry] = []

    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        offset = HEADER_SIZE
        for i in range(count):
            header = f.read(RECORD_HEADER_SIZE)
            if len(header) < RECORD_HEADER_SIZE:
                raise IntegrityError(
                    f"{path}: truncated record header for record {i} of {count}"
                )
            content_hash, stored_len, plain_len = _RECORD.unpack(header)
            payload_offset = offset + RECORD_HEADER_SIZE
            if payload_offset + stored_len > size - TRAILER_SIZE:
                raise IntegrityError(
                    f"{path}: record {i} payload runs past the end of the pack"
                )
            payload = f.read(stored_len)
            entries.append(
                PackEntry(
                    content_hash=content_hash,
                    offset=payload_offset,
                    stored_len=stored_len,
                    plain_len=plain_len,
                    checksum=blake3.blake3(payload).digest()[:CHECKSUM_SIZE],
                )
            )
            offset = payload_offset + stored_len

        if offset != size - TRAILER_SIZE:
            raise IntegrityError(
                f"{path}: {size - TRAILER_SIZE - offset} unaccounted bytes between "
                f"the last record and the trailer"
            )
    return entries


def verify_pack(path: Path) -> bytes:
    """Recompute a pack's trailer and check it matches what is stored.

    Returns the pack hash on success. Raises IntegrityError on mismatch --
    this is precisely the bit-rot / tampering check CLI.md reserves exit 4
    for.

    Separate from opening a pack on purpose: this reads the whole file, so it
    belongs in `verify`, never in the read hot path.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < HEADER_SIZE + TRAILER_SIZE:
        raise IntegrityError(f"{path}: too small to be a pack ({size} bytes)")

    digest = blake3.blake3()
    remaining = size - TRAILER_SIZE
    with open(path, "rb") as f:
        while remaining:
            block = f.read(min(_HASH_BLOCK, remaining))
            if not block:
                raise IntegrityError(f"{path}: unexpected EOF while hashing")
            digest.update(block)
            remaining -= len(block)
        stored_trailer = f.read(TRAILER_SIZE)

    computed = digest.digest()
    if computed != stored_trailer:
        raise IntegrityError(
            f"{path}: trailer mismatch -- stored {stored_trailer.hex()[:16]}..., "
            f"computed {computed.hex()[:16]}... (pack is corrupt or was tampered with)"
        )
    return computed
