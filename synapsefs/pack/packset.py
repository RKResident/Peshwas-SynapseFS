"""Repo-level chunk lookup across every pack (docs/FORMAT.md section 6.3).

`PackIndex` answers *"is this chunk in **this** pack?"*. This module answers
the question the rest of the system actually asks: *"which pack, if any, has
this chunk?"*

That distinction exists because packs are **immutable once sealed** -- a pack
is named for the hash of its own contents, and `verify` depends on that. So a
new commit cannot append to an existing pack; it writes its own. FORMAT.md
section 5 makes this explicit (one storage pack per commit, plus transfer
packs built on demand for push/pull), which means a repo accumulates packs and
any given chunk may live in any of them.

Two things follow, and both are the point of this module:

**Cross-commit dedup becomes possible.** `codec.checkpoint.encode_checkpoint`
takes an `already_have(hash)` predicate meaning "the store already holds this
chunk from an earlier commit". `PackSet.has` is that predicate. Without it,
every commit re-stores chunks it shares with its parent and `chunks_deduped`
only ever counts duplicates *within* one checkpoint.

**Probe order matters.** Indexes are probed newest-pack-first, because a
checkout of `HEAD` mostly wants chunks the newest commit introduced, so the
search usually terminates on the first index. The order lives in
`objects/pack/order`, newline-separated pack hashes, newest first.

`OPEN QUESTION` (FORMAT.md 6.3) -- probe cost grows linearly in pack count.
At ~32 packs this wants either a repack trigger or a per-pack bloom filter.
Nothing here is structured to prevent either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import blake3

from synapsefs.errors import IntegrityError
from synapsefs.pack.index import PackIndex
from synapsefs.pack.pack import PackEntry
from synapsefs.store.atomic import atomic_write

__all__ = ["ORDER_FILENAME", "Located", "PackSet"]

ORDER_FILENAME = "order"


@dataclass(frozen=True)
class Located:
    """Where a chunk lives: which pack, and where inside it."""

    pack_hash: bytes
    pack_path: Path
    entry: PackEntry


class PackSet:
    """Every pack in a repo, opened for lookup. Context manager; also `.close()`.

    Construction mmaps each `.idx` -- cheap, and near-zero resident, since an
    index is designed to be searched in place. Pack files themselves are
    opened lazily, on the first read that needs one, so a lookup-only workload
    (which is what `already_have` is) never opens a single pack.
    """

    def __init__(self, pack_dir: Path, *, tmp_dir: Optional[Path] = None):
        self.pack_dir = Path(pack_dir)
        # Defaults to the sibling `objects/tmp/`, which is where every other
        # atomic write in the repo stages. Must be on the same filesystem.
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else self.pack_dir.parent / "tmp"

        self._indexes: List[PackIndex] = []
        self._pack_files: Dict[bytes, object] = {}
        self._closed = False
        self._open_indexes()

    # -- construction ------------------------------------------------------

    def _read_order(self) -> List[str]:
        order_path = self.pack_dir / ORDER_FILENAME
        if not order_path.is_file():
            return []
        text = order_path.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _open_indexes(self) -> None:
        if not self.pack_dir.is_dir():
            return
        available = {p.stem: p for p in self.pack_dir.glob("*.idx")}

        # Anything named in `order` first, in that order. Then anything on
        # disk that `order` does not mention.
        #
        # That second group is not paranoia: a pack is renamed into place
        # before `order` is rewritten, so a crash between those two steps
        # leaves exactly this state. Such a pack is in fact the *newest*, but
        # it is appended as oldest rather than prepended -- probe order is a
        # performance heuristic, never a correctness property, and guessing
        # wrong here only costs an extra probe. Sorted for determinism.
        ordered = [available.pop(h) for h in self._read_order() if h in available]
        ordered.extend(available[h] for h in sorted(available))

        for path in ordered:
            self._indexes.append(PackIndex(path))

    # -- lookup ------------------------------------------------------------

    def lookup(self, content_hash: bytes) -> Optional[Located]:
        """Find a chunk anywhere in the repo, newest pack first."""
        self._check_open()
        for index in self._indexes:
            entry = index.lookup(content_hash)
            if entry is not None:
                return Located(
                    pack_hash=index.pack_hash,
                    pack_path=self._pack_path(index.pack_hash),
                    entry=entry,
                )
        return None

    def has(self, content_hash: bytes) -> bool:
        """Pass this straight to `encode_checkpoint(already_have=...)`.

        Touches only index hash arrays -- no pack file is opened and no payload
        is read, which is what makes dedup cheap enough to run per chunk.
        """
        return self.lookup(content_hash) is not None

    def read(self, content_hash: bytes, *, verify_checksum: bool = True) -> Optional[bytes]:
        """Return a chunk's stored (still-compressed) payload, or None.

        This is the read hot path in miniature: one index probe, then a single
        `pread` at the recorded offset with no arithmetic, because index
        offsets point at the payload rather than the record header
        (FORMAT.md 5.2).

        `verify_checksum` compares the stored 8-byte blake3 prefix, which is
        how bit-rot is caught without decompressing anything (FORMAT.md 12).
        It is on by default because silently returning rotted bytes is far
        worse than the cost; hashing is comfortably cheaper than the zstd
        decompression that follows. Turn it off only with a benchmark in hand.
        """
        located = self.lookup(content_hash)
        if located is None:
            return None

        handle = self._pack_file(located.pack_hash)
        handle.seek(located.entry.offset)
        payload = handle.read(located.entry.stored_len)
        if len(payload) != located.entry.stored_len:
            raise IntegrityError(
                f"{located.pack_path}: short read for chunk {content_hash.hex()[:16]} "
                f"({len(payload)} of {located.entry.stored_len} bytes)"
            )
        if verify_checksum:
            actual = blake3.blake3(payload).digest()[: len(located.entry.checksum)]
            if actual != located.entry.checksum:
                raise IntegrityError(
                    f"{located.pack_path}: chunk {content_hash.hex()[:16]} failed its "
                    f"stored checksum -- the pack is corrupt or was tampered with"
                )
        return payload

    # -- mutation ----------------------------------------------------------

    def register(self, pack_hash: bytes) -> None:
        """Record a newly written pack as the newest, and open its index.

        Call this *after* `PackWriter` and `write_index` have both completed:
        `order` naming a pack whose index does not exist yet would be a
        dangling reference, whereas an index not yet named in `order` is
        merely un-prioritised (see `_open_indexes`). Ordering the two steps
        this way makes a crash between them harmless in the direction that
        matters.
        """
        self._check_open()
        index_path = self.pack_dir / f"{pack_hash.hex()}.idx"
        if not index_path.is_file():
            raise IntegrityError(
                f"cannot register pack {pack_hash.hex()[:16]}: no index at {index_path}"
            )

        existing = [h for h in self._read_order() if h != pack_hash.hex()]
        order = [pack_hash.hex()] + existing
        atomic_write(
            self.pack_dir / ORDER_FILENAME,
            ("\n".join(order) + "\n").encode("utf-8"),
            tmp_dir=self.tmp_dir,
        )

        # Reopen so the new pack is visible, and is probed first.
        self._close_indexes()
        self._open_indexes()

    def verify(self) -> None:
        """Verify every index's trailer. Reads every index in full, so this
        belongs in `verify`, not in a read path."""
        self._check_open()
        for index in self._indexes:
            index.verify()

    # -- internals ---------------------------------------------------------

    def _pack_path(self, pack_hash: bytes) -> Path:
        return self.pack_dir / f"{pack_hash.hex()}.pack"

    def _pack_file(self, pack_hash: bytes):
        handle = self._pack_files.get(pack_hash)
        if handle is None:
            path = self._pack_path(pack_hash)
            if not path.is_file():
                raise IntegrityError(
                    f"index {pack_hash.hex()[:16]}.idx exists but its pack does not: {path}"
                )
            handle = open(path, "rb")
            self._pack_files[pack_hash] = handle
        return handle

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("pack set is closed")

    def _close_indexes(self) -> None:
        for index in self._indexes:
            index.close()
        self._indexes = []

    def pack_hashes(self) -> List[bytes]:
        """Open packs, in probe order (newest first)."""
        return [index.pack_hash for index in self._indexes]

    def __len__(self) -> int:
        return len(self._indexes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_indexes()
        for handle in self._pack_files.values():
            handle.close()
        self._pack_files = {}

    def __enter__(self) -> "PackSet":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
