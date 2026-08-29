"""Tests for ObjectStore: put/get/has, content-addressed dedup, and startup
GC of stale temp files.

Run:
    pytest tests/test_objectstore.py -v
"""

from __future__ import annotations

from pathlib import Path

import blake3
import pytest

from synapsefs.errors import ObjectNotFoundError
from synapsefs.store.objectstore import ObjectStore


def test_put_then_get_roundtrips(tmp_path: Path):
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"hello")
    assert store.get(h) == b"hello"


def test_has_reflects_reality(tmp_path: Path):
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"data")
    assert store.has(h)
    assert not store.has("0" * 64)


def test_same_content_dedupes(tmp_path: Path):
    """Same bytes -> same hash -> written once. This IS the dedup
    mechanism, not a bolt-on feature -- verify it by checking the returned
    hash is stable and exactly one file exists under that hash's shard."""
    store = ObjectStore(tmp_path / "objects")
    h1 = store.put(b"identical")
    h2 = store.put(b"identical")
    assert h1 == h2
    leaf = tmp_path / "objects" / h1[:2] / h1[2:4]
    assert [p.name for p in leaf.iterdir()] == [h1[4:]]


def test_get_missing_object_raises_typed_error(tmp_path: Path):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(ObjectNotFoundError):
        store.get("f" * 64)


def test_startup_gc_clears_stale_tmp_files(tmp_path: Path):
    """A stray file in objects/tmp/ (simulating a prior crash) should be gone
    once a new ObjectStore is opened over the same directory -- but only after
    it is old enough to be unambiguously abandoned rather than in flight.

    The mtime is backdated rather than the gate being disabled, so this test
    exercises `ObjectStore.__init__`'s real call with its real default.
    See `atomic.gc_tmp_dir` for why the gate exists.
    """
    import os
    import time

    from synapsefs.store.atomic import TMP_MIN_AGE_SECONDS

    objects_dir = tmp_path / "objects"
    tmp_dir = objects_dir / "tmp"
    tmp_dir.mkdir(parents=True)
    stale = tmp_dir / "stale.tmp"
    stale.write_bytes(b"junk")
    old = time.time() - TMP_MIN_AGE_SECONDS - 60
    os.utime(stale, (old, old))

    ObjectStore(objects_dir)  # opening should GC as a side effect

    assert list(tmp_dir.iterdir()) == []


def test_startup_gc_spares_a_concurrent_writers_temp_file(tmp_path: Path):
    """The other half: a *fresh* temp file must survive an ObjectStore open,
    because it probably belongs to a commit that is still running."""
    objects_dir = tmp_path / "objects"
    tmp_dir = objects_dir / "tmp"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "in-flight.tmp").write_bytes(b"partial")

    ObjectStore(objects_dir)

    assert [p.name for p in tmp_dir.iterdir()] == ["in-flight.tmp"]


def test_hash_is_content_addressed_blake3(tmp_path: Path):
    """Pins the hash algorithm against a directly-computed BLAKE3 digest,
    so a future refactor can't silently swap hash algorithms (e.g. to
    SHA-256) without a test noticing -- FileFormat.md requires BLAKE3."""
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"synapsefs")
    assert h == blake3.blake3(b"synapsefs").hexdigest()


def test_sharded_two_levels_with_the_prefix_stripped(tmp_path: Path):
    """Locks in the layout: objects/<ab>/<cd>/<60 hex> (ARCHITECTURE.md 3.3).

    One scheme for every object kind, chunks included, and the filename is the
    hash *minus* the shard prefix -- not the full hash repeated."""
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"shard-me")
    assert (tmp_path / "objects" / h[:2] / h[2:4] / h[4:]).is_file()
    assert not (tmp_path / "objects" / h[:2] / h).exists()


def test_put_at_stores_under_a_caller_supplied_hash(tmp_path: Path):
    """Chunk payloads are named for their *decompressed* stream, so the store
    cannot derive the name from the bytes it is given."""
    store = ObjectStore(tmp_path / "objects")
    name = "ab" * 32
    store.put_at(name, b"compressed-payload")
    assert store.get(name) == b"compressed-payload"
    assert store.has(name)
