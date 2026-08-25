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
    shard_dir = tmp_path / "objects" / h1[:2]
    assert [p.name for p in shard_dir.iterdir()] == [h1]


def test_get_missing_object_raises_typed_error(tmp_path: Path):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(ObjectNotFoundError):
        store.get("f" * 64)


def test_startup_gc_clears_stale_tmp_files(tmp_path: Path):
    """A stray file in objects/tmp/ (simulating a prior crash) should be
    gone the moment a new ObjectStore is opened over the same directory."""
    objects_dir = tmp_path / "objects"
    tmp_dir = objects_dir / "tmp"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "stale.tmp").write_bytes(b"junk")

    ObjectStore(objects_dir)  # opening should GC as a side effect

    assert list(tmp_dir.iterdir()) == []


def test_hash_is_content_addressed_blake3(tmp_path: Path):
    """Pins the hash algorithm against a directly-computed BLAKE3 digest,
    so a future refactor can't silently swap hash algorithms (e.g. to
    SHA-256) without a test noticing -- FileFormat.md requires BLAKE3."""
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"synapsefs")
    assert h == blake3.blake3(b"synapsefs").hexdigest()


def test_sharded_by_first_two_hex_chars(tmp_path: Path):
    """Locks in the exact sharding scheme from FileFormat.md ~3."""
    store = ObjectStore(tmp_path / "objects")
    h = store.put(b"shard-me")
    expected_path = tmp_path / "objects" / h[:2] / h
    assert expected_path.is_file()
