"""Tests for the packfile container and its index (docs/FORMAT.md sections 5-6).

The structure of this file follows the one real insight about testing a binary
format: **the index is checked against `scan_pack`, never against itself.** A
pack's 40-byte record headers make it self-describing, so a linear walk can
rebuild every entry independently of how the index was written. Two
implementations that agree is evidence; one implementation compared to its own
output is not.
"""

from __future__ import annotations

from pathlib import Path

import blake3
import pytest

from synapsefs.errors import IntegrityError
from synapsefs.pack.index import ARRAYS_START, HEADER_SIZE as IDX_HEADER, PackIndex, write_index
from synapsefs.pack.pack import (
    HEADER_SIZE,
    RECORD_HEADER_SIZE,
    TRAILER_SIZE,
    PackWriter,
    read_pack_header,
    scan_pack,
    verify_pack,
)


@pytest.fixture
def dirs(tmp_path: Path):
    pack_dir = tmp_path / "pack"
    tmp_dir = tmp_path / "tmp"
    pack_dir.mkdir()
    tmp_dir.mkdir()
    return pack_dir, tmp_dir


def make_payloads(n: int, seed: int = 0) -> list[bytes]:
    """Distinct payloads with distinct hashes, deterministic across runs."""
    return [f"chunk-{seed}-{i}".encode() * (3 + i % 7) for i in range(n)]


def build_pack(pack_dir: Path, tmp_dir: Path, payloads: list[bytes]) -> PackWriter:
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        for p in payloads:
            writer.add(blake3.blake3(p).digest(), p, plain_len=len(p) * 2)
    return writer


# ---------------------------------------------------------------------------
# Pack writer / scanner
# ---------------------------------------------------------------------------


def test_writer_and_scanner_agree(dirs):
    """The oracle test. Everything else in this file leans on it."""
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(20))
    assert writer.entries == scan_pack(writer.pack_path)


def test_pack_is_named_for_its_own_content_hash(dirs):
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(3))
    assert writer.pack_path.name == f"{writer.pack_hash.hex()}.pack"
    # ...and the same bytes always produce the same name.
    other = build_pack(pack_dir, tmp_dir, make_payloads(3))
    assert other.pack_hash == writer.pack_hash


def test_trailer_verifies_and_equals_the_pack_hash(dirs):
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(5))
    assert verify_pack(writer.pack_path) == writer.pack_hash


def test_a_single_flipped_byte_is_detected(dirs):
    """Bit-rot detection is a graded requirement (PS module 2a), and this is
    the check that has to catch it."""
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(5))
    raw = bytearray(writer.pack_path.read_bytes())
    raw[HEADER_SIZE + RECORD_HEADER_SIZE + 2] ^= 0x01  # one bit, inside a payload
    writer.pack_path.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError, match="trailer mismatch"):
        verify_pack(writer.pack_path)


def test_offsets_point_at_the_payload_not_the_record_header(dirs):
    """FORMAT.md 5.2. Getting this backwards makes every FUSE read return 40
    bytes of record header followed by truncated payload -- and still 'work'
    often enough to be missed."""
    pack_dir, tmp_dir = dirs
    payloads = make_payloads(6)
    writer = build_pack(pack_dir, tmp_dir, payloads)
    with open(writer.pack_path, "rb") as f:
        for entry, expected in zip(writer.entries, payloads):
            f.seek(entry.offset)
            assert f.read(entry.stored_len) == expected
            # The header is immediately before it.
            f.seek(entry.offset - RECORD_HEADER_SIZE)
            assert f.read(32) == entry.content_hash


def test_checksum_covers_stored_bytes_not_uncompressed_content(dirs):
    """`content_hash` and `checksum` deliberately hash different things --
    identity vs. bit-rot. A payload whose 'plain' content differs from its
    stored bytes must still checksum over the stored bytes."""
    pack_dir, tmp_dir = dirs
    payload = b"stored-bytes-here"
    content_hash = blake3.blake3(b"totally different uncompressed content").digest()
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        entry = writer.add(content_hash, payload, plain_len=999)
    assert entry.content_hash == content_hash
    assert entry.checksum == blake3.blake3(payload).digest()[:8]


def test_empty_pack_is_valid(dirs):
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, [])
    assert writer.entries == [] == scan_pack(writer.pack_path)
    assert verify_pack(writer.pack_path) == writer.pack_hash
    assert writer.pack_path.stat().st_size == HEADER_SIZE + TRAILER_SIZE


def test_header_records_the_count(dirs):
    """The count is written as a placeholder and patched at close, because a
    streaming writer cannot know it up front."""
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(11))
    _version, flags, count, dict_hash = read_pack_header(writer.pack_path)
    assert count == 11 and flags == 0 and dict_hash == b"\0" * 32


def test_dictionary_flag_and_hash_round_trip(dirs):
    pack_dir, tmp_dir = dirs
    dict_hash = blake3.blake3(b"a dictionary object").digest()
    with PackWriter(pack_dir, tmp_dir=tmp_dir, dict_hash=dict_hash) as writer:
        writer.add(blake3.blake3(b"x").digest(), b"x", plain_len=1)
    _v, flags, _count, stored = read_pack_header(writer.pack_path)
    assert flags & 1 and stored == dict_hash


def test_failure_mid_write_leaves_no_pack_behind(dirs):
    """Crash safety (PS module 2h): a partial pack must never become visible."""
    pack_dir, tmp_dir = dirs
    with pytest.raises(RuntimeError, match="boom"):
        with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
            writer.add(blake3.blake3(b"a").digest(), b"a", plain_len=1)
            raise RuntimeError("boom")
    assert list(pack_dir.iterdir()) == []
    assert list(tmp_dir.iterdir()) == []


def test_add_after_close_is_rejected(dirs):
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(2))
    with pytest.raises(ValueError, match="closed"):
        writer.add(blake3.blake3(b"z").digest(), b"z", plain_len=1)


def test_a_file_that_is_not_a_pack_is_rejected(tmp_path):
    bogus = tmp_path / "nope.pack"
    bogus.write_bytes(b"NOTAPACK" + b"\0" * 100)
    with pytest.raises(IntegrityError, match="bad pack magic"):
        read_pack_header(bogus)


def test_truncated_pack_is_rejected(dirs):
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, make_payloads(4))
    raw = writer.pack_path.read_bytes()
    writer.pack_path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(IntegrityError):
        scan_pack(writer.pack_path)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def build_indexed(pack_dir: Path, tmp_dir: Path, payloads: list[bytes]):
    writer = build_pack(pack_dir, tmp_dir, payloads)
    idx = write_index(
        pack_dir / f"{writer.pack_hash.hex()}.idx",
        pack_hash=writer.pack_hash,
        entries=writer.entries,
        tmp_dir=tmp_dir,
    )
    return writer, idx


def test_every_chunk_in_the_pack_is_findable(dirs):
    """Checked against `scan_pack`, not against the writer's own entries."""
    pack_dir, tmp_dir = dirs
    writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(200))
    truth = scan_pack(writer.pack_path)
    with PackIndex(idx_path) as idx:
        assert len(idx) == len(truth)
        assert idx.pack_hash == writer.pack_hash
        for entry in truth:
            assert idx.lookup(entry.content_hash) == entry


def test_absent_hashes_return_none(dirs):
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(50))
    with PackIndex(idx_path) as idx:
        for i in range(50):
            missing = blake3.blake3(f"not-in-pack-{i}".encode()).digest()
            assert idx.lookup(missing) is None


def test_hashes_are_stored_in_ascending_order(dirs):
    """The binary search is only correct if this holds, and entries arrive in
    tensor order rather than hash order."""
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(100))
    with PackIndex(idx_path) as idx:
        hashes = [e.content_hash for e in idx.entries()]
    assert hashes == sorted(hashes)


def test_fanout_is_cumulative_and_consistent(dirs):
    """`fanout[b]` counts entries with first byte <= b, not == b. An
    off-by-one here yields an index that finds most hashes and misses a few."""
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(300))
    with PackIndex(idx_path) as idx:
        first_bytes = [e.content_hash[0] for e in idx.entries()]
        for b in range(256):
            assert idx._fanout_at(b) == sum(1 for fb in first_bytes if fb <= b)
        assert idx._fanout_at(255) == len(idx)


def test_lookup_handles_the_fanout_boundaries(dirs):
    """First byte 0x00 takes the `lo = 0` branch and 0xFF takes the last
    bucket; both are the places an off-by-one hides."""
    pack_dir, tmp_dir = dirs
    payloads, seen = [], set()
    i = 0
    while seen != {0x00, 0xFF}:
        p = f"seek-{i}".encode()
        b = blake3.blake3(p).digest()[0]
        if b in (0x00, 0xFF):
            payloads.append(p)
            seen.add(b)
        i += 1
    payloads.extend(make_payloads(40, seed=9))

    writer, idx_path = build_indexed(pack_dir, tmp_dir, payloads)
    with PackIndex(idx_path) as idx:
        for entry in scan_pack(writer.pack_path):
            assert idx.lookup(entry.content_hash) == entry


def test_index_entries_match_a_linear_scan_exactly(dirs):
    pack_dir, tmp_dir = dirs
    writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(64))
    with PackIndex(idx_path) as idx:
        assert idx.entries() == sorted(scan_pack(writer.pack_path),
                                       key=lambda e: e.content_hash)


def test_empty_index_is_valid(dirs):
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, [])
    with PackIndex(idx_path) as idx:
        assert len(idx) == 0
        assert idx.lookup(b"\0" * 32) is None
        idx.verify()


def test_duplicate_hashes_are_rejected(dirs):
    """A duplicated key in a binary-searched array produces plausible-but-wrong
    lookups rather than an error, so it has to be caught at write time."""
    pack_dir, tmp_dir = dirs
    writer = build_pack(pack_dir, tmp_dir, [b"same", b"same"])
    with pytest.raises(ValueError, match="duplicate chunk hash"):
        write_index(pack_dir / "x.idx", pack_hash=writer.pack_hash,
                    entries=writer.entries, tmp_dir=tmp_dir)


def test_index_trailer_detects_corruption(dirs):
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(30))
    with PackIndex(idx_path) as idx:
        idx.verify()
    raw = bytearray(idx_path.read_bytes())
    raw[ARRAYS_START + 5] ^= 0x01
    idx_path.write_bytes(bytes(raw))
    with PackIndex(idx_path) as idx:
        with pytest.raises(IntegrityError, match="index trailer mismatch"):
            idx.verify()


def test_index_size_must_match_its_count(dirs):
    """Cheap structural check at open time -- catches a truncated index
    without reading the whole file."""
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(10))
    raw = idx_path.read_bytes()
    idx_path.write_bytes(raw[:-40])
    with pytest.raises(IntegrityError, match="does not match"):
        PackIndex(idx_path)


def test_a_file_that_is_not_an_index_is_rejected(tmp_path):
    bogus = tmp_path / "nope.idx"
    bogus.write_bytes(b"NOTANIDX" + b"\0" * 2000)
    with pytest.raises(IntegrityError, match="bad index magic"):
        PackIndex(bogus)


def test_lookup_rejects_a_wrong_length_hash(dirs):
    pack_dir, tmp_dir = dirs
    _writer, idx_path = build_indexed(pack_dir, tmp_dir, make_payloads(4))
    with PackIndex(idx_path) as idx:
        with pytest.raises(ValueError, match="must be 32 bytes"):
            idx.lookup(b"\x00" * 16)


# ---------------------------------------------------------------------------
# End to end: checkpoint -> pack -> index -> reconstruction
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="fixtures not generated")
@pytest.mark.parametrize("model", ["mlp-tiny", "cnn-tiny"])
def test_commit_then_reconstruct_through_the_pack(dirs, model):
    """The whole storage path, with nothing held in memory between halves.

    Encode a checkpoint straight into a pack, write its index, then rebuild
    every tensor using only the manifests and index lookups -- which is what
    `checkout` and the FUSE read path will both do. Byte-exactness here is the
    PS's core deliverable.
    """
    import numpy as np

    from synapsefs.codec.checkpoint import encode_checkpoint
    from synapsefs.codec.chunk import DELTA, decode_chunk
    from synapsefs.safetensors_io import SafetensorsFile

    pack_dir, tmp_dir = dirs
    base_path = FIXTURES / model / "base" / "model.safetensors"
    target_path = FIXTURES / model / "finetuned" / "model.safetensors"

    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        result = encode_checkpoint(target_path, base_path, emit=writer.add_record)
    idx_path = write_index(
        pack_dir / f"{writer.pack_hash.hex()}.idx",
        pack_hash=writer.pack_hash,
        entries=writer.entries,
        tmp_dir=tmp_dir,
    )

    verify_pack(writer.pack_path)
    with PackIndex(idx_path) as idx, open(writer.pack_path, "rb") as pack, \
            SafetensorsFile(base_path) as base, SafetensorsFile(target_path) as target:
        idx.verify()
        for name, manifest in result.manifests.items():
            rebuilt = []
            for chunk in manifest["chunks"]:
                entry = idx.lookup(bytes.fromhex(chunk["object"]))
                assert entry is not None, f"{name}: chunk missing from index"

                # The FUSE hot path in miniature: one pread at the recorded
                # offset, no arithmetic, no linear scan.
                pack.seek(entry.offset)
                payload = pack.read(entry.stored_len)
                assert blake3.blake3(payload).digest()[:8] == entry.checksum

                base_rows = (
                    base.rows(name, chunk["row_start"], chunk["row_end"] + 1)
                    if chunk["encoding"] == DELTA
                    else None
                )
                rebuilt.append(
                    decode_chunk(chunk["encoding"], payload, base_rows,
                                 dtype=manifest["dtype"])
                )
            assert np.array_equal(np.concatenate(rebuilt), target.whole(name).ravel())
