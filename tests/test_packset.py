"""Tests for repo-level, multi-pack chunk lookup (docs/FORMAT.md section 6.3)."""

from __future__ import annotations

from pathlib import Path

import blake3
import pytest

from synapsefs.errors import IntegrityError
from synapsefs.pack.index import write_index
from synapsefs.pack.pack import PackWriter
from synapsefs.pack.packset import ORDER_FILENAME, PackSet


@pytest.fixture
def dirs(tmp_path: Path):
    pack_dir, tmp_dir = tmp_path / "pack", tmp_path / "tmp"
    pack_dir.mkdir()
    tmp_dir.mkdir()
    return pack_dir, tmp_dir


def commit_pack(pack_dir: Path, tmp_dir: Path, payloads: list[bytes]) -> bytes:
    """Write one pack + index and register nothing; returns its pack hash."""
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        for p in payloads:
            writer.add(blake3.blake3(p).digest(), p, plain_len=len(p) * 2)
    write_index(
        pack_dir / f"{writer.pack_hash.hex()}.idx",
        pack_hash=writer.pack_hash,
        entries=writer.entries,
        tmp_dir=tmp_dir,
    )
    return writer.pack_hash


def h(payload: bytes) -> bytes:
    return blake3.blake3(payload).digest()


def test_finds_chunks_across_several_packs(dirs):
    pack_dir, tmp_dir = dirs
    groups = [[b"a1", b"a2"], [b"b1", b"b2", b"b3"], [b"c1"]]
    hashes = [commit_pack(pack_dir, tmp_dir, g) for g in groups]

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        for ph in hashes:
            packs.register(ph)
        assert len(packs) == 3
        for group, ph in zip(groups, hashes):
            for payload in group:
                located = packs.lookup(h(payload))
                assert located is not None
                assert located.pack_hash == ph
                assert packs.read(h(payload)) == payload


def test_missing_chunk_returns_none(dirs):
    pack_dir, tmp_dir = dirs
    ph = commit_pack(pack_dir, tmp_dir, [b"present"])
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(ph)
        assert packs.lookup(h(b"absent")) is None
        assert packs.read(h(b"absent")) is None
        assert not packs.has(h(b"absent"))
        assert packs.has(h(b"present"))


def test_newest_pack_is_probed_first(dirs):
    """A checkout of HEAD mostly wants the newest commit's chunks, so the
    search should usually terminate on the first index. When the same content
    exists in two packs, the newest one must win."""
    pack_dir, tmp_dir = dirs
    old = commit_pack(pack_dir, tmp_dir, [b"shared", b"only-old"])
    new = commit_pack(pack_dir, tmp_dir, [b"shared", b"only-new"])

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(old)
        packs.register(new)          # registered last => probed first
        assert packs.pack_hashes() == [new, old]
        assert packs.lookup(h(b"shared")).pack_hash == new
        assert packs.lookup(h(b"only-old")).pack_hash == old


def test_order_file_survives_reopen(dirs):
    pack_dir, tmp_dir = dirs
    first = commit_pack(pack_dir, tmp_dir, [b"x"])
    second = commit_pack(pack_dir, tmp_dir, [b"y"])
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(first)
        packs.register(second)
    assert (pack_dir / ORDER_FILENAME).read_text().split() == [second.hex(), first.hex()]

    with PackSet(pack_dir, tmp_dir=tmp_dir) as reopened:
        assert reopened.pack_hashes() == [second, first]


def test_a_pack_missing_from_order_is_still_found(dirs):
    """A pack is renamed into place before `order` is rewritten, so a crash
    between those two steps leaves an index nothing points at. Probe order is
    a performance heuristic; findability is not negotiable."""
    pack_dir, tmp_dir = dirs
    registered = commit_pack(pack_dir, tmp_dir, [b"in-order"])
    orphan = commit_pack(pack_dir, tmp_dir, [b"not-in-order"])

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(registered)   # `orphan` deliberately never registered

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert len(packs) == 2
        assert packs.has(h(b"not-in-order"))
        assert packs.lookup(h(b"not-in-order")).pack_hash == orphan


def test_a_stale_order_entry_is_ignored(dirs):
    """`order` naming a pack that no longer exists (deleted by a repack) must
    not break startup."""
    pack_dir, tmp_dir = dirs
    ph = commit_pack(pack_dir, tmp_dir, [b"real"])
    (pack_dir / ORDER_FILENAME).write_text("00" * 32 + "\n" + ph.hex() + "\n")

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert packs.pack_hashes() == [ph]
        assert packs.has(h(b"real"))


def test_empty_repo_has_no_packs(dirs):
    pack_dir, tmp_dir = dirs
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert len(packs) == 0
        assert packs.lookup(h(b"anything")) is None


def test_read_detects_a_rotted_payload(dirs):
    """The stored 8-byte checksum is how bit-rot is caught without
    decompressing (FORMAT.md 12)."""
    pack_dir, tmp_dir = dirs
    ph = commit_pack(pack_dir, tmp_dir, [b"some payload bytes"])
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(ph)
        located = packs.lookup(h(b"some payload bytes"))

    raw = bytearray((pack_dir / f"{ph.hex()}.pack").read_bytes())
    raw[located.entry.offset] ^= 0x01
    (pack_dir / f"{ph.hex()}.pack").write_bytes(bytes(raw))

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        with pytest.raises(IntegrityError, match="failed its stored checksum"):
            packs.read(h(b"some payload bytes"))
        # ...and the check is skippable for a caller that has benchmarked it.
        assert packs.read(h(b"some payload bytes"), verify_checksum=False) is not None


def test_registering_a_pack_with_no_index_is_refused(dirs):
    pack_dir, tmp_dir = dirs
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        with pytest.raises(IntegrityError, match="no index at"):
            packs.register(b"\xab" * 32)


def test_verify_checks_every_index(dirs):
    pack_dir, tmp_dir = dirs
    a = commit_pack(pack_dir, tmp_dir, [b"p", b"q"])
    b = commit_pack(pack_dir, tmp_dir, [b"r"])
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(a)
        packs.register(b)
        packs.verify()

    idx = pack_dir / f"{b.hex()}.idx"
    raw = bytearray(idx.read_bytes())
    raw[1080] ^= 0x01
    idx.write_bytes(bytes(raw))
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        with pytest.raises(IntegrityError, match="trailer mismatch"):
            packs.verify()


def test_cross_commit_dedup_actually_works(dirs):
    """The point of this module. `has` is `encode_checkpoint`'s `already_have`
    predicate; without it every commit re-stores chunks it shares with its
    parent."""
    import numpy as np
    from safetensors.numpy import save_file

    from synapsefs.codec.checkpoint import encode_checkpoint

    pack_dir, tmp_dir = dirs
    rng = np.random.default_rng(0)
    frozen = rng.standard_normal((32, 32)).astype(np.float16)

    ckpt1 = tmp_dir.parent / "c1.safetensors"
    ckpt2 = tmp_dir.parent / "c2.safetensors"
    save_file({"frozen": frozen, "head": rng.standard_normal((8, 8)).astype(np.float16)},
              str(ckpt1))
    save_file({"frozen": frozen, "head": rng.standard_normal((8, 8)).astype(np.float16)},
              str(ckpt2))

    # Commit 1: root, everything is new.
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as w1:
        r1 = encode_checkpoint(ckpt1, None, emit=w1.add_record)
    write_index(pack_dir / f"{w1.pack_hash.hex()}.idx", pack_hash=w1.pack_hash,
                entries=w1.entries, tmp_dir=tmp_dir)
    assert r1.chunks_new == 2 and r1.chunks_deduped == 0

    # Commit 2: root again, but `frozen` is byte-identical to commit 1's.
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        packs.register(w1.pack_hash)
        with PackWriter(pack_dir, tmp_dir=tmp_dir) as w2:
            r2 = encode_checkpoint(ckpt2, None, emit=w2.add_record,
                                   already_have=packs.has)

    assert r2.chunks_deduped == 1, "the unchanged tensor should not be re-stored"
    assert r2.chunks_new == 1
    assert len(w2.entries) == 1
    # The manifest still references the deduped chunk -- dedup changes storage,
    # never the manifest -- and that chunk resolves through the older pack.
    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        obj = bytes.fromhex(r2.manifests["frozen"]["chunks"][0]["object"])
        assert packs.lookup(obj).pack_hash == w1.pack_hash


# ---------------------------------------------------------------------------
# Startup recovery (FORMAT.md 9.5)
# ---------------------------------------------------------------------------


def test_a_sealed_pack_with_no_index_is_rebuilt_not_lost(dirs):
    """A crash between sealing a pack and writing its index must not lose data.

    The `.pack` is authoritative; the `.idx` is derived and regenerable by
    rescanning from offset 52. Before this, such a pack was not merely degraded
    -- `PackSet` enumerates `*.idx`, so it was *invisible*, and every chunk in
    it reported as absent.
    """
    pack_dir, tmp_dir = dirs
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        writer.add(h(b"committed"), b"committed", plain_len=9)
    # deliberately no write_index()

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert any("index rebuilt" in n for n in packs.recovery_notes)
        assert len(packs) == 1
        assert packs.read(h(b"committed")) == b"committed"


def test_a_corrupt_index_is_rebuilt(dirs):
    pack_dir, tmp_dir = dirs
    ph = commit_pack(pack_dir, tmp_dir, [b"alpha", b"beta"])
    (pack_dir / f"{ph.hex()}.idx").write_bytes(b"garbage" * 200)

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert any("rebuilding" in n for n in packs.recovery_notes)
        assert packs.read(h(b"alpha")) == b"alpha"


def test_an_unsealed_pack_is_ignored_not_deleted(dirs):
    """A pack whose trailer does not verify was never sealed -- a crashed
    partial write. No ref can reference it (refs are written last), so ignoring
    it loses nothing. It is not deleted: removing data during recovery is how a
    bug becomes data loss."""
    pack_dir, tmp_dir = dirs
    ph = commit_pack(pack_dir, tmp_dir, [b"x"])
    pack_path = pack_dir / f"{ph.hex()}.pack"
    (pack_dir / f"{ph.hex()}.idx").unlink()
    raw = bytearray(pack_path.read_bytes())
    raw[-1] ^= 0xFF                      # break the trailer
    pack_path.write_bytes(bytes(raw))

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        assert any("not sealed" in n for n in packs.recovery_notes)
        assert len(packs) == 0
    assert pack_path.exists(), "recovery must not delete data"


def test_repair_can_be_disabled(dirs):
    pack_dir, tmp_dir = dirs
    with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
        writer.add(h(b"y"), b"y", plain_len=1)
    with PackSet(pack_dir, tmp_dir=tmp_dir, repair=False) as packs:
        assert packs.recovery_notes == [] and len(packs) == 0
