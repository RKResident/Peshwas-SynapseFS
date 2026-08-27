"""Tests for lineage verification (PS module 2, CLI.md ~7, FORMAT.md 12B).

Half of these are *tamper* tests, and they are the point of the file. A
verification tool that reports OK on a clean repo proves nothing -- the only
way to know a tier detects what it claims is to break the repo deliberately
and watch it fail, and the only way to know a tier *cannot* detect something
is to break the repo and watch it pass.

The central pair is:

  test_a_repaired_injection_survives_the_fast_tier
  test_a_repaired_injection_is_caught_by_the_default_tier

Together they demonstrate the property the whole design rests on: every
checksum below the manifest level is stored in a file an attacker rewrites
alongside the payload, so only the ref-anchored content hash detects a
substituted block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapsefs import graph, verify as verify_engine
from synapsefs.cli.main import main
from synapsefs.pack.index import write_index
from synapsefs.pack.pack import PackWriter, scan_pack
from synapsefs.store.repo import Repo

from tests.test_graph import commit_series, make_repo


def build(tmp_path, n=4):
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, n)
    return repo_dir, Repo.find(repo_dir)


def run(repo_dir, *args) -> verify_engine.VerifyReport:
    repo = Repo.find(repo_dir)
    return verify_engine.verify_lineage(repo, [repo.resolve_ref("HEAD")], **_kw(args))


def _kw(args):
    kw = {}
    for a in args:
        if a in verify_engine.TIERS:
            kw["tier"] = a
        else:
            kw[a] = True
    return kw


def pack_dir_of(repo_dir) -> Path:
    return Path(repo_dir) / ".synapse" / "objects" / "pack"


def kinds(report) -> set:
    return {f.kind for f in report.failures}


# -- clean -----------------------------------------------------------------


@pytest.mark.parametrize("tier", verify_engine.TIERS)
def test_a_clean_repo_verifies_at_every_tier(tmp_path, tier):
    repo_dir, _ = build(tmp_path)
    report = run(repo_dir, tier)
    assert report.ok, report.failures
    assert report.commits == 4
    assert report.chunks > 0


def test_clean_repo_passes_the_content_hash_tier(tmp_path):
    repo_dir, _ = build(tmp_path)
    report = run(repo_dir, "check_content_hash")
    assert report.ok, report.failures


def test_only_the_content_tier_reads_payloads(tmp_path):
    repo_dir, _ = build(tmp_path)
    assert run(repo_dir, verify_engine.STRUCTURE).bytes_verified == 0
    assert run(repo_dir, verify_engine.CHECKSUM).bytes_verified > 0


def test_shared_chunks_are_verified_once(tmp_path):
    """Dedup and FORMAT.md 4.5 manifest reuse mean references far outnumber
    distinct chunks; the memo is most of why verification stays fast."""
    repo_dir, _ = build(tmp_path, 6)
    report = run(repo_dir)
    assert report.chunks_distinct < report.chunks


# -- tier 1: the object graph ----------------------------------------------


def test_a_tampered_tensor_manifest_is_caught_by_shallow(tmp_path):
    repo_dir, repo = build(tmp_path)
    head = graph.get_json(repo.store, repo.resolve_ref("HEAD"))
    manifest_hash = next(iter(
        graph.get_json(repo.store, head["checkpoint_manifest"])["tensors"].values()
    ))
    path = repo.objects_dir / manifest_hash[:2] / manifest_hash
    blob = json.loads(path.read_bytes())
    blob["shape"] = [1, 1]
    path.write_bytes(json.dumps(blob).encode())

    report = run(repo_dir, verify_engine.STRUCTURE)
    assert not report.ok
    assert kinds(report) == {"object-hash-mismatch"}


def test_a_deleted_object_is_reported_not_raised(tmp_path):
    """A missing object must become a failure record, so the walk continues and
    reports every problem rather than aborting on the first."""
    repo_dir, repo = build(tmp_path)
    head = graph.get_json(repo.store, repo.resolve_ref("HEAD"))
    target = head["checkpoint_manifest"]
    (repo.objects_dir / target[:2] / target).unlink()

    report = run(repo_dir, verify_engine.STRUCTURE)
    assert not report.ok
    assert "missing-object" in kinds(report)


def test_a_chunk_in_no_pack_is_a_broken_link(tmp_path):
    repo_dir, _ = build(tmp_path)
    for path in pack_dir_of(repo_dir).iterdir():
        if path.suffix in (".pack", ".idx"):
            path.unlink()

    report = run(repo_dir, verify_engine.STRUCTURE)
    assert not report.ok
    assert kinds(report) == {"missing-chunk"}


# -- tier 2: bit-rot -------------------------------------------------------


def flip_a_payload_byte(repo_dir) -> tuple:
    """Corrupt one byte of one chunk payload, repairing nothing. This is rot,
    not an attack: the index and both trailers now disagree with the pack."""
    pack = sorted(pack_dir_of(repo_dir).glob("*.pack"))[0]
    entry = scan_pack(pack)[0]
    data = bytearray(pack.read_bytes())
    data[entry.offset] ^= 0xFF
    pack.write_bytes(bytes(data))
    return pack, entry


def test_bit_rot_is_caught_by_the_fast_tier(tmp_path):
    repo_dir, _ = build(tmp_path)
    flip_a_payload_byte(repo_dir)
    report = run(repo_dir, verify_engine.CHECKSUM)
    assert not report.ok
    assert "chunk-checksum" in kinds(report)


def test_bit_rot_is_invisible_to_the_shallow_tier(tmp_path):
    """Documents the boundary: shallow touches no payload, so it cannot see
    rot. Asserted rather than assumed, because a tier that silently started
    reading payloads would make `--shallow`'s cost claim false."""
    repo_dir, _ = build(tmp_path)
    flip_a_payload_byte(repo_dir)
    assert run(repo_dir, verify_engine.STRUCTURE).ok


def test_bit_rot_is_also_caught_by_the_default_tier(tmp_path):
    repo_dir, _ = build(tmp_path)
    flip_a_payload_byte(repo_dir)
    assert not run(repo_dir).ok


def test_a_corrupt_pack_trailer_is_caught_when_asked(tmp_path):
    repo_dir, _ = build(tmp_path)
    pack = sorted(pack_dir_of(repo_dir).glob("*.pack"))[0]
    data = bytearray(pack.read_bytes())
    data[-1] ^= 0xFF
    pack.write_bytes(bytes(data))

    assert run(repo_dir).ok                      # trailer is not on the default path
    report = run(repo_dir, "verify_packs")
    assert not report.ok
    assert kinds(report) == {"pack-trailer"}


# -- tier 3: malicious injection -------------------------------------------


def inject_and_repair(repo_dir) -> tuple:
    """Substitute one chunk's content for another's, then repair *every*
    self-certifying structure below the manifest: the pack's records, its
    trailer, and the whole index (offsets, lengths, checksums, trailer).

    This is the threat model PS 2b names. The pack is rebuilt from scratch, so
    nothing an attacker can reach is left inconsistent -- the substituted
    payload is stored under the victim's hash and every checksum agrees with
    it. Only the manifest's `chunks[].object`, reached by walking down from the
    ref, still disagrees.

    The pack keeps its original filename. An attacker has no reason to rename
    it, and nothing may treat a name matching a content hash as evidence --
    that would be one more self-certifying check.
    """
    # The richest pack, not an arbitrary one: after the root commit most packs
    # hold a single chunk, and sorting by hex filename picks between them at
    # random.
    packs = sorted(pack_dir_of(repo_dir).glob("*.pack"))
    pack = max(packs, key=lambda path: len(scan_pack(path)))
    entries = scan_pack(pack)
    assert len(entries) >= 2, "fixture must produce a pack with >= 2 chunks"
    victim, donor = entries[0], entries[1]

    original = pack.read_bytes()
    payloads = {
        e.content_hash: original[e.offset: e.offset + e.stored_len] for e in entries
    }
    plain_lens = {e.content_hash: e.plain_len for e in entries}
    payloads[victim.content_hash] = payloads[donor.content_hash]
    plain_lens[victim.content_hash] = donor.plain_len

    staging = pack.parent / "rebuild"
    staging.mkdir(exist_ok=True)
    with PackWriter(staging, tmp_dir=staging) as writer:
        for entry in entries:
            writer.add(
                entry.content_hash,
                payloads[entry.content_hash],
                plain_lens[entry.content_hash],
            )
    pack.write_bytes(writer.pack_path.read_bytes())
    writer.pack_path.unlink()
    staging.rmdir()

    write_index(
        pack.with_suffix(".idx"),
        pack_hash=bytes.fromhex(pack.stem),
        entries=scan_pack(pack),
        tmp_dir=pack.parent,
    )
    return victim, donor


def test_a_repaired_injection_survives_the_fast_tier(tmp_path):
    """The negative half of the pair, and the reason `content` is the default.

    Not a bug being documented -- a structural limit. The fast tier compares a
    payload against a value in the pack index, and the index is as writable as
    the pack. `--packs` is included to show that re-hashing the whole pack
    against its own trailer does not help either.
    """
    repo_dir, _ = build(tmp_path)
    inject_and_repair(repo_dir)
    assert run(repo_dir, verify_engine.CHECKSUM, "verify_packs").ok


def test_a_repaired_injection_is_caught_by_the_default_tier(tmp_path):
    repo_dir, _ = build(tmp_path)
    victim, donor = inject_and_repair(repo_dir)

    report = run(repo_dir)
    assert not report.ok
    [failure] = [f for f in report.failures if f.kind == "chunk-content-mismatch"]
    assert failure.object == victim.content_hash.hex()
    assert failure.actual == donor.content_hash.hex()
    # CLI.md ~7 requires the referrer, so a human can find the damaged tensor.
    assert "tensor-manifest" in failure.referenced_by
    assert failure.pack.endswith(".pack")


# -- content_hash tier ------------------------------------------------------


def test_content_hash_catches_a_manifest_that_lies_about_its_tensor(tmp_path):
    """The failure per-chunk hashing structurally cannot see.

    Every chunk still hashes correctly here -- only the tensor they reconstruct
    to disagrees with what the manifest records. That is the exact signature of
    a permutation composed in the wrong order, which is why this check exists
    before the aligner does.
    """
    repo_dir, repo = build(tmp_path)
    store = repo.store
    head_hash = repo.resolve_ref("HEAD")
    commit = graph.get_json(store, head_hash)
    manifest = graph.get_json(store, commit["checkpoint_manifest"])

    name, tensor_hash = next(iter(manifest["tensors"].items()))
    tensor = graph.get_json(store, tensor_hash)
    tensor["content_hash"] = "00" * 32
    manifest["tensors"][name] = graph.put_json(store, tensor)

    commit["checkpoint_manifest"] = graph.put_json(store, manifest)
    repo.update_ref("main", graph.put_json(store, commit))

    assert run(repo_dir).ok                     # self-consistent to every chunk check
    report = run(repo_dir, "check_content_hash")
    assert not report.ok
    assert kinds(report) == {"tensor-content-mismatch"}


# -- behaviour --------------------------------------------------------------


def test_verify_never_repairs_what_it_inspects(tmp_path):
    """PackSet rebuilds a missing index on open by default. Verify must not:
    repairing the artifact under inspection and then reporting OK would make
    the whole command worthless."""
    repo_dir, _ = build(tmp_path)
    index = sorted(pack_dir_of(repo_dir).glob("*.idx"))[0]
    index.unlink()

    report = run(repo_dir)
    assert not index.exists(), "verify rebuilt an index it was asked to check"
    assert not report.ok


def test_all_parents_are_walked_not_just_the_first(tmp_path):
    """PS 2f: verify any branch, not a linear chain. A commit reachable only
    through a second parent must still be verified."""
    repo_dir, repo = build(tmp_path, 3)
    store = repo.store
    first, second = [h for h, _ in graph.walk_first_parent(store, repo.resolve_ref("HEAD"))][:2]

    merge = graph.get_json(store, first)
    merge["parents"] = [first, second]
    merge["message"] = "merge"
    merge_hash = graph.put_json(store, merge)
    repo.update_ref("main", merge_hash)

    report = verify_engine.verify_lineage(repo, [merge_hash])
    assert report.ok, report.failures
    # merge + both parents' lineages, deduped by the visited set.
    assert report.commits == 4


def test_failures_are_capped(tmp_path):
    repo_dir, repo = build(tmp_path)
    # Corrupt only the tensor-manifests. Corrupting the commits instead would
    # break the walk at its first step and produce a single failure, which
    # would pass this assertion for the wrong reason.
    store = repo.store
    for commit_hash, commit in graph.walk_first_parent(store, repo.resolve_ref("HEAD")):
        manifest = graph.get_json(store, commit["checkpoint_manifest"])
        for tensor_hash in manifest["tensors"].values():
            (repo.objects_dir / tensor_hash[:2] / tensor_hash).write_bytes(b"corrupt")

    report = verify_engine.verify_lineage(
        repo, [repo.resolve_ref("HEAD")], max_failures=2
    )
    assert len(report.failures) == 2
    assert report.truncated


def test_an_unknown_tier_is_a_programming_error(tmp_path):
    repo_dir, repo = build(tmp_path)
    with pytest.raises(ValueError, match="unknown tier"):
        verify_engine.verify_lineage(repo, [repo.resolve_ref("HEAD")], tier="nope")


# -- CLI --------------------------------------------------------------------


def test_cli_exits_zero_on_a_clean_repo(tmp_path):
    repo_dir, _ = build(tmp_path)
    assert main(["-C", str(repo_dir), "verify"]) == 0


def test_cli_exits_four_on_tampering(tmp_path):
    """CLI.md ~1.3 reserves exit 4 for verification-class failures, and graders
    script against it."""
    repo_dir, _ = build(tmp_path)
    inject_and_repair(repo_dir)
    assert main(["-C", str(repo_dir), "verify"]) == 4


def test_cli_json_reports_ok_and_failures(tmp_path, capsys):
    repo_dir, _ = build(tmp_path)
    capsys.readouterr()
    assert main(["-C", str(repo_dir), "verify", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["failures"] == []
    assert payload["tier"] == verify_engine.CONTENT


def test_cli_verifies_every_branch_with_all(tmp_path, capsys):
    repo_dir, repo = build(tmp_path, 4)
    older = graph.walk_first_parent(repo.store, repo.resolve_ref("HEAD"))[2][0]
    assert main(["-C", str(repo_dir), "branch", "side", older]) == 0

    capsys.readouterr()
    assert main(["-C", str(repo_dir), "verify", "--all", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["commits"] == 4


def test_cli_on_an_unborn_head_is_a_usage_error(tmp_path):
    repo_dir = make_repo(tmp_path)
    assert main(["-C", str(repo_dir), "verify"]) == 2
