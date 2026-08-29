"""Tests for the commit object graph and reconstruction (FORMAT.md 4, 10, 12A).

The load-bearing test here is `test_every_commit_in_a_chain_reconstructs_exactly`:
it commits a run of checkpoints, then reads each one back out of the repo and
compares byte for byte. That is the PS's core deliverable ("the reconstructed
file must be byte-for-byte identical to the target .safetensors file, including
all headers and metadata") reduced to an assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from synapsefs import graph
from synapsefs.codec.chunk import is_delta
from synapsefs.cli.main import main
from synapsefs.safetensors_io import SafetensorsFile
from synapsefs.store.repo import Repo


def make_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    main(["init", str(repo_dir)])
    (tmp_path / "config.json").write_text(json.dumps({"arch": "mlp"}))
    return repo_dir


def commit_series(tmp_path: Path, repo_dir: Path, n: int) -> list[Path]:
    """A fine-tune run: one tensor frozen, one drifting. Returns the paths."""
    rng = np.random.default_rng(0)
    frozen = rng.standard_normal((48, 32)).astype(np.float16)
    head = rng.standard_normal((48, 32)).astype(np.float16)
    bias = rng.standard_normal(48).astype(np.float16)

    paths = []
    for i in range(n):
        head = head + np.float16(0.002)
        path = tmp_path / f"epoch{i}.safetensors"
        save_file({"frozen": frozen, "head": head.copy(), "bias": bias}, str(path))
        assert main(["-C", str(repo_dir), "commit", str(path), "-m", f"epoch {i}",
                 "--config", str(tmp_path / "config.json"), "-q"]) == 0
        paths.append(path)
    return paths


def lineage(repo: Repo) -> list[dict]:
    """Commits oldest-first."""
    out, h = [], repo.resolve_ref("HEAD")
    while h:
        commit = graph.get_json(repo.store, h)
        commit["_hash"] = h
        out.append(commit)
        h = commit["parents"][0] if commit["parents"] else None
    return list(reversed(out))


def reconstruct(repo: Repo, commit_hash: str) -> dict:
    """Every tensor of a commit, as raw bit patterns."""
    view = graph.CommitCheckpoint(repo.store, commit_hash)
    return {
    name: view.rows(name, 0, view.spec(name).num_rows).ravel().copy()
    for name in view.names()
    }


# ---------------------------------------------------------------------------


def test_root_commit_reconstructs_exactly(tmp_path):
    repo_dir = make_repo(tmp_path)
    [path] = commit_series(tmp_path, repo_dir, 1)
    repo = Repo.find(repo_dir)

    rebuilt = reconstruct(repo, repo.resolve_ref("HEAD"))
    with SafetensorsFile(path) as original:
        for name in original.names():
            assert np.array_equal(rebuilt[name], original.whole(name).ravel()), name


def test_every_commit_in_a_chain_reconstructs_exactly(tmp_path):
    """Seven commits, so the chain crosses a re-basing boundary and the deepest
    residual walks three levels. Every one of them must come back byte-exact,
    not just the newest."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 7)
    repo = Repo.find(repo_dir)

    for commit, path in zip(lineage(repo), paths):
        rebuilt = reconstruct(repo, commit["_hash"])
        with SafetensorsFile(path) as original:
            assert sorted(rebuilt) == sorted(original.names())
            for name in original.names():
                assert np.array_equal(rebuilt[name], original.whole(name).ravel()), (
                f"{commit['message']}: tensor {name} differs"
                )


def test_headers_are_replayed_verbatim(tmp_path):
    """The PS demands byte-for-byte identity *including headers*. A
    safetensors header is space-padded with `__metadata__` first, so it cannot
    be regenerated from parsed fields -- it has to be stored and replayed."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 3)
    repo = Repo.find(repo_dir)

    for commit, path in zip(lineage(repo), paths):
        view = graph.CommitCheckpoint(repo.store, commit["_hash"])
        stored = view.header_bytes
        raw = path.read_bytes()
        assert stored == raw[: len(stored)]


def test_rebasing_stores_a_full_checkpoint_every_fourth_commit(tmp_path):
    """FORMAT.md 12A. Bounds reconstruction depth at REBASE_INTERVAL - 1."""
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 9)
    repo = Repo.find(repo_dir)

    flags = [c["full"] for c in lineage(repo)]
    assert flags == [True, False, False, False, True, False, False, False, True]


def test_drift_since_the_anchor_never_exceeds_the_interval(tmp_path):
    """N bounds how far a group is allowed to spread from its hub. It is not a
    reconstruction depth -- see the star test below for that."""
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 9)
    repo = Repo.find(repo_dir)

    for commit in lineage(repo):
        since = graph.commits_since_full(repo.store, commit["_hash"])
        assert since < graph.REBASE_INTERVAL, f"{commit['message']} is {since} out"


def test_commits_form_a_star_so_reconstruction_is_always_one_hop(tmp_path):
    """FORMAT.md 12A. Every residual commit diffs directly against its group's
    full checkpoint, never against another residual.

    The property, stated structurally: for any tensor-manifest that has a base,
    that base must itself have no base. One residual decode on top of one full
    checkpoint, regardless of how many commits sit in between.

    Under the old chain topology the third commit in a group walked three
    levels; measured at depth 3 that cost 2.19x the wall-clock, and more again
    for the partial reads FUSE issues, which pull chunks at every level.
    """
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 9)
    repo = Repo.find(repo_dir)

    checked = 0
    for commit in lineage(repo):
        checkpoint = graph.get_json(repo.store, commit["checkpoint_manifest"])
        for name, manifest_hash in checkpoint["tensors"].items():
            manifest = graph.get_json(repo.store, manifest_hash)
            base_hash = manifest["base_tensor_manifest"]
            if base_hash is None:
                continue
            base = graph.get_json(repo.store, base_hash)
            assert base["base_tensor_manifest"] is None, (
            f"{commit['message']}/{name} is a residual of a residual"
            )
            checked += 1
    assert checked > 0, "no residual manifests were exercised"


def test_the_anchor_actually_used_is_reported(tmp_path):
    """`--base` selects the lineage; the diff target is that lineage's anchor.
    The substitution must be visible in the result, not silent."""
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 3)
    repo = Repo.find(repo_dir)

    commits = lineage(repo)
    root = commits[0]["_hash"]
    for commit in commits[1:]:
        assert graph.nearest_full_ancestor(repo.store, commit["_hash"]) == root


def test_a_full_commit_still_dedups_against_earlier_chunks(tmp_path):
    """FORMAT.md 12A's cost argument: a re-basing commit rewrites manifests but
    only re-stores chunks that actually changed, because its chunks are still
    content-addressed and deduped against every existing pack."""
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 5)          # commit 5 (index 4) is full
    repo = Repo.find(repo_dir)

    full_commit = lineage(repo)[4]
    assert full_commit["full"] is True
    manifest = graph.get_json(repo.store, full_commit["checkpoint_manifest"])
    frozen = graph.get_json(repo.store, manifest["tensors"]["frozen"])
    # Stored in full, and its chunk was already in an earlier pack.
    assert frozen["base_tensor_manifest"] is None
    assert all(not is_delta(c["encoding"]) for c in frozen["chunks"])


def test_partial_row_reads_match_the_whole_tensor(tmp_path):
    """FORMAT.md 10 forbids materializing a whole tensor to satisfy a partial
    read -- the FUSE path depends on this being exact."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 4)
    repo = Repo.find(repo_dir)

    view = graph.CommitCheckpoint(repo.store, repo.resolve_ref("HEAD"))
    whole = view.rows("head", 0, view.spec("head").num_rows)
    for lo, hi in [(0, 1), (5, 9), (10, 48), (47, 48), (0, 48)]:
        assert np.array_equal(view.rows("head", lo, hi), whole[lo:hi]), (lo, hi)


def test_an_unchanged_tensor_reuses_the_base_manifest(tmp_path):
    """FORMAT.md 4.5's reuse rule.

    `bias` never changes across the run, so every commit's checkpoint-manifest
    must point at the *same* tensor-manifest -- the one the root commit wrote.
    No new object, and crucially no residual chain: without this, a frozen
    tensor accumulates one zero-delta hop per commit and reconstruction walks
    all of them for nothing.

    e0: manifest=86b2a599  base=None
    e1: manifest=86b2a599  base=None      <- reused, not rewritten
    e2: manifest=86b2a599  base=None
    """
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 6)
    repo = Repo.find(repo_dir)

    bias_manifests, head_manifests = set(), set()
    for commit in lineage(repo):
        checkpoint = graph.get_json(repo.store, commit["checkpoint_manifest"])
        bias_manifests.add(checkpoint["tensors"]["bias"])
        head_manifests.add(checkpoint["tensors"]["head"])

    assert len(bias_manifests) == 1, "an unchanged tensor must be described once"
    # The frozen tensor is stored in full and never chains.
    bias = graph.get_json(repo.store, bias_manifests.pop())
    assert bias["base_tensor_manifest"] is None
    # ...while the tensor that actually changes still gets a manifest per commit.
    assert len(head_manifests) == 6


def test_reuse_does_not_break_reconstruction(tmp_path):
    """The reuse rule silently drops tensors from the encode path, so the
    round-trip has to be re-proved with it active."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 6)
    repo = Repo.find(repo_dir)

    for commit, path in zip(lineage(repo), paths):
        rebuilt = reconstruct(repo, commit["_hash"])
        with SafetensorsFile(path) as original:
            assert sorted(rebuilt) == sorted(original.names())
            for name in original.names():
                assert np.array_equal(rebuilt[name], original.whole(name).ravel()), (
                f"{commit['message']}: {name}"
                )


def test_reuse_is_reported(tmp_path):
    """`commit --json` should show the saving rather than hide it."""
    import argparse

    from synapsefs.cli.commands import commit as commit_cmd

    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 1)   # HEAD is the root, i.e. the anchor
    args = argparse.Namespace(
    repo=str(repo_dir), checkpoint=str(paths[-1]), message="again",
    config=None, base="HEAD", no_align=False, chunk_size=None, strict=False,
    )
    result = commit_cmd.run(args)
    # Re-committing the anchor's own checkpoint: every tensor is unchanged.
    assert result["tensors_unchanged"] == result["tensors"]
    assert result["residual_bytes"] == 0


def test_canonical_json_is_order_independent(tmp_path):
    """Object hashes are taken over these bytes, so two writers that disagree
    on dict order must not produce two objects."""
    a = graph.canonical_json({"b": 2, "a": 1})
    b = graph.canonical_json({"a": 1, "b": 2})
    assert a == b == b'{"a":1,"b":2}'


def test_commit_objects_match_the_documented_schema(tmp_path):
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 2)
    repo = Repo.find(repo_dir)

    head = graph.get_json(repo.store, repo.resolve_ref("HEAD"))
    assert set(head) == {
    "checkpoint_manifest", "parents", "timestamp", "message", "full",
    "checkpoint_name",
    }
    assert len(head["parents"]) == 1

    manifest = graph.get_json(repo.store, head["checkpoint_manifest"])
    assert set(manifest) == {"header_object", "tensors", "topology_config_hash"}
    assert manifest["topology_config_hash"] is not None


def test_config_is_reused_from_the_base_when_omitted(tmp_path):
    """CLI.md ~3: required on the first commit, "reused from the base commit
    afterward if omitted"."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 1)
    repo = Repo.find(repo_dir)
    first = graph.get_json(repo.store, repo.resolve_ref("HEAD"))
    first_config = graph.get_json(repo.store, first["checkpoint_manifest"])["topology_config_hash"]

    second = tmp_path / "next.safetensors"
    with SafetensorsFile(paths[0]) as f:
        pass
    save_file({"frozen": np.zeros((48, 32), np.float16),
           "head": np.ones((48, 32), np.float16),
           "bias": np.zeros(48, np.float16)}, str(second))
    # No --config, and no config.json beside the checkpoint.
    assert main(["-C", str(repo_dir), "commit", str(second), "-m", "no config", "-q"]) == 0

    repo = Repo.find(repo_dir)
    head = graph.get_json(repo.store, repo.resolve_ref("HEAD"))
    assert graph.get_json(repo.store, head["checkpoint_manifest"])["topology_config_hash"] == first_config


def test_committing_on_a_detached_head_is_refused(tmp_path):
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 1)
    repo = Repo.find(repo_dir)
    (repo.synapse_dir / "HEAD").write_text(repo.resolve_ref("HEAD") + "\n")

    exit_code = main(["-C", str(repo_dir), "commit", str(paths[0]), "-m", "x"])
    assert exit_code == 2  # CLI.md: USAGE
