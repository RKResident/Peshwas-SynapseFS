"""Tests for `Repo`'s ref-resolution surface: `read_head`, `resolve_ref`,
`update_ref`, and `set_head_branch` (synapsefs/store/repo.py).

`commit`, `checkout`, `branch`, and `log` all need identical ref-resolution
logic -- these tests exercise it once, here, rather than once per command.

Run:
    pytest tests/test_repo_refs.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapsefs.errors import UsageError
from synapsefs.store.repo import Repo


def _fake_hash(byte: str, n: int = 64) -> str:
    """A syntactically valid hex "commit hash" for tests that don't need a
    real object to exist -- just repeat a hex digit out to full length."""
    return (byte * n)[:n]


def test_unborn_head_after_init(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    branch, commit_hash = repo.read_head()
    assert branch == "main"
    assert commit_hash is None


def test_resolve_head_on_unborn_repo_returns_none(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    assert repo.resolve_ref("HEAD") is None


def test_attached_head_after_update_ref(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    commit_hash = _fake_hash("a")
    repo.update_ref("main", commit_hash)

    branch, resolved = repo.read_head()
    assert branch == "main"
    assert resolved == commit_hash
    assert repo.resolve_ref("HEAD") == commit_hash
    assert repo.resolve_ref("main") == commit_hash


def test_detached_head(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    commit_hash = _fake_hash("b")
    # Detached HEAD is a raw hash with no "ref: " prefix -- not produced by
    # any Repo method yet (checkout <commit> lands later), so write it
    # directly the way `checkout` will.
    repo.head_path.write_text(f"{commit_hash}\n", encoding="utf-8")

    branch, resolved = repo.read_head()
    assert branch is None
    assert resolved == commit_hash
    assert repo.resolve_ref("HEAD") == commit_hash


def test_resolve_unknown_branch_raises_usage_error(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    with pytest.raises(UsageError):
        repo.resolve_ref("no-such-branch")


def test_resolve_path_traversal_branch_name_raises_usage_error(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    with pytest.raises(UsageError):
        repo.update_ref("../../etc/passwd", _fake_hash("c"))


@pytest.mark.parametrize(
    "bad_name",
    ["", "-oops", "a/b", "..", "foo/../bar", "a\\b"],
)
def test_invalid_branch_names_rejected(tmp_path: Path, bad_name: str):
    repo = Repo.init_at(tmp_path, branch="main")
    with pytest.raises(UsageError):
        repo.update_ref(bad_name, _fake_hash("d"))
    with pytest.raises(UsageError):
        repo.set_head_branch(bad_name)


def test_update_ref_then_resolve_by_full_hash(tmp_path: Path):
    """A full 64-char hash resolves only once *some* object on disk
    actually has that hash -- resolve_ref treats an unknown full hash the
    same as an unknown branch: a UsageError, not a silent pass-through."""
    repo = Repo.init_at(tmp_path, branch="main")
    commit_hash = repo.store.put(b"pretend-commit-object")
    repo.update_ref("main", commit_hash)

    assert repo.resolve_ref(commit_hash) == commit_hash
    with pytest.raises(UsageError):
        repo.resolve_ref("f" * 64)


def test_resolve_abbreviated_hash(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    commit_hash = repo.store.put(b"pretend-commit-object-2")
    abbrev = commit_hash[:8]
    assert repo.resolve_ref(abbrev) == commit_hash


def test_resolve_ambiguous_abbreviated_hash_raises_usage_error(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    # Fabricate two objects that share a 6-char prefix by writing directly
    # under objects/, rather than relying on finding a real BLAKE3
    # collision (infeasible) -- resolve_ref only reads the filesystem
    # layout, so this is a faithful way to exercise the ambiguity branch.
    shard = repo.objects_dir / "ab"
    shard.mkdir(parents=True)
    (shard / ("ab1234" + "0" * 58)).write_bytes(b"x")
    (shard / ("ab1234" + "1" * 58)).write_bytes(b"y")

    with pytest.raises(UsageError):
        repo.resolve_ref("ab1234")


def test_set_head_branch_switches_attached_branch(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    repo.set_head_branch("experiment")
    branch, commit_hash = repo.read_head()
    assert branch == "experiment"
    assert commit_hash is None  # still unborn on the new branch


def test_resolve_empty_ref_raises_usage_error(tmp_path: Path):
    repo = Repo.init_at(tmp_path, branch="main")
    with pytest.raises(UsageError):
        repo.resolve_ref("")
