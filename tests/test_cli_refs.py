"""Tests for `branch`, `checkout`, `log` and `restore` (CLI.md ~4, ~5, ~6).

These are the commands that make a repo navigable, so most of what is worth
asserting is about *state after the command*, not the text it printed: where
HEAD points, which refs exist, and whether the working tree file matches the
commit it claims to be.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapsefs import graph
from synapsefs.cli.main import main
from synapsefs.store.repo import Repo

from tests.test_graph import commit_series, make_repo


def run_json(capsys, argv) -> dict:
    capsys.readouterr()          # drop anything an earlier command left behind
    assert main(argv + ["--json"]) == 0
    return json.loads(capsys.readouterr().out)


def repo_with(tmp_path, n: int):
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, n)
    return repo_dir, paths, Repo.find(repo_dir)


# -- log -------------------------------------------------------------------


def test_log_walks_newest_first_and_marks_the_hub_commits(tmp_path, capsys):
    repo_dir, _, _ = repo_with(tmp_path, 6)
    result = run_json(capsys, ["-C", str(repo_dir), "log"])

    messages = [c["message"] for c in result["commits"]]
    assert messages == [f"epoch {i}" for i in reversed(range(6))]

    # FORMAT.md 12A: every REBASE_INTERVAL-th commit is a new star hub.
    full_at = [i for i, c in enumerate(reversed(result["commits"])) if c["full"]]
    assert full_at == [0, graph.REBASE_INTERVAL]


def test_log_n_limits_the_walk(tmp_path, capsys):
    repo_dir, _, _ = repo_with(tmp_path, 5)
    result = run_json(capsys, ["-C", str(repo_dir), "log", "-n", "2"])
    assert len(result["commits"]) == 2


def test_log_rejects_a_non_positive_count(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "log", "-n", "0"]) == 2


def test_log_on_an_unborn_head_is_empty_not_an_error(tmp_path, capsys):
    repo_dir = make_repo(tmp_path)
    result = run_json(capsys, ["-C", str(repo_dir), "log"])
    assert result["commits"] == []
    assert result["branch"] == "main"


def test_log_sizes_are_derived_not_stored(tmp_path, capsys):
    """The figures must come out of the object graph, so they must not appear
    in the commit object itself (see graph.checkpoint_sizes)."""
    repo_dir, _, repo = repo_with(tmp_path, 2)
    result = run_json(capsys, ["-C", str(repo_dir), "log"])
    newest = result["commits"][0]
    assert newest["stored_bytes"] > 0
    assert newest["original_bytes"] > newest["stored_bytes"]

    commit = graph.get_json(repo.store, newest["commit"])
    assert "stored_bytes" not in commit and "original_bytes" not in commit


def test_log_no_size_omits_the_figures(tmp_path, capsys):
    repo_dir, _, _ = repo_with(tmp_path, 2)
    result = run_json(capsys, ["-C", str(repo_dir), "log", "--no-size"])
    assert "stored_bytes" not in result["commits"][0]


def test_log_decorates_head_and_branches(tmp_path, capsys):
    repo_dir, _, _ = repo_with(tmp_path, 2)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 0
    result = run_json(capsys, ["-C", str(repo_dir), "log"])
    assert set(result["commits"][0]["refs"]) == {"HEAD -> main", "side"}


# -- branch ----------------------------------------------------------------


def test_branch_create_does_not_switch(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 3)
    older = graph.walk_first_parent(repo.store, repo.resolve_ref("HEAD"))[2][0]

    result = run_json(capsys, ["-C", str(repo_dir), "branch", "side", older])
    assert result["commit"] == older
    assert repo.read_head() == ("main", repo.resolve_ref("main"))
    assert repo.resolve_ref("side") == older


def test_branch_create_defaults_to_head(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 2)
    run_json(capsys, ["-C", str(repo_dir), "branch", "side"])
    assert repo.resolve_ref("side") == repo.resolve_ref("HEAD")


def test_branch_refuses_to_overwrite_an_existing_branch(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 0
    assert main(["-C", str(repo_dir), "branch", "side"]) == 2


def test_branch_refuses_a_path_traversing_name(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "../escape"]) == 2


def test_branch_cannot_be_created_on_an_unborn_head(tmp_path):
    repo_dir = make_repo(tmp_path)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 2


def test_branch_delete_refuses_the_current_branch(tmp_path):
    """CLI.md ~5: deleting the current branch fails with 2."""
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "-d", "main"]) == 2


def test_branch_delete_removes_the_ref(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 0
    run_json(capsys, ["-C", str(repo_dir), "branch", "-d", "side"])
    assert "side" not in repo.list_branches()


def test_branch_rename_follows_head_when_it_was_current(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 1)
    run_json(capsys, ["-C", str(repo_dir), "branch", "-m", "main", "trunk"])
    branch, commit = repo.read_head()
    assert branch == "trunk"
    assert commit is not None
    assert "main" not in repo.list_branches()


def test_branch_rename_onto_an_existing_name_is_refused(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 0
    assert main(["-C", str(repo_dir), "branch", "-m", "side", "main"]) == 2


def test_branch_listing_marks_the_current_branch(tmp_path, capsys):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "branch", "side"]) == 0
    result = run_json(capsys, ["-C", str(repo_dir), "branch"])
    current = {b["branch"]: b["current"] for b in result["branches"]}
    assert current == {"main": True, "side": False}


# -- checkout --------------------------------------------------------------


def test_checkout_branch_attaches_head_and_materializes(tmp_path, capsys):
    repo_dir, paths, repo = repo_with(tmp_path, 3)
    older = graph.walk_first_parent(repo.store, repo.resolve_ref("HEAD"))[2][0]
    assert main(["-C", str(repo_dir), "branch", "side", older]) == 0

    result = run_json(capsys, ["-C", str(repo_dir), "checkout", "side"])
    assert result["detached"] is False
    assert repo.read_head() == ("side", older)

    # Restored under the name recorded in the commit, into the repo root.
    restored = Path(repo_dir) / "epoch0.safetensors"
    assert restored.read_bytes() == paths[0].read_bytes()


def test_checkout_commit_detaches_head(tmp_path, capsys):
    repo_dir, paths, repo = repo_with(tmp_path, 3)
    older = graph.walk_first_parent(repo.store, repo.resolve_ref("HEAD"))[1][0]

    out = tmp_path / "detached.safetensors"
    result = run_json(
        capsys, ["-C", str(repo_dir), "checkout", older[:10], "--out", str(out)]
    )
    assert result["detached"] is True
    assert repo.read_head() == (None, older)
    assert out.read_bytes() == paths[1].read_bytes()


def test_checkout_no_materialize_writes_nothing(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 2)
    before = {p.name for p in Path(repo_dir).iterdir()}

    result = run_json(
        capsys, ["-C", str(repo_dir), "checkout", "main", "--no-materialize"]
    )
    assert result["materialized"] is None
    assert {p.name for p in Path(repo_dir).iterdir()} == before


def test_checkout_prefers_a_branch_over_a_hash_shaped_name(tmp_path, capsys):
    """`Repo.resolve_ref` resolves branch-first; `checkout` must agree, or a
    branch named like a hex prefix would silently detach HEAD."""
    repo_dir, _, repo = repo_with(tmp_path, 2)
    assert main(["-C", str(repo_dir), "branch", "abcdef"]) == 0
    result = run_json(capsys, ["-C", str(repo_dir), "checkout", "abcdef"])
    assert result["detached"] is False
    assert repo.read_head()[0] == "abcdef"


def test_checkout_of_an_unknown_ref_is_a_usage_error(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "checkout", "nope"]) == 2


def test_checkout_does_not_move_head_when_reconstruction_fails(tmp_path, monkeypatch):
    """The ordering rule: reconstruct first, move HEAD last. A decode failure
    must leave the repo exactly as it was."""
    repo_dir, _, repo = repo_with(tmp_path, 2)
    before = repo.read_head()

    import synapsefs.cli.commands.checkout as checkout_cmd

    def boom(*a, **k):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(checkout_cmd, "materialize", boom)
    assert main(["-C", str(repo_dir), "checkout", "main"]) == 1
    assert repo.read_head() == before


def test_checkout_of_a_commit_without_a_recorded_name_falls_back(tmp_path, capsys):
    """Commits written before `checkpoint_name` existed must still check out."""
    repo_dir, _, repo = repo_with(tmp_path, 1)
    head = repo.resolve_ref("HEAD")
    commit = graph.get_json(repo.store, head)
    commit.pop("checkpoint_name")
    legacy = graph.put_json(repo.store, commit)
    repo.update_ref("main", legacy)

    run_json(capsys, ["-C", str(repo_dir), "checkout", "main"])
    assert (Path(repo_dir) / "model.safetensors").is_file()


# -- restore ---------------------------------------------------------------


def test_restore_writes_a_byte_identical_file_and_reports_no_drift(tmp_path, capsys):
    repo_dir, paths, _ = repo_with(tmp_path, 5)
    out = tmp_path / "restored.safetensors"
    result = run_json(
        capsys,
        ["-C", str(repo_dir), "restore", "HEAD", "--out", str(out),
         "--compare", str(paths[-1]), "--strict"],
    )
    assert result["identical"] is True
    assert result["identical_bytes"] is True
    assert result["tensors_differing"] == 0
    assert out.read_bytes() == paths[-1].read_bytes()


def test_restore_never_moves_head(tmp_path, capsys):
    repo_dir, _, repo = repo_with(tmp_path, 3)
    older = graph.walk_first_parent(repo.store, repo.resolve_ref("HEAD"))[2][0]
    before = repo.read_head()

    run_json(
        capsys,
        ["-C", str(repo_dir), "restore", older,
         "--out", str(tmp_path / "old.safetensors")],
    )
    assert repo.read_head() == before


def test_restore_reports_real_drift_between_two_epochs(tmp_path, capsys):
    repo_dir, paths, _ = repo_with(tmp_path, 3)
    result = run_json(
        capsys,
        ["-C", str(repo_dir), "restore", "HEAD", "--compare", str(paths[0])],
    )
    assert result["identical"] is False
    drifting = {c["name"] for c in result["comparisons"]}
    # `head` drifts every epoch in the fixture; `frozen` and `bias` never do.
    assert drifting == {"head"}


def test_restore_strict_exits_with_the_integrity_code(tmp_path):
    repo_dir, paths, _ = repo_with(tmp_path, 3)
    assert main([
        "-C", str(repo_dir), "restore", "HEAD",
        "--compare", str(paths[0]), "--strict",
    ]) == 4


def test_restore_needs_somewhere_to_go(tmp_path):
    repo_dir, _, _ = repo_with(tmp_path, 1)
    assert main(["-C", str(repo_dir), "restore", "HEAD"]) == 2
