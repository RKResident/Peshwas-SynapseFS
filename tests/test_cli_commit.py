"""Tests for `synapsefs commit` through the real CLI dispatch path
(parser -> args.func -> Repo -> output), mirroring test_cli_init.py.

This is deliberately the regression test for the `nargs="1"` class of bug
that used to take down the *entire* CLI (argparse raised ValueError while
building the parser, so even `synapsefs --help` died) -- `--help` for both
the top-level command and `commit` itself must keep succeeding no matter
what else changes in this module.

Run:
    pytest tests/test_cli_commit.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapsefs.cli.main import main


def _init_repo(tmp_path: Path) -> Path:
    target = tmp_path / "myrepo"
    main(["init", str(target)])
    return target


def _make_checkpoint(tmp_path: Path, name: str = "model.safetensors", *, seed: int = 0) -> Path:
    """A real, small `.safetensors` file.

    This used to be a stand-in blob, because commit's validation stopped at
    "the path exists" and the codec did not exist. It now runs the whole
    pipeline, so the file has to be genuine.
    """
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    path = tmp_path / name
    save_file(
        {
            "frozen": np.arange(64, dtype=np.float16).reshape(8, 8),
            "head": rng.standard_normal((8, 8)).astype(np.float16),
        },
        str(path),
    )
    return path


def test_top_level_help_exits_zero(capsys):
    """Regression test: a broken commit parser (e.g. nargs="1") used to
    raise ValueError while *building* the top-level parser, so even this
    unrelated --help call would die."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_commit_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["commit", "--help"])
    assert exc_info.value.code == 0


def test_missing_message_is_usage_error(tmp_path: Path, capsys):
    repo_dir = _init_repo(tmp_path)
    checkpoint = _make_checkpoint(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        main(["-C", str(repo_dir), "commit", str(checkpoint)])
    assert exc_info.value.code == 2


def test_nonexistent_checkpoint_is_usage_error(tmp_path: Path, capsys):
    repo_dir = _init_repo(tmp_path)
    missing = tmp_path / "does-not-exist.safetensors"
    exit_code = main(
        ["-C", str(repo_dir), "commit", str(missing), "-m", "msg"]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "checkpoint" in err


def test_global_flag_works_before_and_after_subcommand(tmp_path: Path, capsys):
    """CLI.md ~1.1: global flags are accepted before OR after the subcommand
    -- mirrors test_cli_init.py's version of this test. Both invocations must
    succeed, proving -C/--repo and --json were parsed regardless of position.
    """
    import json

    repo_dir = _init_repo(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}")
    capsys.readouterr()  # discard init's banner so the JSON below stands alone

    first = _make_checkpoint(tmp_path, "a.safetensors", seed=1)
    assert main(["-C", str(repo_dir), "commit", str(first), "-m", "one", "--json"]) == 0
    doc_after = json.loads(capsys.readouterr().out)

    second = _make_checkpoint(tmp_path, "b.safetensors", seed=2)
    assert main(["--json", "-C", str(repo_dir), "commit", str(second), "-m", "two"]) == 0
    doc_before = json.loads(capsys.readouterr().out)

    assert doc_after["branch"] == doc_before["branch"] == "main"
    # The second commit is a residual against the first, so it is much smaller.
    assert doc_after["full"] is True and doc_before["full"] is False
    assert doc_before["base"] == doc_after["commit"]


def test_run_returns_the_documented_result_keys(tmp_path: Path):
    """CLI.md ~3.1 fixes the `--json` key set; `format_human` and the
    benchmark harness both read it."""
    import argparse

    from synapsefs.cli.commands import commit as commit_cmd

    repo_dir = _init_repo(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    checkpoint = _make_checkpoint(tmp_path)

    args = argparse.Namespace(
        repo=str(repo_dir),
        checkpoint=str(checkpoint),
        message="epoch 1",
        config=None,
        base="HEAD",
        no_align=False,
        chunk_size=None,
        strict=False,
    )
    result = commit_cmd.run(args)

    for key in (
        "commit", "branch", "base", "message", "tensors", "original_bytes",
        "residual_bytes", "residual_ratio", "chunks_new", "chunks_deduped",
        "alignment",
    ):
        assert key in result, key
    assert result["branch"] == "main"
    assert result["base"] is None            # root commit has no base
    assert result["tensors"] == 2
    assert 0 < result["residual_ratio"] <= 1.5


def test_run_validates_checkpoint_before_not_implemented(tmp_path: Path):
    from synapsefs.cli.commands import commit as commit_cmd
    from synapsefs.errors import UsageError
    import argparse

    repo_dir = _init_repo(tmp_path)
    args = argparse.Namespace(
        repo=str(repo_dir),
        checkpoint=str(tmp_path / "nope.safetensors"),
        message="epoch 1",
        config=None,
        base="HEAD",
        no_align=False,
        chunk_size=None,
        strict=False,
    )
    with pytest.raises(UsageError):
        commit_cmd.run(args)


def test_run_requires_config_on_root_commit(tmp_path: Path):
    from synapsefs.cli.commands import commit as commit_cmd
    from synapsefs.errors import UsageError
    import argparse

    repo_dir = _init_repo(tmp_path)
    checkpoint = _make_checkpoint(tmp_path)
    args = argparse.Namespace(
        repo=str(repo_dir),
        checkpoint=str(checkpoint),
        message="epoch 1",
        config=None,  # no config.json beside the checkpoint either
        base="HEAD",
        no_align=False,
        chunk_size=None,
        strict=False,
    )
    with pytest.raises(UsageError):
        commit_cmd.run(args)
