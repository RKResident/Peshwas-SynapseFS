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


def _make_checkpoint(tmp_path: Path, name: str = "model.safetensors") -> Path:
    """A real safetensors parse is out of scope for this module (align/codec
    don't exist yet) -- commit's own validation only checks the path exists
    and is a file, so a stand-in file is enough here."""
    path = tmp_path / name
    path.write_bytes(b"not-a-real-checkpoint")
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
    """CLI.md ~1.1: global flags are accepted before OR after the
    subcommand -- mirrors test_cli_init.py's version of this test.

    Both invocations reach run()'s NotImplementedError (exit 1, generic
    ERROR) rather than a usage error, proving -C/--repo and --json were
    parsed correctly regardless of position.
    """
    repo_dir = _init_repo(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}")
    checkpoint = _make_checkpoint(tmp_path)

    exit_after = main(
        ["-C", str(repo_dir), "commit", str(checkpoint), "-m", "msg", "--json"]
    )
    exit_before = main(
        ["--json", "-C", str(repo_dir), "commit", str(checkpoint), "-m", "msg"]
    )

    assert exit_after == 1
    assert exit_before == 1
    err = capsys.readouterr().err
    assert "not implemented" in err.lower()


def test_run_reaches_not_implemented_only_after_validation_passes(tmp_path: Path):
    """run() must validate the checkpoint path, the config path (on the
    root commit), and resolve --base *before* it ever raises
    NotImplementedError -- otherwise none of that plumbing is actually
    exercised."""
    from synapsefs.cli.commands import commit as commit_cmd
    import argparse

    repo_dir = _init_repo(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}")
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
    with pytest.raises(NotImplementedError):
        commit_cmd.run(args)


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
