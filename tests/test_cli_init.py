"""End-to-end tests of `synapsefs init` through the real CLI dispatch path
(parser -> args.func -> Repo -> output) rather than calling Repo.init_at()
directly -- this is what actually proves main.py's wiring works, including
the exit-code contract and the global-flag-position rule.

Run:
    pytest tests/test_cli_init.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from synapsefs.cli.main import main


def test_init_creates_expected_layout(tmp_path: Path, capsys):
    target = tmp_path / "myrepo"
    exit_code = main(["init", str(target)])

    assert exit_code == 0
    synapse_dir = target / ".synapse"
    assert (synapse_dir / "objects" / "tmp").is_dir()
    assert (synapse_dir / "refs" / "heads").is_dir()
    assert (synapse_dir / "HEAD").read_text() == "ref: refs/heads/main\n"

    out = capsys.readouterr().out
    assert "Initialized empty SynapseFS repository" in out


def test_init_respects_branch_flag(tmp_path: Path):
    target = tmp_path / "myrepo"
    main(["init", str(target), "--branch", "dev"])
    head = (target / ".synapse" / "HEAD").read_text()
    assert head == "ref: refs/heads/dev\n"


def test_init_json_output_is_single_document(tmp_path: Path, capsys):
    """CLI.md ~1.2: under --json, stdout must be exactly one JSON document,
    nothing else -- json.loads() failing here means something else got
    written to stdout alongside it."""
    target = tmp_path / "myrepo"
    main(["init", str(target), "--json"])
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["branch"] == "main"
    assert doc["path"].endswith(".synapse")


def test_init_twice_fails_with_usage_exit_code(tmp_path: Path, capsys):
    target = tmp_path / "myrepo"
    main(["init", str(target)])
    exit_code = main(["init", str(target)])
    assert exit_code == 2  # CLI.md: USAGE
    err = capsys.readouterr().err
    assert "already exists" in err


def test_global_flag_works_before_and_after_subcommand(tmp_path: Path, capsys):
    """CLI.md ~1.1: global flags are accepted before OR after the subcommand."""
    target1 = tmp_path / "repo-flag-after"
    main(["init", str(target1), "--json"])
    out_after = capsys.readouterr().out

    target2 = tmp_path / "repo-flag-before"
    main(["--json", "init", str(target2)])
    out_before = capsys.readouterr().out

    assert json.loads(out_after)["branch"] == "main"
    assert json.loads(out_before)["branch"] == "main"


def test_bad_branch_name_leaves_nothing_on_disk(tmp_path: Path, capsys):
    """A rejected --branch must be a total no-op on disk, not a partial init.

    Regression test for a wedge: `Repo.init_at` used to create the whole
    `.synapse/` tree and only then write HEAD (which is where the branch name
    gets validated). A bad name therefore aborted *after* the mkdirs, leaving
    a `.synapse/` with every directory and no HEAD -- and because init refuses
    to touch an existing `.synapse/`, every retry then failed too. One typo
    wedged the directory until someone deleted it by hand.
    """
    target = tmp_path / "myrepo"
    exit_code = main(["init", str(target), "--branch", "../evil"])

    assert exit_code == 2  # CLI.md: USAGE
    assert "invalid branch name" in capsys.readouterr().err
    assert not (target / ".synapse").exists()

    # The real point of the test: a retry with a valid name must now work.
    assert main(["init", str(target), "--branch", "main"]) == 0
    assert (target / ".synapse" / "HEAD").read_text() == "ref: refs/heads/main\n"


def test_no_subcommand_is_usage_error(capsys):
    exit_code = main([])
    assert exit_code == 2
