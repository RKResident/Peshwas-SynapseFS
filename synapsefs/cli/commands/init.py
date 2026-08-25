"""`synapsefs init` -- CLI.md ~2.

    synapsefs init [<path>] [--branch <name>]
"""

from __future__ import annotations

import argparse

from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    """Register this command's arguments on the shared subparsers object.

    `parents=[global_parser]` is what lets `--json`/`-q`/etc. be typed
    *after* `init` too, not just before it -- see parser.py's
    `_build_global_parser` docstring for why that works.
    """
    parser = subparsers.add_parser(
        "init",
        parents=[global_parser],
        help="Create a new SynapseFS repository",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Directory to initialize (default: current directory)",
    )
    parser.add_argument(
        "--branch", default="main",
        help="Name of the initial branch (default: main)",
    )
    # func/format_human ride on the namespace so main.py can dispatch and
    # render without knowing anything about "init" specifically.
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    """Do the actual work. Returns a plain dict -- never prints anything
    itself; cli/output.py decides how the result is rendered."""
    repo = Repo.init_at(args.path, branch=args.branch)
    return {
        "path": str(repo.synapse_dir),
        "branch": args.branch,
    }


def format_human(result: dict) -> str:
    """Human-readable rendering of run()'s result, matching CLI.md's
    worked example::

        Initialized empty SynapseFS repository in /home/u/myrepo/.synapse (branch: main)
    """
    return (
        f"Initialized empty SynapseFS repository in "
        f"{result['path']} (branch: {result['branch']})"
    )
