"""`synapsefs checkout` -- CLI.md ~4.

    synapsefs checkout <branch|commit> [--out <path>] [--no-materialize]

Pre-2.23 git semantics, one command with two jobs: move HEAD, and put the
checkpoint in the working tree. A branch name attaches HEAD to
`refs/heads/<name>`; anything else resolves to a commit and detaches it.

Ambiguity is resolved branch-first, matching `Repo.resolve_ref` -- a branch
literally named `9f2c1a` still checks out as a branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from synapsefs import graph
from synapsefs.cli import output
from synapsefs.errors import UsageError
from synapsefs.materialize import materialize
from synapsefs.store.repo import Repo

DEFAULT_CHECKPOINT_NAME = "model.safetensors"


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "checkout",
        parents=[global_parser],
        help="Switch branches or restore a commit's checkpoint",
    )
    parser.add_argument(
        "ref", metavar="<branch|commit>",
        help="Branch to switch to, or commit to detach HEAD at.",
    )
    parser.add_argument(
        "--out", default=None, metavar="<path>",
        help="Write the reconstructed checkpoint here instead of into the"
             " working tree.",
    )
    parser.add_argument(
        "--no-materialize", action="store_true",
        help="Move HEAD only; write no file.",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    """Reconstruct first, move HEAD second.

    That ordering is the same rule `commit` follows and for the same reason:
    the pointer moves last, so a failure anywhere in reconstruction leaves the
    repo exactly as it was rather than pointing HEAD at a commit whose
    checkpoint never made it to disk. The reverse order would turn a decode
    error into a repo whose HEAD and working tree disagree, which is precisely
    the state a version control system exists to prevent.
    """
    repo = Repo.find(args.repo)

    is_branch = repo.branch_exists(args.ref)
    commit_hash = repo.resolve_ref(args.ref)
    if commit_hash is None:
        raise UsageError(
            f"cannot check out {args.ref!r}: it does not name a commit yet"
        )

    store = repo.store
    commit = graph.get_json(store, commit_hash)

    written: Optional[dict] = None
    if not args.no_materialize:
        out_path = (
            Path(args.out) if args.out is not None
            else repo.root / _working_tree_name(commit)
        )
        source = graph.CommitCheckpoint(store, commit_hash)
        written = materialize(source, source.header_bytes, out_path)

    if is_branch:
        repo.set_head_branch(args.ref)
    else:
        repo.set_head_detached(commit_hash)

    return {
        "commit": commit_hash,
        "branch": args.ref if is_branch else None,
        "detached": not is_branch,
        "message": commit.get("message", ""),
        "materialized": written,
    }


def _working_tree_name(commit: dict) -> str:
    """Filename to restore into the working tree.

    `Path(...).name` is applied even though `commit` only ever records a
    basename: this value comes out of an object that could have been written
    by another implementation or fetched from a peer, and joining an
    attacker-chosen `../../.ssh/authorized_keys` onto the repo root would be a
    write-anywhere primitive. Cheap to strip, so strip it.

    Commits written before `checkpoint_name` existed have no such field; those
    fall back to the conventional name rather than failing a checkout.
    """
    recorded = commit.get("checkpoint_name") or DEFAULT_CHECKPOINT_NAME
    name = Path(recorded).name
    return name if name and name not in (".", "..") else DEFAULT_CHECKPOINT_NAME


def format_human(result: dict) -> str:
    """Matches CLI.md ~4's two worked examples."""
    lines = []
    if result["detached"]:
        lines.append(
            f"HEAD is now at {result['commit'][:6]} (detached) -- "
            f"{result['message']}"
        )
    else:
        lines.append(f"Switched to branch '{result['branch']}'")

    written = result.get("materialized")
    if written:
        lines.append(
            f"Wrote {written['path']} "
            f"({output.human_bytes(written['total_bytes'])}, "
            f"{written['tensors']} tensors)"
        )
    return "\n".join(lines)
