"""`synapsefs branch` -- CLI.md ~5.

    synapsefs branch                          # list
    synapsefs branch <name> [<start-point>]   # create at <start-point>, default HEAD
    synapsefs branch -d <name>                # delete
    synapsefs branch -m <old> <new>           # rename

Creating does **not** switch -- pre-2.23 git semantics, same as the rest of
this CLI. Use `checkout <name>` afterwards.
"""

from __future__ import annotations

import argparse
from typing import List

from synapsefs import graph
from synapsefs.errors import UsageError
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "branch",
        parents=[global_parser],
        help="List, create, delete, or rename branches",
    )
    # -d and -m are mutually exclusive *modes*, and the positional operands
    # mean different things under each. argparse cannot express "two names
    # under -m, one under -d, zero-to-two otherwise", so the operands are
    # collected loosely here and validated in run() where the mode is known --
    # which also lets the error messages name the actual mode.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-d", "--delete", action="store_true",
        help="Delete <name>. Fails with exit 2 on the current branch.",
    )
    mode.add_argument(
        "-m", "--move", action="store_true",
        help="Rename <old> to <new>.",
    )
    parser.add_argument(
        "names", nargs="*", metavar="<name>",
        help="Branch name(s); meaning depends on the mode. With no operands"
             " and no mode flag, lists every branch.",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    repo = Repo.find(args.repo)
    head_branch, head_commit = repo.read_head()
    names: List[str] = list(args.names)

    if args.delete:
        if len(names) != 1:
            raise UsageError("branch -d takes exactly one branch name")
        target = names[0]
        # CLI.md ~5: "Deleting the current branch fails with 2." Checked here
        # rather than in Repo.delete_branch because it is a policy about what
        # HEAD means, not about whether the ref is writable -- and `merge` or a
        # future `push --delete` may want the same ref removal without it.
        if target == head_branch:
            raise UsageError(
                f"cannot delete branch '{target}': it is the current branch"
            )
        deleted = repo.delete_branch(target)
        return {"action": "delete", "branch": target, "commit": deleted}

    if args.move:
        if len(names) != 2:
            raise UsageError("branch -m takes exactly two branch names")
        old, new = names
        moved = repo.rename_branch(old, new)
        return {"action": "rename", "from": old, "branch": new, "commit": moved}

    if not names:
        return _list(repo, head_branch, head_commit)

    if len(names) > 2:
        raise UsageError("branch takes at most a name and a start point")

    name = names[0]
    start_point = names[1] if len(names) == 2 else "HEAD"
    if repo.branch_exists(name):
        raise UsageError(f"branch already exists: {name!r}")

    start = repo.resolve_ref(start_point)
    if start is None:
        # Unborn HEAD: there is no commit to point the new ref at, and a ref
        # file containing nothing is not a thing this format has.
        raise UsageError(
            f"cannot create branch '{name}': {start_point} does not name a "
            f"commit yet (no commits on this branch)"
        )

    repo.update_ref(name, start)
    return {
        "action": "create", "branch": name, "commit": start,
        "start_point": start_point,
    }


def _list(repo: Repo, head_branch, head_commit) -> dict:
    """Every branch with its tip's short hash and message (CLI.md ~5's example).

    The message costs one object read per branch, which is why it is not
    behind a flag: branch counts are small by construction, unlike the commit
    counts `log --no-size` guards against.
    """
    store = repo.store
    branches = []
    for name, commit_hash in repo.list_branches().items():
        commit = graph.get_json(store, commit_hash)
        branches.append({
            "branch": name,
            "commit": commit_hash,
            "message": commit.get("message", ""),
            "current": name == head_branch,
        })
    return {
        "action": "list",
        "branches": branches,
        "head": head_commit,
        "current": head_branch,
        "detached": head_branch is None,
    }


def format_human(result: dict) -> str:
    action = result["action"]
    if action == "create":
        return f"Created branch '{result['branch']}' at {result['commit'][:6]}"
    if action == "delete":
        return f"Deleted branch '{result['branch']}' (was {result['commit'][:6]})"
    if action == "rename":
        return f"Renamed branch '{result['from']}' to '{result['branch']}'"

    branches = result["branches"]
    if not branches:
        return "No branches yet (HEAD is unborn -- make a commit first)"

    width = max(len(b["branch"]) for b in branches)
    lines = []
    if result.get("detached") and result.get("head"):
        lines.append(f"* (HEAD detached at {result['head'][:6]})")
    for entry in branches:
        marker = "*" if entry["current"] else " "
        lines.append(
            f"{marker} {entry['branch']:<{width}}  {entry['commit'][:6]}  "
            f"{entry['message']}"
        )
    return "\n".join(lines)
