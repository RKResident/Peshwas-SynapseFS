"""`synapsefs log` -- CLI.md ~6.

    synapsefs log [<ref>] [-n <count>] [--graph] [--oneline]

Walks the commit DAG and reports what each commit cost. Read-only: it opens
the object store and (for sizes) the pack indexes, and writes nothing.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from synapsefs import graph
from synapsefs.cli import output
from synapsefs.errors import UsageError
from synapsefs.pack.packset import PackSet
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "log",
        parents=[global_parser],
        help="Show commit history",
    )
    parser.add_argument(
        "ref", nargs="?", default="HEAD",
        help="Where to start walking (branch, commit hash, or HEAD).",
    )
    parser.add_argument(
        "-n", "--max-count", default=None, type=int, metavar="<count>",
        help="Show at most <count> commits.",
    )
    parser.add_argument(
        "--graph", action="store_true",
        help="Show every parent of each commit, not just the first.",
    )
    parser.add_argument(
        "--oneline", action="store_true",
        help="One commit per line.",
    )
    parser.add_argument(
        "--no-size", action="store_true",
        help="Skip the stored/original byte figures. They are derived, not"
             " recorded, so computing them reads every tensor-manifest of"
             " every commit listed -- worth skipping on a deep history.",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    """Walk from `ref` and describe each commit.

    Sizes are recomputed from the object graph rather than read off the commit
    (see `graph.checkpoint_sizes` for why they are not stored there), which is
    the one part of this command that is not O(1) per commit. `--no-size`
    exists for that reason.
    """
    repo = Repo.find(args.repo)

    if args.max_count is not None and args.max_count <= 0:
        raise UsageError(f"-n must be positive, got {args.max_count}")

    start = repo.resolve_ref(args.ref)
    head_branch, head_commit = repo.read_head()

    if start is None:
        # Unborn HEAD after `init`. An empty history is a legitimate answer,
        # not an error -- `git log` on a fresh repo fails, but there is nothing
        # to be gained from making a script special-case that here.
        return {
            "ref": args.ref, "head": None, "branch": head_branch, "commits": [],
            "oneline": args.oneline, "graph": args.graph,
        }

    store = repo.store
    walk = graph.walk_first_parent(store, start, limit=args.max_count)

    decorations = _decorations(repo, head_branch, head_commit)

    commits: List[dict] = []
    with PackSet(repo.objects_dir / "pack", tmp_dir=repo.objects_dir / "tmp") as packs:
        for commit_hash, commit in walk:
            entry = {
                "commit": commit_hash,
                "parents": list(commit.get("parents") or []),
                "message": commit.get("message", ""),
                "timestamp": commit.get("timestamp", ""),
                "full": bool(commit.get("full", False)),
                "checkpoint_name": commit.get("checkpoint_name"),
                "refs": decorations.get(commit_hash, []),
            }
            if not args.no_size:
                entry.update(graph.checkpoint_sizes(store, packs, commit_hash))
            commits.append(entry)

    return {
        "ref": args.ref,
        "head": head_commit,
        "branch": head_branch,
        "commits": commits,
        "oneline": args.oneline,
        "graph": args.graph,
    }


def _decorations(
    repo: Repo, head_branch: Optional[str], head_commit: Optional[str]
) -> Dict[str, List[str]]:
    """`commit hash -> ["HEAD -> main", "experiment", ...]`.

    Built once for the whole walk rather than per commit, so listing a long
    history still reads `refs/heads/` exactly once.
    """
    out: Dict[str, List[str]] = {}
    branches = repo.list_branches()
    for name, commit_hash in branches.items():
        label = f"HEAD -> {name}" if name == head_branch else name
        out.setdefault(commit_hash, []).append(label)
    if head_branch is None and head_commit is not None:
        # Detached: HEAD decorates a commit no branch names.
        out.setdefault(head_commit, []).insert(0, "HEAD")
    return out


def _size_suffix(entry: dict) -> str:
    if "stored_bytes" not in entry:
        return ""
    size = output.human_bytes(entry["stored_bytes"])
    return f"   {size} (full)" if entry["full"] else f"   {size}"


def format_human(result: dict) -> str:
    """Render the walk, matching CLI.md ~6's worked example under `--oneline`."""
    commits = result["commits"]
    if not commits:
        branch = result.get("branch")
        return (
            f"No commits yet on branch '{branch}'" if branch
            else "No commits yet"
        )

    lines: List[str] = []
    for entry in commits:
        short = entry["commit"][:6]
        refs = f" ({', '.join(entry['refs'])})" if entry["refs"] else ""

        if result.get("oneline"):
            lines.append(
                f"{short}  {entry['message']:<16}{entry['timestamp']}"
                f"{_size_suffix(entry)}{refs}"
            )
            continue

        lines.append(f"commit {entry['commit']}{refs}")
        if result.get("graph") and len(entry["parents"]) > 1:
            lines.append(
                "Merge:   " + " ".join(p[:6] for p in entry["parents"])
            )
        lines.append(f"Date:    {entry['timestamp']}")
        if "stored_bytes" in entry:
            original = entry["original_bytes"]
            ratio = (entry["stored_bytes"] / original * 100) if original else 0.0
            kind = "full checkpoint" if entry["full"] else "residual"
            lines.append(
                f"Stored:  {output.human_bytes(entry['stored_bytes'])} of "
                f"{output.human_bytes(original)} ({ratio:.2f}%, {kind}, "
                f"{entry['tensors']} tensors)"
            )
        lines.append("")
        lines.append(f"    {entry['message']}")
        lines.append("")

    return "\n".join(lines).rstrip("\n")
