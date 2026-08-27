"""`synapsefs unmount` -- CLI.md ~11.

    synapsefs unmount <mountpoint>

Unmounts a previously mounted virtual filesystem, terminating its daemon
and falling back to fusermount3 if necessary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synapsefs.fuse.daemon import unmount_fuse
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    """Register `unmount` arguments on the shared subparsers object."""
    parser = subparsers.add_parser(
        "unmount",
        parents=[global_parser],
        help="Unmount a mounted virtual filesystem",
    )
    parser.add_argument(
        "mountpoint",
        help="Mountpoint directory to unmount",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    try:
        repo = Repo.find(args.repo)
    except Exception:
        # If run outside repo, create a dummy repo wrapper for finding records
        repo = Repo(Path.cwd())

    mountpoint = Path(args.mountpoint)
    return unmount_fuse(repo, mountpoint)


def format_human(result: dict) -> str:
    """Human-readable rendering matching CLI.md ~11:

    Unmounted /mnt/syn
    """
    mountpoint = result.get("mountpoint", "")
    return f"Unmounted {mountpoint}"

