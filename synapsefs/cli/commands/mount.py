"""`synapsefs mount` -- CLI.md ~11.

    synapsefs mount <mountpoint> [--ref <branch|commit>] [--foreground]
                    [--cache-size <bytes>] [--allow-other] [--debug-fuse]

Read-only POSIX mount. Daemonizes unless `--foreground`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from synapsefs.fuse.cache import DEFAULT_CACHE_SIZE_BYTES
from synapsefs.fuse.daemon import mount_fuse
from synapsefs.store.repo import Repo


def _human_bytes(n: float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} {unit}" if value.is_integer() else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    """Register `mount` arguments on the shared subparsers object."""
    parser = subparsers.add_parser(
        "mount",
        parents=[global_parser],
        help="Mount a read-only virtual filesystem",
    )
    parser.add_argument(
        "mountpoint",
        help="Directory to mount the virtual filesystem at",
    )
    parser.add_argument(
        "--ref", default=None, type=str,
        help="Restrict the namespace to one branch or commit ref",
    )
    parser.add_argument(
        "--foreground", action="store_true",
        help="Run in the foreground rather than daemonizing",
    )
    parser.add_argument(
        "--cache-size", default=DEFAULT_CACHE_SIZE_BYTES, type=int,
        help=(
            "Hard cap on decoded-chunk cache in bytes "
            f"(default: {DEFAULT_CACHE_SIZE_BYTES // (1024 * 1024)} MiB). Raise it only "
            "for workloads that re-read the same chunks; a larger cache costs RSS "
            "roughly one-for-one and buys no throughput past the in-flight set."
        ),
    )
    parser.add_argument(
        "--allow-other", action="store_true",
        help="Allow other users access (requires user_allow_other in /etc/fuse.conf)",
    )
    parser.add_argument(
        "--debug-fuse", action="store_true",
        help="Enable FUSE protocol tracing",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    repo = Repo.find(args.repo)
    mountpoint = Path(args.mountpoint)

    return mount_fuse(
        repo=repo,
        mountpoint=mountpoint,
        ref=args.ref,
        foreground=args.foreground,
        cache_size=args.cache_size,
        allow_other=args.allow_other,
        debug_fuse=args.debug_fuse,
    )


def format_human(result: dict) -> str:
    """Human-readable rendering matching CLI.md ~11:

    Mounted /home/u/repo at /mnt/syn (read-only, cache 512 MiB)
    """
    repo = result.get("repo", "")
    mountpoint = result.get("mountpoint", "")
    cache_str = _human_bytes(result.get("cache_size", 512 * 1024 * 1024))
    return f"Mounted {repo} at {mountpoint} (read-only, cache {cache_str})"

