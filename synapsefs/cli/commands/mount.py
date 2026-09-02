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


#: Shown as the `mount --help` epilog. The default is tuned for streaming
#: reads, and is actively bad for memory-mapped ones -- a difference big
#: enough that leaving it undocumented would be a trap rather than a default.
_CACHE_SIZING_HELP = """
CACHE SIZING
  --cache-size caps the decoded-chunk cache. FUSE hands out small reads against
  4 MiB chunks, so on a miss a whole chunk -- plus the chunk it is a delta
  against -- is decoded to serve a fraction of it. The cache exists to absorb
  that, and it needs to hold the WORKING SET, which is about 64 MiB per
  checkpoint being read concurrently:

      cache >= 64 MiB x (number of DISTINCT files read at the same time)

  Distinct is the word that matters. Eight processes loading the SAME
  checkpoint all want the same chunk at the same moment, so they share one
  working set and the default is plenty. Eight processes loading eight
  DIFFERENT commits share nothing, need eight working sets, and will thrash a
  small cache -- decoding, evicting and re-decoding the same chunks.

  Measured on a 25-epoch 90M checkpoint, 8 concurrent readers, cold, with
  aggregate throughput, the bytes the daemon read, and that as a multiple of
  one cold pass over the distinct files:

    8 readers, SAME file (e.g. DDP ranks)
         32 MiB      170 MB/s     1.9 GiB   6.6x    262 MiB RSS   <- default
         64 MiB      183 MB/s     0.7 GiB   2.4x    294 MiB RSS
        128 MiB      159 MB/s     0.6 GiB   1.9x    332 MiB RSS

    8 readers, 8 DISTINCT commits
         32 MiB       30 MB/s    69.5 GiB  29.7x    301 MiB RSS
        256 MiB       59 MB/s    17.6 GiB   7.7x    567 MiB RSS
        512 MiB       78 MB/s     3.3 GiB   1.4x    837 MiB RSS
       1024 MiB       83 MB/s     3.0 GiB   1.3x   1372 MiB RSS

  Both follow the rule: 64 MiB x distinct files puts you near the plateau, and
  512 MiB is the knee for eight of them. Past the plateau you buy little.

  The default is adequate for the same-file case -- 170 MB/s is usable -- but
  it is not free there either: it runs 6.6x amplified, and 64 MiB cuts that to
  2.4x for 32 MiB more RSS. If you are serving several different checkpoints at
  once (a sweep, a multi-commit eval, a diffing tool), the default is genuinely
  bad and you want the rule.

  Amplification does not go below ~2x at any cache size, because a residual
  chunk's delta base is decoded every time the chunk is, and that decode is not
  cached. That floor is a property of the reconstruction path, not of this
  setting.
"""


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CACHE_SIZING_HELP,
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
            "Hard cap on the decoded-chunk cache, in bytes (default: "
            f"{DEFAULT_CACHE_SIZE_BYTES // (1024 * 1024)} MiB). Size it by how many "
            "DISTINCT checkpoints the mount serves at once, roughly 64 MiB each -- "
            "not by the number of readers, who share a working set when they read "
            "the same file. See CACHE SIZING below."
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

