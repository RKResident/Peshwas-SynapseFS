"""stdout/stderr discipline (CLI.md ~1.2)::

    stdout - results only. Under --json, exactly one JSON document, nothing else.
    stderr - progress, warnings, errors. Always.

The benchmark harness parses stdout, so this module is deliberately the
*only* place that should ever write a command's result to stdout. Keeping
that centralized is what prevents a stray print() inside some command
module from leaking a progress line onto stdout and silently breaking the
harness's parse.
"""

from __future__ import annotations

import argparse
import json
import sys


def emit(result: dict, args: argparse.Namespace, human: str) -> None:
    """Print a command's result to stdout in the form the caller asked for.

    `--json`: one JSON document (`result`, machine-readable), nothing else.
    otherwise: `human`, a single pre-formatted string the command module
    built from that same `result` dict (see
    `cli/commands/init.py::format_human` for the pattern) -- kept as one
    string rather than several print() calls, so a command can't
    accidentally interleave a stray write from somewhere else in its logic.

    `--quiet` is deliberately not handled here: per CLI.md it suppresses
    *progress*, which belongs on stderr and is each command's own concern
    to manage, not this function's. The result itself is always emitted.
    """
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(result) + "\n")
    else:
        sys.stdout.write(human + "\n")


def human_bytes(n: float) -> str:
    """Render a byte count the way CLI.md's worked examples do (e.g. `3.2 GiB`).

    Lives here rather than in any one command because `commit`, `log`,
    `checkout` and `compare` all print sizes, and four independently-written
    roundings would eventually disagree with each other in the demo output.
    """
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable, see loop above
