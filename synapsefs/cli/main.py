"""Process entry point.

Wired up two ways (see pyproject.toml):

  - `python -m synapsefs.cli.main ...`  -- works immediately, no install,
    the fast loop to use while iterating.
  - the `synapsefs` console script pip generates from `[project.scripts]`
    once the package is installed -- what CLI.md's examples assume.

This file is the *only* place that calls sys.exit() and the *only* place
that catches SynapseError. Every subcommand raises typed exceptions
(synapsefs/errors.py) and returns a plain result dict; it never touches the
process exit code or stdout/stderr formatting directly.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from synapsefs.cli import output
from synapsefs.cli.parser import build_parser, parse_args
from synapsefs.errors import SynapseError


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    # Always go through parser.py's parse_args() wrapper here, not
    # parser.parse_args() directly -- it fills in the global-flag baseline
    # defaults *after* parsing, which is required for --json/-q/etc. to work
    # correctly regardless of whether they appear before or after the
    # subcommand. See parser.py's _build_global_parser and _GLOBAL_DEFAULTS
    # docstrings for why that can't just be a set_defaults() call.
    args = parse_args(parser, argv)

    if not hasattr(args, "func"):
        # No subcommand given -- CLI.md doesn't define a bare-invocation
        # behavior, so treat it as a usage error like git does.
        parser.print_usage(sys.stderr)
        return 2

    try:
        result = args.func(args)
    except SynapseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - genuinely unexpected bug
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1

    output.emit(result, args, args.format_human(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
