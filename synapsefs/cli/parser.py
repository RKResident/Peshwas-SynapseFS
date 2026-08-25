"""Top-level argparse wiring.

This module knows *that* subcommands exist and how global flags are shared
across the top-level parser and every subcommand parser (CLI.md ~1.1:
global flags are "accepted before or after the subcommand"). It does not
know *how* any individual subcommand works -- that lives in
`cli/commands/<name>.py`. Adding a new command should only ever require a
new file in `cli/commands/` plus one line in `_COMMAND_MODULES` below;
nothing here should need to change per-command.
"""

from __future__ import annotations

import argparse

from synapsefs import __version__
from synapsefs.cli.commands import init as init_cmd

# Every module listed here must expose add_subparser(subparsers, global_parser).
# Add commit, checkout, branch, log, verify, push, pull, serve, merge, mount,
# unmount here as each one lands.
_COMMAND_MODULES = [init_cmd]


def _build_global_parser() -> argparse.ArgumentParser:
    """Parser holding only the global flags (CLI.md ~1.1).

    Used as a `parents=[...]` base for both the top-level parser and every
    subcommand parser, so the same flags are recognized whichever parser
    actually consumes them.

    Every argument here uses `default=argparse.SUPPRESS`, not a real
    default. This matters and is worth understanding precisely, because the
    naive approach (a normal default on both parsers, relying on a shared
    namespace) is subtly broken: `argparse`'s `_SubParsersAction` parses the
    subcommand's remaining args into a **fresh** namespace, not the
    top-level one, and then copies *every* attribute from that fresh
    namespace onto the real one -- unconditionally, including whatever
    default the subparser's copy of `--json` has, even if the flag was
    never mentioned again after the subcommand name. So `synapsefs --json
    init myrepo` would parse `--json` correctly at the top level, then have
    it silently clobbered back to `False` during the merge, since the `init`
    subparser's own `--json` (unset in its remaining args) reasserts its
    default. `synapsefs init myrepo --json` happens to work by accident,
    because in that ordering the subparser genuinely sees `--json` in its
    own slice of argv.

    `default=argparse.SUPPRESS` avoids this: an argument that wasn't
    actually passed to *this* particular parse call gets no attribute at
    all on that fresh namespace, so the merge step has nothing to
    overwrite the other parser's value with. The real baseline defaults
    (`json=False`, etc.) are established once via `set_defaults()` on the
    top-level parser only, in `build_parser()` below -- see there for why
    that's the right place for them.

    `add_help=False` avoids a duplicate `-h/--help` registration error when
    this parser is reused as a `parents=` base elsewhere.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-C", "--repo", default=argparse.SUPPRESS, metavar="<path>",
        help="Operate on the repo at <path> (default: search upward from cwd)",
    )
    parser.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="Emit machine-readable JSON on stdout instead of human text",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="Suppress progress output on stderr",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="Per-tensor detail",
    )
    parser.add_argument(
        "--no-color", action="store_true", default=argparse.SUPPRESS,
        help="Disable ANSI color (auto-disabled when stdout is not a TTY)",
    )
    return parser


# Baseline values for the global flags, applied once parsing is fully done
# (see parse_args() below) -- deliberately NOT via ArgumentParser.set_defaults()
# on the top-level parser. That would be the obvious way to do it, but it's
# wrong here: parents=[global_parser] shares the *same* Action objects
# between the top parser and every subparser (not copies), and
# set_defaults() mutates an action's `.default` in place wherever it's
# found in self._actions. Calling top_parser.set_defaults(json=False) would
# therefore silently overwrite the shared --json action's default from
# SUPPRESS back to False on *every* subparser too -- reintroducing the exact
# clobbering bug the SUPPRESS trick above exists to prevent. Confirmed by
# inspecting argparse.ArgumentParser.set_defaults's source directly rather
# than assumed; it's a genuinely easy trap to fall into with `parents=`.
_GLOBAL_DEFAULTS = {
    "repo": ".", "json": False, "quiet": False, "verbose": False, "no_color": False,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the full `synapsefs` argument parser."""
    global_parser = _build_global_parser()

    parser = argparse.ArgumentParser(prog="synapsefs", parents=[global_parser])
    parser.add_argument(
        "--version", action="version", version=f"synapsefs {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module in _COMMAND_MODULES:
        module.add_subparser(subparsers, global_parser)

    return parser


def parse_args(parser: argparse.ArgumentParser, argv):
    """Parse `argv` and fill in baseline values for any global flag that
    ended up unset (i.e. the user never passed it, in either position).

    This is the one place `_GLOBAL_DEFAULTS` gets applied, and it happens
    strictly *after* `parser.parse_args()` returns -- a plain post-processing
    pass over the resulting Namespace, touching only that Namespace object,
    never any Action's `.default`. main.py should always go through this
    function rather than calling `parser.parse_args()` directly.
    """
    args = parser.parse_args(argv)
    for dest, default in _GLOBAL_DEFAULTS.items():
        if not hasattr(args, dest):
            setattr(args, dest, default)
    return args
