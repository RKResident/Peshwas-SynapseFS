"""`synapsefs commit` -- CLI.md ~3.

    synapsefs commit <checkpoint.safetensors> -m <message>
                     [--config <config.json>] [--base <ref>]
                     [--no-align] [--chunk-size <bytes>] [--strict]

Ingests a checkpoint, aligns it against the base, stores the residual,
writes a commit, and advances the current branch.

`--base` is always a **ref**: a branch name, a commit hash (full or
abbreviated), or `HEAD` -- resolved through `Repo.resolve_ref`
(synapsefs/store/repo.py). It is never a file path. There is no need to
reconstruct a base checkpoint from its diff series by hand here either --
that reconstruction is `Repo.resolve_ref` + the (future) object graph's
job, not this command module's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synapsefs.errors import UsageError
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    """Register this command's arguments on the shared subparsers object.

    `parents=[global_parser]` lets `--json`/`-q`/etc. be typed either before
    or after `commit` -- see `init.py::add_subparser` and parser.py's
    `_build_global_parser` docstring for why that works.
    """
    parser = subparsers.add_parser(
        "commit",
        parents=[global_parser],
        help="Create a new commit",
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint to commit. Expects a safetensors checkpoint.",
    )
    parser.add_argument(
        "-m", "--message", required=True, type=str,
        help="Commit message (required).",
    )
    parser.add_argument(
        "--config", default=None, type=str,
        help="Topology config. Defaults to config.json beside the checkpoint."
             " Required for the first commit; reused from the base commit"
             " afterward if omitted.",
    )
    parser.add_argument(
        "--base", default="HEAD", type=str,
        help="Base to diff against, as a ref -- a branch name, a commit hash"
             " (full or abbreviated), or HEAD. Never a file path. Default:"
             " current HEAD. Ignored on the root commit.",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip permutation matching; assume identity.",
    )
    parser.add_argument(
        "--chunk-size", default=None, type=int,
        help="Override the default chunk size, in bytes. Unset means"
             " 'use the format default' -- there is no fixed numeric"
             " default here yet.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 5 instead of 0 when a tensor is not meaningfully"
             " alignable.",
    )
    # func/format_human ride on the namespace so main.py can dispatch and
    # render without knowing anything about "commit" specifically.
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    """Do the actual work. Returns a plain dict -- never prints anything
    itself; cli/output.py decides how the result is rendered.

    This only implements the argument-and-ref plumbing that does not
    require the align/codec/pack modules (none of which exist yet). It
    validates everything it can, resolves `--base` to a commit hash, and
    then raises NotImplementedError for the encode/align/store step so the
    plumbing above it is genuinely exercised by tests instead of being
    dead code behind a stub.
    """
    repo = Repo.find(args.repo)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise UsageError(f"checkpoint not found or not a file: {checkpoint}")

    config_path = (
        Path(args.config) if args.config is not None
        else checkpoint.parent / "config.json"
    )

    # --base defaults to "HEAD". resolve_ref() returns None precisely when
    # HEAD is unborn (no commit yet reachable from it) -- CLI.md ~3 says
    # --base is "Ignored" on the root commit, so that None is expected here,
    # not an error. Any other unresolvable ref (bad branch name, unknown or
    # ambiguous hash, ...) still raises UsageError from resolve_ref itself.
    base_hash = repo.resolve_ref(args.base)
    is_root_commit = base_hash is None

    if not config_path.is_file():
        if is_root_commit:
            raise UsageError(
                f"--config is required for the first commit "
                f"(no config found at {config_path})"
            )
        # Non-root commit with no local config file: CLI.md ~3 says it's
        # reused from the base commit instead. Reading it back out of the
        # base commit's manifest needs the object graph, which doesn't
        # exist yet -- see the NotImplementedError below.



    raise NotImplementedError(
        "commit: encode/align/pack are not implemented yet -- waiting on "
        "synapsefs.align, synapsefs.codec, and synapsefs.pack"
    )


def _human_bytes(n: float) -> str:
    """Render a byte count the way CLI.md ~3.1's worked example does
    (e.g. `3.2 GiB`)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable, see loop above


def format_human(result: dict) -> str:
    """Human-readable rendering of run()'s result, matching CLI.md ~3.1's
    worked example::

        Aligning against 9f2c1a (16 permutation groups)
          identity permutation detected -- fast path
        Encoding 64 tensors, 3.2 GiB
          residual: 41.7 MiB (1.29% of original)
          new chunks: 412   deduped: 1088
        [main 4d8e2f] epoch 3

    Driven by the keys in that section's `--json` block: `commit`,
    `branch`, `base`, `original_bytes`, `residual_bytes`, `residual_ratio`,
    `tensors`, `chunks_new`, `chunks_deduped`, and
    `alignment.{groups, identity}`. Not yet exercised by any test --
    `run()` cannot produce a result dict until the encode/align/pack
    modules land -- so this uses `.get()` defensively rather than assuming
    every key mentioned above is guaranteed to exist.
    """
    alignment = result.get("alignment") or {}
    lines = []
    if alignment.get("groups"):
        base = result.get("base") or "root"
        lines.append(
            f"Aligning against {base[:6]} ({alignment['groups']} permutation groups)"
        )
        if alignment.get("identity"):
            lines.append("  identity permutation detected -- fast path")
    lines.append(
        f"Encoding {result.get('tensors', 0)} tensors, "
        f"{_human_bytes(result.get('original_bytes', 0))}"
    )
    lines.append(
        f"  residual: {_human_bytes(result.get('residual_bytes', 0))} "
        f"({result.get('residual_ratio', 0.0) * 100:.2f}% of original)"
    )
    lines.append(
        f"  new chunks: {result.get('chunks_new', 0)}   "
        f"deduped: {result.get('chunks_deduped', 0)}"
    )
    commit_hash = (result.get("commit") or "??????")[:6]
    lines.append(
        f"[{result.get('branch', '?')} {commit_hash}] {result.get('message', '')}"
    )
    return "\n".join(lines)
