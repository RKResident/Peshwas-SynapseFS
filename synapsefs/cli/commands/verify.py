"""`synapsefs verify` -- CLI.md ~7.

    synapsefs verify [<ref>] [--shallow | --fast | --deep] [--content]
                     [--packs] [--all] [--json]

Walks and cryptographically verifies lineage. Independent of `checkout` and
`mount` by construction: it talks to the object store and pack set directly,
so integrity can be graded on a repo whose FUSE mount does not work at all.

The default tier is `--deep` (FORMAT.md 12B). That is a deliberate departure
from CLI.md's original table, which made the checksum tier the default: the
checksum tier compares a payload against a value stored in the pack index,
and an attacker who rewrote the payload rewrote the index too. It detects
rot, never tampering. Since PS module 2b asks specifically for malicious
block injection to be rejected, the tier that can actually do that is the one
that runs when you type `synapsefs verify`.
"""

from __future__ import annotations

import argparse
from typing import List

from synapsefs import verify as verify_engine
from synapsefs.cli import output
from synapsefs.errors import IntegrityError, UsageError
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "verify",
        parents=[global_parser],
        help="Cryptographically verify lineage",
    )
    parser.add_argument(
        "ref", nargs="?", default="HEAD",
        help="Where to start (branch, commit hash, or HEAD). Every ancestor"
             " reachable through every parent is verified.",
    )
    tier = parser.add_mutually_exclusive_group()
    tier.add_argument(
        "--shallow", action="store_true",
        help="Object graph only: re-hash every loose object and probe that"
             " every chunk exists. Touches no payload.",
    )
    tier.add_argument(
        "--fast", action="store_true",
        help="Above, plus each chunk's stored payload against the pack index"
             " checksum. No decompression. Detects bit-rot only -- it cannot"
             " detect tampering, because the index is as writable as the pack.",
    )
    tier.add_argument(
        "--deep", action="store_true",
        help="The default. Above, plus decompress every chunk and re-hash it"
             " against the hash its tensor-manifest names.",
    )
    parser.add_argument(
        "--content", action="store_true",
        help="Additionally reconstruct every tensor and check it against its"
             " manifest's content_hash. Much slower -- this is the only check"
             " that requires reconstruction -- but the only one that catches a"
             " permutation applied in the wrong order.",
    )
    parser.add_argument(
        "--packs", action="store_true",
        help="Additionally re-hash each pack file against its own trailer."
             " Off by default: at the deep tier every referenced byte is"
             " already checked against a stronger, ref-anchored hash, so this"
             " only adds coverage of framing and unreferenced regions.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Verify every branch, not just <ref>'s ancestry.",
    )
    parser.set_defaults(func=run, format_human=format_human)


def _tier(args: argparse.Namespace) -> str:
    if args.shallow:
        return verify_engine.STRUCTURE
    if args.fast:
        return verify_engine.CHECKSUM
    return verify_engine.CONTENT


def run(args: argparse.Namespace) -> dict:
    repo = Repo.find(args.repo)

    if args.all:
        branches = repo.list_branches()
        if not branches:
            raise UsageError("no branches to verify")
        roots: List[str] = sorted(set(branches.values()))
        scope = f"all branches ({len(branches)})"
    else:
        head = repo.resolve_ref(args.ref)
        if head is None:
            raise UsageError(
                f"{args.ref!r} does not name a commit yet (nothing to verify)"
            )
        roots = [head]
        scope = args.ref

    report = verify_engine.verify_lineage(
        repo, roots,
        tier=_tier(args),
        check_content_hash=args.content,
        verify_packs=args.packs,
    )

    result = report.as_dict()
    result["scope"] = scope
    result["roots"] = roots

    if not report.ok:
        # CLI.md ~1.3 reserves exit 4 for exactly this. Raising rather than
        # returning means the failure detail has to travel on the exception,
        # so the summary is rendered here and the structured list goes with
        # it -- `--json` consumers get `failures` either way because main.py
        # prints the message and the caller re-runs with --json for detail.
        raise VerifyFailed(_failure_message(result), result)
    return result


class VerifyFailed(IntegrityError):
    """Carries the full report alongside the message, so `--json` output and
    the exit code stay consistent with each other."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def _failure_message(result: dict) -> str:
    lines = [
        f"FAIL -- {len(result['failures'])} integrity failure(s) in "
        f"{result['commits']} commit(s)"
    ]
    for failure in result["failures"][:10]:
        lines.append(f"  {failure['kind']}: {failure['object'][:16] or '(pack)'}")
        if failure.get("expected") and failure.get("actual"):
            lines.append(
                f"       expected {failure['expected'][:16]}...  "
                f"got {failure['actual'][:16]}..."
            )
        if failure.get("pack"):
            lines.append(f"       in pack {failure['pack']}")
        if failure.get("referenced_by"):
            lines.append(f"       referenced by {failure['referenced_by']}")
        if failure.get("detail"):
            lines.append(f"       {failure['detail']}")
    if len(result["failures"]) > 10:
        lines.append(f"  ... and {len(result['failures']) - 10} more")
    if result.get("truncated"):
        lines.append("  (failure list truncated)")
    return "\n".join(lines)


def format_human(result: dict) -> str:
    """Matches CLI.md ~7's worked example on the success path."""
    tier = result["tier"]
    extras = []
    if result.get("content_hash_checked"):
        extras.append("+content_hash")
    suffix = f" [{tier}{' ' + ' '.join(extras) if extras else ''}]"

    lines = [
        f"Verifying lineage for {result['scope']!r} "
        f"({result['commits']} commits){suffix}",
        f"  commits {result['commits']}   manifests {result['manifests']}   "
        f"chunks {result['chunks']}   objects {result['objects']}",
    ]
    rate = (
        result["bytes_verified"] / result["elapsed"]
        if result["elapsed"] > 0 else 0.0
    )
    lines.append(
        f"OK  -- {result['commits']} commits, {result['chunks']} chunks, "
        f"{output.human_bytes(result['bytes_verified'])} verified in "
        f"{result['elapsed']:.2f}s ({output.human_bytes(rate)}/s)"
    )
    if tier != verify_engine.CONTENT:
        lines.append(
            f"NOTE: the {tier!r} tier detects corruption, not tampering -- "
            f"its reference hashes live in files an attacker also controls. "
            f"Run without --{'shallow' if tier == 'structure' else 'fast'} "
            f"for the ref-anchored check."
        )
    return "\n".join(lines)
