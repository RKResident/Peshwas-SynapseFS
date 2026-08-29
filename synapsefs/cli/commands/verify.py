"""`synapsefs verify` -- CLI.md ~7.

    synapsefs verify [<ref>] [--shallow | --fast | --deep] [--content]
                     [--all] [--json]

Walks and cryptographically verifies lineage. Independent of `checkout` and
`mount` by construction: it talks to the object store and pack set directly,
so integrity can be graded on a repo whose FUSE mount does not work at all.

`--deep` is the default. Since chunks became loose objects, `--fast` is also
ref-anchored -- its checksum lives in the tensor-manifest, which the commit
hash covers -- so it detects substitution too, at ~2.9x the throughput.
`--deep` stays the default because it additionally rules out a forged 8-byte
prefix, and because it is cheap: it hashes the decompressed stream, so a
residual chunk never needs its base.
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
        help="Above, plus each chunk's bytes against the checksum its"
             " tensor-manifest records. No decompression. Ref-anchored, so it"
             " detects substitution as well as rot.",
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
    if tier == verify_engine.STRUCTURE:
        lines.append(
            "NOTE: --shallow checks that chunks exist, not what is in them. "
            "Run without it to check the bytes."
        )
    return "\n".join(lines)
