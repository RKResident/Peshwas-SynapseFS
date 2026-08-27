"""`synapsefs restore` -- reconstruct a commit's weights, and check them.

    synapsefs restore <ref> [--out <path>] [--compare <checkpoint.safetensors>]
                      [--strict] [--all]

Not in CLI.md's required set. It exists because `checkout` couples
reconstruction to moving HEAD, and there are two situations where you want the
first without the second:

* **Inspecting a commit.** Pulling epoch 3's weights out to a scratch path to
  load in torch, without disturbing which commit the repo is on.
* **Proving the codec is lossless.** `--compare` streams the commit and a
  reference `.safetensors` past each other and reports, per tensor, how many
  elements differ and by how many ULPs. For a correct reconstruction every
  number is zero -- and "zero" is a claim you can check yourself on your own
  checkpoints rather than take from a test suite.

The ULP column is the one that matters. Comparing floats with a tolerance
answers "is this close enough"; this codec's claim is stronger than that --
the residual is exact integer arithmetic, so the answer must be *bit-equal*,
and a single differing mantissa bit should show up rather than round away.
"""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path
from typing import List, Optional

from synapsefs import graph
from synapsefs.cli import output
from synapsefs.errors import IntegrityError, UsageError
from synapsefs.materialize import compare_sources, materialize
from synapsefs.pack.packset import PackSet
from synapsefs.safetensors_io import SafetensorsFile
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "restore",
        parents=[global_parser],
        help="Reconstruct a commit's checkpoint and optionally check it "
             "against a reference file",
    )
    parser.add_argument(
        "ref", nargs="?", default="HEAD",
        help="Commit to reconstruct (branch, hash, or HEAD).",
    )
    parser.add_argument(
        "--out", default=None, metavar="<path>",
        help="Write the reconstructed checkpoint here. HEAD is never moved.",
    )
    parser.add_argument(
        "--compare", default=None, metavar="<checkpoint.safetensors>",
        help="Reference file to check the reconstruction against, tensor by"
             " tensor.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Report every tensor, not just the ones that differ.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 4 if the reconstruction is not identical to --compare."
             " Off by default because two genuinely different checkpoints are"
             " a legitimate thing to diff; on, this is a lossless-codec"
             " assertion suitable for CI.",
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    repo = Repo.find(args.repo)

    if args.out is None and args.compare is None:
        raise UsageError(
            "restore needs --out (where to write) or --compare (what to check "
            "against), or both"
        )

    commit_hash = repo.resolve_ref(args.ref)
    if commit_hash is None:
        raise UsageError(f"{args.ref!r} does not name a commit yet")

    reference: Optional[Path] = None
    if args.compare is not None:
        reference = Path(args.compare)
        if not reference.is_file():
            raise UsageError(f"reference checkpoint not found: {reference}")

    store = repo.store
    commit = graph.get_json(store, commit_hash)

    written: Optional[dict] = None
    comparisons: List[dict] = []
    identical_bytes: Optional[bool] = None

    with PackSet(
        repo.objects_dir / "pack", tmp_dir=repo.objects_dir / "tmp"
    ) as packs:
        source = graph.CommitCheckpoint(store, packs, commit_hash)

        if args.out is not None:
            written = materialize(source, source.header_bytes, Path(args.out))

        if reference is not None:
            with SafetensorsFile(reference) as ref_file:
                # Compared against the *live* commit, not against whatever was
                # written above. That keeps the answer meaningful when --out
                # was not given, and it isolates the codec from the writer: if
                # these agree but the files below do not, the bug is in
                # materialize(), not in the residual chain.
                comparisons = [
                    _as_dict(item)
                    for item in compare_sources(source, ref_file)
                ]

            if written is not None:
                # The stronger, end-to-end check: whole files, headers
                # included. This is the property CLI.md ~4 actually requires,
                # and the tensor table above cannot see a header difference.
                identical_bytes = filecmp.cmp(
                    Path(written["path"]), reference, shallow=False
                )

    differing = [c for c in comparisons if c["status"] != "identical"]
    result = {
        "commit": commit_hash,
        "message": commit.get("message", ""),
        "written": written,
        "reference": str(reference) if reference else None,
        "tensors_compared": len(comparisons),
        "tensors_differing": len(differing),
        "identical": bool(comparisons) and not differing,
        "identical_bytes": identical_bytes,
        "comparisons": comparisons if args.all else differing,
        "show_all": args.all,
    }

    if args.strict and reference is not None:
        if differing or identical_bytes is False:
            # A verification-class failure in the strict sense CLI.md ~1.3
            # reserves exit 4 for: the bytes this repo returns are not the
            # bytes it was given.
            raise IntegrityError(
                f"commit {commit_hash[:8]} does not reconstruct to {reference}: "
                f"{len(differing)} of {len(comparisons)} tensors differ"
                + ("" if identical_bytes is not False else "; files differ byte-wise")
            )
    return result


def _as_dict(item) -> dict:
    return {
        "name": item.name,
        "status": item.status,
        "elements": item.elements,
        "mismatched": item.mismatched,
        "max_abs_diff": item.max_abs_diff,
        "mean_abs_diff": item.mean_abs_diff,
        "max_ulp_diff": item.max_ulp_diff,
    }


def format_human(result: dict) -> str:
    lines = []
    written = result.get("written")
    if written:
        lines.append(
            f"Wrote {written['path']} "
            f"({output.human_bytes(written['total_bytes'])}, "
            f"{written['tensors']} tensors)"
        )

    if result.get("reference") is None:
        return "\n".join(lines) or f"Reconstructed {result['commit'][:6]}"

    lines.append(f"Comparing {result['commit'][:6]} against {result['reference']}")

    rows = result.get("comparisons") or []
    if rows:
        width = max(len(r["name"]) for r in rows)
        lines.append(
            f"  {'tensor':<{width}}  {'status':<14}  "
            f"{'differing':>12}  {'max |d|':>12}  {'max ULP':>8}"
        )
        for row in rows:
            share = (
                f"{row['mismatched']}/{row['elements']}"
                if row["elements"] else "-"
            )
            lines.append(
                f"  {row['name']:<{width}}  {row['status']:<14}  "
                f"{share:>12}  {row['max_abs_diff']:>12.6g}  "
                f"{row['max_ulp_diff']:>8}"
            )
    elif not result.get("show_all"):
        lines.append("  (no differing tensors)")

    total = result["tensors_compared"]
    diff = result["tensors_differing"]
    lines.append(
        f"  {total - diff}/{total} tensors bit-identical"
        + ("" if diff == 0 else f", {diff} differ")
    )
    if result.get("identical_bytes") is not None:
        verdict = "byte-identical" if result["identical_bytes"] else "DIFFERENT"
        lines.append(f"  whole-file comparison: {verdict}")
    return "\n".join(lines)
