"""`synapsefs merge` -- CLI.md ~8.

    synapsefs merge <branch> [-m <message>] [--ff-only] [--no-ff] [--average]

Three-way merge of two branches. Tensors only one side touched are taken from
that side; tensors both sides changed are **averaged after alignment**, which
is a stronger claim than git makes and is gated behind an explicit flag.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from synapsefs import graph, merge as merge_engine
from synapsefs.align import config_parser, solver
from synapsefs.align.Error import AlignError
from synapsefs.align.IR import TensorRef
from synapsefs.codec.checkpoint import TensorPermutation, encode_checkpoint
from synapsefs.errors import ConflictError, UsageError
from synapsefs.materialize import materialize
from synapsefs.store.atomic import TMP_MIN_AGE_SECONDS  # noqa: F401  (tmp dir contract)
from synapsefs.store.repo import Repo

CAVEAT = [
    "Averaging is not a neutral operation on weights:",
    "  * two independently-initialised models occupy different basins; their",
    "    elementwise mean is near-chance unless one is permuted into the",
    "    other's basis first. That alignment ran, and is reported above.",
    "  * even aligned, wide networks merge well and narrow ones keep a real",
    "    loss barrier. This is empirical, not guaranteed.",
    "  * SynapseFS guarantees the merge is deterministic, hash-verified and",
    "    byte-reproducible. It CANNOT tell you the merged model is any good --",
    "    that needs evaluation on data, which a storage system does not have.",
]

#: Printed only when normalisation buffers were averaged. Kept separate from
#: CAVEAT because it is not a caveat -- it is a defect in the output that the
#: user can repair, and it names the tensors to repair.
RECALIBRATE = [
    "warning: this merge contains averaged activation statistics, which are wrong.",
    "         'running_mean' and 'running_var' measure what flowed through each",
    "         PARENT. The merged network is a different function, so its true",
    "         statistics were never present in either parent and no average of",
    "         them recovers it.",
    "",
    "         Fix by resetting those buffers and re-estimating from training",
    "         data before using or evaluating the model:",
    "",
    "             for m in model.modules():",
    "                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):",
    "                     m.reset_running_stats()",
    "                     m.momentum = None      # cumulative average",
    "             model.train()",
    "             with torch.no_grad():",
    "                 for i, (x, _) in enumerate(train_loader):",
    "                     if i >= 100: break",
    "                     model(x)",
    "",
    "         Measured on two independently trained MNIST classifiers, merged",
    "         after alignment: 95.81% before recalibration, 98.15% after,",
    "         against parents at 98.47% / 98.44%.",
]


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "merge", parents=[global_parser],
        help="Merge another branch into the current one",
    )
    parser.add_argument("branch", help="Branch (or commit) to merge in.")
    parser.add_argument("-m", "--message", default=None)
    parser.add_argument("--ff-only", action="store_true",
                        help="Refuse anything that is not a fast-forward.")
    parser.add_argument("--no-ff", action="store_true",
                        help="Always create a merge commit, even if a "
                             "fast-forward were possible.")
    parser.add_argument("--average", action="store_true",
                        help="Resolve tensors both sides changed by averaging "
                             "them after alignment. Without this, such tensors "
                             "are a conflict and the merge stops (exit 6).")
    parser.set_defaults(func=run, format_human=format_human)


def _solve_alignment(store, plan, config_path: Path, notes: list):
    """Permutation bringing THEIRS into OURS' basis. None if not solvable."""
    ours = graph.CommitCheckpoint(store, plan.ours)
    refs = {n: TensorRef(n, tuple(ours.spec(n).shape), ours.spec(n).dtype)
            for n in ours.names()}
    config = None
    if config_path and config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except ValueError:
            pass
    try:
        topo = config_parser.parse(refs, config)
        result = solver.align_checkpoints(
            ours, graph.CommitCheckpoint(store, plan.theirs), topo)
    except AlignError as exc:
        notes.append(f"alignment skipped: {exc}")
        return None, None

    alignment = {}
    for name, ta in result.tensors.items():
        if ta.identity:
            continue
        verdict = result.assessments.get(name)
        if verdict is not None and not verdict.helped:
            continue
        alignment[name] = TensorPermutation(
            row=ta.pi_row, col=ta.pi_col, col_block_size=ta.col_block_size)
    return (alignment or None), result


def run(args: argparse.Namespace) -> dict:
    repo = Repo.find(args.repo)
    store = repo.store
    branch, head = repo.read_head()
    if branch is None:
        raise UsageError("cannot merge onto a detached HEAD; check out a branch")
    if head is None:
        raise UsageError("nothing to merge into: this branch has no commits")

    theirs = repo.resolve_ref(args.branch)
    if theirs is None:
        raise UsageError(f"{args.branch!r} does not name a commit")
    if theirs == head:
        return {"result": "already-up-to-date", "branch": branch,
                "commit": head, "notes": []}

    base = graph.merge_base(store, head, theirs)
    if base == theirs:
        # Their tip is already an ancestor of ours -- a previous merge, or they
        # never advanced. Checking `theirs == head` alone misses this and would
        # build a merge commit whose second parent contributes nothing.
        return {"result": "already-up-to-date", "branch": branch,
                "commit": head, "notes": []}
    if base is None:
        raise UsageError(
            f"{args.branch!r} shares no history with {branch!r}; there is no "
            f"common ancestor to merge against")

    # Fast-forward: our tip is already an ancestor of theirs, so their history
    # contains ours and moving the ref is the whole merge.
    if base == head and not args.no_ff:
        repo.update_ref(branch, theirs)
        return {"result": "fast-forward", "branch": branch, "commit": theirs,
                "base": base, "notes": []}
    if args.ff_only:
        raise UsageError(
            f"--ff-only: {branch!r} and {args.branch!r} have diverged since "
            f"{base[:8]}")

    notes: list = []
    plan = merge_engine.plan_merge(store, head, theirs, base)

    if plan.conflicts and not args.average:
        raise ConflictError(
            f"{len(plan.conflicts)} tensor(s) changed on both sides and cannot "
            f"be merged automatically: {', '.join(plan.conflicts[:5])}"
            + (" ..." if len(plan.conflicts) > 5 else "")
            + "\n       re-run with --average to resolve them by averaging "
              "after alignment (see the caveats it prints)")

    alignment = align_result = None
    if plan.conflicts:
        config_path = Path(args.repo) / "config.json"
        alignment, align_result = _solve_alignment(store, plan, config_path, notes)

    merged = merge_engine.MergedCheckpoint(store, plan, alignment)
    tmp_dir = repo.objects_dir / "tmp"
    staged = tmp_dir / f"merge-{head[:8]}-{theirs[:8]}.safetensors"
    written = materialize(merged, merged.header_bytes, staged, tmp_dir=tmp_dir)

    try:
        encoded = encode_checkpoint(
            staged, None,
            emit=lambda r: store.put_at(r.content_hash.hex(), r.payload),
            already_have=lambda h: store.has(h.hex()),
        )
        their_commit = graph.get_json(store, theirs)
        objects = graph.write_checkpoint_objects(
            store, header_bytes=encoded.header_bytes,
            manifests=encoded.manifests,
            topology_config_hash=graph.get_json(
                store, graph.get_json(store, head)["checkpoint_manifest"]
            ).get("topology_config_hash"),
            reused_manifests=encoded.reused_manifests,
        )
        message = args.message or f"Merge {args.branch} into {branch}"
        commit_hash = graph.write_commit_object(
            store, checkpoint_manifest=objects.checkpoint_manifest,
            parents=[head, theirs], message=message,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # A merge is always stored in full: it is a new checkpoint that
            # matches neither parent, so neither is a useful base for it.
            full=True,
            checkpoint_name=graph.get_json(store, head).get(
                "checkpoint_name", "model.safetensors"),
        )
        repo.update_ref(branch, commit_hash)
    finally:
        staged.unlink(missing_ok=True)

    return {
        "result": "merged", "branch": branch, "commit": commit_hash,
        "base": base, "ours": head, "theirs": theirs, "message": message,
        "tensors": len(plan.decisions), "resolution": plan.counts(),
        "conflicts": plan.conflicts,
        "averaged": len(plan.conflicts),
        "stale_statistics": plan.stale_statistics,
        "alignment": align_result.as_json() if align_result else None,
        "bytes": written["total_bytes"],
        "notes": notes,
    }


def format_human(result: dict) -> str:
    r = result["result"]
    if r == "already-up-to-date":
        return "Already up to date."
    if r == "fast-forward":
        return (f"Fast-forward to {result['commit'][:6]} "
                f"(branch '{result['branch']}')")

    lines = [f"Merging into '{result['branch']}' "
             f"(base {result['base'][:6]}, theirs {result['theirs'][:6]})"]
    for action, n in sorted(result["resolution"].items()):
        lines.append(f"  {action:<14}{n:>4} tensor(s)")
    a = result.get("alignment")
    if a:
        lines.append(f"  aligned theirs into our basis: {a['groups']} groups, "
                     f"identity={a['identity']}")
    lines.append(f"[{result['branch']} {result['commit'][:6]}] {result['message']}")
    if result["averaged"]:
        lines.append("")
        lines.extend(CAVEAT)
    stale = result.get("stale_statistics") or []
    if stale:
        lines.append("")
        lines.extend(RECALIBRATE)
        shown = ", ".join(stale[:4])
        more = f", and {len(stale) - 4} more" if len(stale) > 4 else ""
        lines.append(f"         affected ({len(stale)}): {shown}{more}")
    for n in result.get("notes", []):
        lines.append(f"  note: {n}")
    return "\n".join(lines)
