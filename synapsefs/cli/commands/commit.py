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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import json

from synapsefs import graph
from synapsefs.align import config_parser, lap, report as align_report, solver
from synapsefs.align.Error import AlignError
from synapsefs.align.IR import TensorRef
from synapsefs.cli import output
from synapsefs.codec.checkpoint import TensorPermutation
from synapsefs.codec.checkpoint import encode_checkpoint
from synapsefs.errors import NotAlignableError, UsageError
from synapsefs.store.repo import Repo

#: Fraction of tensors that must fail to align before the anchor is judged
#: worthless and the commit is stored in full instead. Above this, the base is
#: a different model rather than an earlier version of this one.
#:
#: Not fired under --no-align: that path skips the residual pass, so
#: `not_alignable` only ever contains missing or shape-mismatched tensors.
UNUSABLE_ANCHOR_FRACTION = 0.5


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
        "--timing", action="store_true",
        help="Report wall-clock per phase. Alignment and encoding scale with "
             "different things -- the solver with unit counts, the codec with "
             "bytes -- so a single total hides which one moved.",
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
    """Ingest a checkpoint, store its residual, and advance the branch.

    The whole pipeline, in order:

        resolve refs -> decide full-vs-residual (FORMAT.md 12A)
          -> open every pack for cross-commit dedup
          -> encode chunks straight into a new pack
          -> write that pack's index and register it as newest
          -> write header / tensor-manifest / checkpoint-manifest / commit
          -> move the branch

    Two ordering rules in there are load-bearing rather than stylistic:

    * The pack and its index are written and registered **before** any object
      referencing their chunks. A manifest naming a chunk that is not yet in a
      pack is a dangling reference; a pack holding chunks nothing references
      yet is merely unreferenced, and the next commit dedups against it. Only
      one of those two failure modes is recoverable.
    * The branch ref moves **last**, through `atomic_write`. Until that
      rename, a crash leaves objects and a pack on disk that no ref points at
      -- invisible, harmless, and collected later. The PS's crash requirement
      (module 2h) is satisfied by that ordering, not by a transaction.
    """
    started = time.perf_counter()
    timings: dict = {}

    repo = Repo.find(args.repo)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise UsageError(f"checkpoint not found or not a file: {checkpoint}")

    config_path = (
        Path(args.config) if args.config is not None
        else checkpoint.parent / "config.json"
    )

    # `--base` is what we *diff against*; the commit's parent is wherever the
    # branch currently points. They are the same thing in normal use, and
    # deliberately separable: choosing a cheaper diff base must not rewrite
    # history. CLI.md ~3 says --base is "Ignored" on the root commit, which is
    # exactly the None that resolve_ref returns for an unborn HEAD.
    base_hash = repo.resolve_ref(args.base)
    branch, head_commit = repo.read_head()
    if branch is None:
        raise UsageError(
            "cannot commit on a detached HEAD; check out a branch first"
        )

    if not config_path.is_file() and base_hash is None:
        raise UsageError(
            f"--config is required for the first commit "
            f"(no config found at {config_path})"
        )

    store = repo.store
    tmp_dir = repo.objects_dir / "tmp"

    # FORMAT.md 12A: commits form a star, not a chain.
    #
    # `--base` selects the *lineage*; what we actually diff against is that
    # lineage's anchor -- its nearest full checkpoint. So a residual commit is
    # never a residual-of-a-residual, and reconstructing any commit is one
    # decode on top of one full checkpoint rather than a walk. The anchor
    # actually used is reported back as `base`, so nothing is silently
    # substituted behind the caller's back.
    #
    # Every REBASE_INTERVAL commits the group starts a new anchor, which
    # bounds how far the data may drift from it. A full commit is structurally
    # identical to a root commit, so this needs no extra code path.
    anchor = (
        graph.nearest_full_ancestor(store, base_hash)
        if base_hash is not None else None
    )
    since_full = (
        graph.commits_since_full(store, head_commit)
        if head_commit is not None else 0
    )
    store_full = anchor is None or since_full >= graph.REBASE_INTERVAL - 1

    notes: list = []
    base_source = (
        None if store_full else graph.CommitCheckpoint(store, anchor)
    )

    _t = time.perf_counter()
    alignment, align_result = _align(
        store, checkpoint, base_source, config_path,
        no_align=args.no_align, notes=notes,
    )
    timings["align"] = time.perf_counter() - _t
    # FORMAT.md 12A's dynamic re-base, now that we can actually detect the
    # trigger. `nearest_full_ancestor` walks first parents for a FULL commit,
    # which across a branch boundary can land on a *different model*: branch
    # off, commit an unrelated checkpoint, and its successors keep diffing
    # against the shared root because none of them is full yet. Measured, the
    # 2nd and 3rd commit of such a branch each stored at ~94% -- essentially
    # raw -- where diffing against their real predecessor gives ~60%.
    #
    # The alignment pass already knows: if most tensors came back
    # not-alignable, the anchor has nothing useful to offer. Store this
    # commit in full instead and let it become the branch's own hub, so
    # everything after it has a base worth diffing against. One extra full
    # commit, paid once per genuine divergence.
    if not store_full and align_result is not None and align_result.tensors:
        unusable = len(align_result.not_alignable) / len(align_result.tensors)
        if unusable > UNUSABLE_ANCHOR_FRACTION:
            notes.append(
                f"anchor {anchor[:8]} unusable ({unusable*100:.0f}% of tensors "
                f"not alignable against it); storing this commit in full"
            )
            store_full, base_source, alignment = True, None, None

    if args.strict and align_result is not None and align_result.not_alignable:
        raise NotAlignableError(
            f"{len(align_result.not_alignable)} tensor(s) not meaningfully "
            f"alignable: {', '.join(align_result.not_alignable[:5])}"
            + (" ..." if len(align_result.not_alignable) > 5 else "")
        )

    # Chunks are loose, content-addressed objects like everything else
    # (ARCHITECTURE.md 3.3). Each lands atomically and independently, so a
    # crash leaves valid reusable chunks rather than a half-written container
    # that has to be discarded -- and `already_have` is a stat, not an index
    # probe.
    _t = time.perf_counter()
    encoded = encode_checkpoint(
        checkpoint,
        base_source,
        emit=lambda record: store.put_at(record.content_hash.hex(), record.payload),
        base_manifests=(
            None if base_source is None
            else base_source.tensor_manifest_hashes()
        ),
        already_have=lambda content_hash: store.has(content_hash.hex()),
        alignment=alignment,
        **({} if args.chunk_size is None
           else {"chunk_size_bytes": args.chunk_size}),
    )
    # Encoding covers reading the checkpoint, subtracting, compressing and
    # writing every chunk object -- the emit callback stores as it goes, so
    # there is no separate "write chunks" phase to attribute.
    timings["encode"] = time.perf_counter() - _t
    _t = time.perf_counter()

    topology_config_hash = _resolve_config(store, config_path, base_hash)

    objects = graph.write_checkpoint_objects(
        store,
        header_bytes=encoded.header_bytes,
        manifests=encoded.manifests,
        topology_config_hash=topology_config_hash,
        reused_manifests=encoded.reused_manifests,
    )
    commit_hash = graph.write_commit_object(
        store,
        checkpoint_manifest=objects.checkpoint_manifest,
        parents=[head_commit] if head_commit is not None else [],
        message=args.message,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        full=store_full,
        # Basename only -- see write_commit_object's docstring. `checkout`
        # joins this onto the repo root, so anything path-shaped here would be
        # a traversal primitive.
        checkpoint_name=checkpoint.name,
    )

    repo.update_ref(branch, commit_hash)
    timings["write"] = time.perf_counter() - _t
    timings["total"] = time.perf_counter() - started

    original = encoded.original_bytes
    result = {
        "commit": commit_hash,
        "branch": branch,
        "base": None if store_full else anchor,
        "full": store_full,
        "message": args.message,
        "checkpoint_name": checkpoint.name,
        "tensors": encoded.tensors,
        "original_bytes": original,
        "residual_bytes": encoded.stored_bytes,
        "residual_ratio": (encoded.stored_bytes / original) if original else 0.0,
        "deduped_bytes": encoded.deduped_bytes,
        "chunks_new": encoded.chunks_new,
        "chunks_deduped": encoded.chunks_deduped,
        "manifests_reused": objects.reused_tensor_manifests,
        "tensors_unchanged": len(encoded.reused_manifests),
        "alignment": (
            align_result.as_json() if align_result is not None
            else {"groups": 0, "identity": True}
        ),
        "notes": encoded.notes + notes,
    }
    if getattr(args, "timing", False):
        result["timing_s"] = {k: round(v, 3) for k, v in timings.items()}
    return result


def _align(store, checkpoint: Path, base_source, config_path: Path,
           *, no_align: bool, notes: list):
    """Solve the permutation between `base_source` and `checkpoint`.

    Returns `(alignment, result)` where `alignment` maps tensor name ->
    `TensorPermutation` with its permutation objects already stored, and
    `result` is the solver's report (or None if alignment did not run).

    Degrades rather than fails. A checkpoint this parser cannot describe --
    attention blocks, an unrecognised layer chain -- still commits, just with
    an identity permutation, and says so in `notes`. Refusing to store a
    checkpoint because we could not *optimise* it would be the wrong trade.
    `--strict` is what turns a degraded alignment into an error.
    """
    if base_source is None:
        return None, None

    refs = {}
    for name in base_source.names():
        spec = base_source.spec(name)
        refs[name] = TensorRef(name=name, shape=tuple(spec.shape), dtype=spec.dtype)

    config = None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except ValueError:
            notes.append(f"{config_path.name}: not valid JSON, ignored for alignment")

    try:
        topo = config_parser.parse(refs, config)
        result = solver.align_checkpoints(
            base_source, str(checkpoint), topo, no_align=no_align,
            # The residual pass exists to decide whether a solved permutation
            # actually helped. Under --no-align there is no permutation, so it
            # measures nothing and costs a full fp16->fp32 upcast of both
            # checkpoints plus float64 norms per tensor -- 42s of CPU on a
            # 176 MiB model, for a number nothing reads.
            measure=not no_align,
        )
    except AlignError as exc:
        notes.append(f"alignment skipped: {exc}")
        return None, None

    # Whether a solved permutation is worth applying is decided in the solver,
    # per group, because a permutation IS per group -- see
    # `solver._reject_unhelpful`. By the time we get here every tensor still
    # carrying a permutation belongs to a group that earned it, and every
    # member of that group carries the same one. There is nothing left to
    # second-guess tensor by tensor, and doing so is what broke the group
    # invariant before.
    alignment = {}
    for name, ta in result.tensors.items():
        if ta.identity:
            continue
        alignment[name] = TensorPermutation(
            row=ta.pi_row,
            col=ta.pi_col,
            col_block_size=ta.col_block_size,
            # Identity is stored as null and gets no object (FORMAT.md 4.2).
            row_object=None if ta.pi_row is None else store.put(lap.pack(ta.pi_row)),
            col_object=None if ta.pi_col is None else store.put(lap.pack(ta.pi_col)),
        )
    if result.rejected_groups:
        notes.append(
            f"{len(result.rejected_groups)} permutation group(s) kept at "
            f"identity: the solved permutation did not reduce the residual"
        )
    return (alignment or None), result


def _resolve_config(store, config_path: Path, base_hash: Optional[str]) -> Optional[str]:
    """Hash of the `config.json` this checkpoint was aligned against.

    CLI.md ~3: the config is required for the first commit and "reused from the
    base commit afterward if omitted". Reuse is a matter of copying the base
    checkpoint-manifest's `topology_config_hash` -- the config object itself is
    already stored and content-addressed, so an unchanged config costs nothing
    to carry forward.
    """
    if config_path.is_file():
        return store.put(config_path.read_bytes())
    if base_hash is None:
        return None
    base_commit = graph.get_json(store, base_hash)
    base_manifest = graph.get_json(store, base_commit["checkpoint_manifest"])
    return base_manifest.get("topology_config_hash")




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
        f"{output.human_bytes(result.get('original_bytes', 0))}"
    )
    lines.append(
        f"  residual: {output.human_bytes(result.get('residual_bytes', 0))} "
        f"({result.get('residual_ratio', 0.0) * 100:.2f}% of original)"
    )
    lines.append(
        f"  new chunks: {result.get('chunks_new', 0)}   "
        f"deduped: {result.get('chunks_deduped', 0)}"
    )
    # Phase timings, only under --timing. Alignment scales with unit counts and
    # encoding with bytes, so they move independently and a single total hides
    # which one did: a fine-tune that suddenly takes minutes is the solver
    # failing to converge, not the codec getting slower.
    t = result.get("timing_s") or {}
    if t:
        total = t.get("total") or 0.0
        lines.append("  timing")
        for phase in ("align", "encode", "write"):
            if phase in t:
                share = f"{t[phase] / total * 100:4.0f}%" if total else "    "
                lines.append(f"    {phase:<7}{t[phase]:8.2f}s {share}")
        lines.append(f"    {'total':<7}{total:8.2f}s")

    commit_hash = (result.get("commit") or "??????")[:6]
    lines.append(
        f"[{result.get('branch', '?')} {commit_hash}] {result.get('message', '')}"
    )
    return "\n".join(lines)
