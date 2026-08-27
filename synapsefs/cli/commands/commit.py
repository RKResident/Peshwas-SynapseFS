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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from synapsefs import graph
from synapsefs.cli import output
from synapsefs.codec.checkpoint import encode_checkpoint
from synapsefs.errors import UsageError
from synapsefs.pack.index import write_index
from synapsefs.pack.pack import PackWriter
from synapsefs.pack.packset import PackSet
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
    pack_dir = repo.objects_dir / "pack"
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

    with PackSet(pack_dir, tmp_dir=tmp_dir) as packs:
        base_source = (
            None if store_full
            else graph.CommitCheckpoint(store, packs, anchor)
        )

        with PackWriter(pack_dir, tmp_dir=tmp_dir) as writer:
            encoded = encode_checkpoint(
                checkpoint,
                base_source,
                emit=writer.add_record,
                # Without these the manifests would record residual chunks
                # with no base to decode them against.
                base_manifests=(
                    None if base_source is None
                    else base_source.tensor_manifest_hashes()
                ),
                # Cross-commit dedup: a chunk already in any pack is counted,
                # referenced by the manifest, and not written again.
                already_have=packs.has,
                **({} if args.chunk_size is None
                   else {"chunk_size_bytes": args.chunk_size}),
            )

        write_index(
            pack_dir / f"{writer.pack_hash.hex()}.idx",
            pack_hash=writer.pack_hash,
            entries=writer.entries,
            tmp_dir=tmp_dir,
        )
        packs.register(writer.pack_hash)

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

    original = encoded.original_bytes
    return {
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
        # Alignment is not implemented yet: every commit currently diffs
        # against the base's own row order, i.e. an identity permutation.
        # `--no-align` and `--strict` are accepted and parsed but cannot
        # change that until the alignment engine lands.
        "alignment": {"groups": 0, "identity": True},
        "notes": encoded.notes,
    }


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
    commit_hash = (result.get("commit") or "??????")[:6]
    lines.append(
        f"[{result.get('branch', '?')} {commit_hash}] {result.get('message', '')}"
    )
    return "\n".join(lines)
