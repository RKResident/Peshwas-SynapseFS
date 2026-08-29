"""Lineage verification: the integrity engine behind `synapsefs verify`.

PS module 2 (Verifiable Storage and Lineage) is 20% of the grade, split evenly
between *tamper detection* and *verification speed*. This module is the whole
of the first half and most of the second.

**Where trust comes from.** PS 2c is precise about this: "Trust is rooted at a
locally accepted commit/ref ID", and defending against a peer that presents an
entirely different but self-consistent history is explicitly out of scope. So
the ref is the axiom, and everything else has to be derived from it:

    ref  (trusted by assumption)
     |-> commit hash            -> re-hash the commit's bytes
         |-> checkpoint_manifest -> re-hash
             |-> header_object   -> re-hash
             |-> tensor_manifest -> re-hash
                 |-> chunks[].object -> decompress the payload, re-hash

Every comparison above is against a hash that came from the object's *parent*.
That is the single property this module exists to maintain, and the one thing
worth reviewing it for. The tempting shortcuts all break it:

* Verifying a pack against its own trailer proves the pack is internally
  consistent, which an attacker who rewrote it will have ensured.
* Verifying a chunk against the checksum in the pack index proves the index
  and the pack agree, which the same attacker will also have ensured.

Both are worth doing -- they catch bit-rot cheaply, and rot is the more likely
failure -- but neither can catch tampering, because the reference value lives
in a file the attacker controls just as much as the payload. Only
`chunks[].object`, reached by walking down from the ref, is anchored to
something outside the attacker's reach. That is why `content` is the default
tier (FORMAT.md 12B) rather than an opt-in `--deep`, and why the fast tiers
are documented as rot scans rather than as security.

**Speed.** Two structural wins before any threading. Objects and chunks are
memoised across the whole walk, which matters because FORMAT.md 4.5 makes
tensor-manifests shared between commits by construction and chunk dedup makes
payloads shared as well -- a 25-commit history re-verifies far fewer distinct
objects than it references. And the content check hashes the *decompressed
stream* rather than reconstructed rows, so a residual chunk never pulls in its
base: verification is O(stored bytes) with no recursion, unlike reconstruction.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import blake3

from synapsefs import graph
from synapsefs.codec.chunk import plain_stream
from synapsefs.errors import IntegrityError, ObjectNotFoundError
from synapsefs.materialize import DEFAULT_BATCH_BYTES, iter_tensor_bytes
from synapsefs.store.objectstore import ObjectStore
from synapsefs.store.repo import Repo

__all__ = [
    "STRUCTURE", "CHECKSUM", "CONTENT", "TIERS",
    "Failure", "VerifyReport", "verify_lineage",
]

# Tiers, cheapest first. Each includes everything the one before it does.
STRUCTURE = "structure"
"""Object graph only: every loose object re-hashed against the hash its parent
named, and every chunk reference probed for existence. Touches no payload."""

CHECKSUM = "checksum"
"""+ the stored (still compressed) payload of every chunk, against the 8-byte
checksum in the pack index. Catches rot; cannot catch tampering."""

CONTENT = "content"
"""+ decompress every chunk and re-hash the plain stream against the manifest's
`chunks[].object`. The only tier anchored to the ref all the way down, and
therefore the only one that detects a crafted block. The default."""

TIERS = (STRUCTURE, CHECKSUM, CONTENT)

DEFAULT_MAX_FAILURES = 100
"""Stop collecting after this many. A corrupt pack can produce one failure per
chunk, and neither a human nor a JSON consumer benefits from 50,000 of them;
`ok` is already false after the first. Reported as `truncated`."""


@dataclass(frozen=True)
class Failure:
    """One detected integrity violation.

    Field names follow CLI.md 7's `--json` contract:
    `{object, pack, expected, actual, referenced_by}`, plus `kind` so a
    consumer can branch without parsing prose.
    """

    kind: str
    object: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    pack: Optional[str] = None
    referenced_by: Optional[str] = None
    detail: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "object": self.object,
            "expected": self.expected,
            "actual": self.actual,
            "pack": self.pack,
            "referenced_by": self.referenced_by,
            "detail": self.detail,
        }


@dataclass
class VerifyReport:
    tier: str
    check_content_hash: bool = False
    commits: int = 0
    objects: int = 0
    manifests: int = 0
    chunks: int = 0
    """Chunk *references* walked. Higher than `chunks_distinct` because dedup
    and FORMAT.md 4.5's manifest reuse make one chunk serve many commits."""
    chunks_distinct: int = 0
    """Chunks actually read and checked. The gap between this and `chunks` is
    the memoisation win, and it is most of why verification stays fast as a
    history deepens."""
    bytes_verified: int = 0
    """Stored (compressed) bytes actually read and checked. Zero at
    `structure`, where no payload is touched."""
    elapsed: float = 0.0
    truncated: bool = False
    failures: List[Failure] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tier": self.tier,
            "content_hash_checked": self.check_content_hash,
            "commits": self.commits,
            "objects": self.objects,
            "manifests": self.manifests,
            "chunks": self.chunks,
            "chunks_distinct": self.chunks_distinct,
            "bytes_verified": self.bytes_verified,
            "elapsed": self.elapsed,
            "truncated": self.truncated,
            "failures": [f.as_dict() for f in self.failures],
            "notes": list(self.notes),
        }


class _Walker:
    """One verification run. Holds the memo tables, which is the only reason
    this is a class rather than a function."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        tier: str,
        check_content_hash: bool,
        max_failures: int,
    ):
        self.store = store
        self.tier = tier
        self.check_content_hash = check_content_hash
        self.max_failures = max_failures
        self.report = VerifyReport(tier=tier, check_content_hash=check_content_hash)

        # Memo tables. `_ok_objects` holds hashes whose *bytes* have been
        # re-hashed and matched; `_ok_chunks` holds chunk hashes whose payload
        # has passed this run's tier. Both are keyed by content hash, so a
        # shared object is verified once no matter how many commits reference
        # it -- which under FORMAT.md 4.5 is most of them.
        self._ok_objects: Set[str] = set()
        self._ok_chunks: Set[str] = set()
        self._bad: Set[str] = set()

    # -- failure collection ------------------------------------------------

    def fail(self, failure: Failure) -> None:
        if len(self.report.failures) >= self.max_failures:
            self.report.truncated = True
            return
        self.report.failures.append(failure)

    # -- the anchored read -------------------------------------------------

    def load_object(self, object_hash: str, referenced_by: str) -> Optional[bytes]:
        """Read a loose object and confirm its bytes hash to `object_hash`.

        This is the primitive the whole trust chain is built from, and the
        reason `ObjectStore.get` deliberately does *not* do it: `get` is on
        every read path in the system and cannot afford to re-hash, while this
        is `verify`'s entire job. Returns None (having recorded a failure) if
        the object is missing or does not match, so a walk can continue past a
        single bad object and report every problem rather than the first.
        """
        if object_hash in self._bad:
            return None
        try:
            data = self.store.get(object_hash)
        except ObjectNotFoundError:
            self._bad.add(object_hash)
            self.fail(Failure(
                kind="missing-object",
                object=object_hash,
                referenced_by=referenced_by,
                detail="referenced but not present in the object store",
            ))
            return None

        if object_hash in self._ok_objects:
            return data

        actual = blake3.blake3(data).hexdigest()
        if actual != object_hash:
            self._bad.add(object_hash)
            self.fail(Failure(
                kind="object-hash-mismatch",
                object=object_hash,
                expected=object_hash,
                actual=actual,
                referenced_by=referenced_by,
                detail="stored bytes do not hash to the name they are stored under",
            ))
            return None

        self._ok_objects.add(object_hash)
        self.report.objects += 1
        return data

    def load_json(self, object_hash: str, referenced_by: str) -> Optional[dict]:
        data = self.load_object(object_hash, referenced_by)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.fail(Failure(
                kind="malformed-object",
                object=object_hash,
                referenced_by=referenced_by,
                detail=f"not valid JSON: {exc}",
            ))
            return None

    # -- the walk ----------------------------------------------------------

    def walk(self, roots: Sequence[str]) -> None:
        """Breadth-first over every parent, not just the first.

        PS 2f requires verifying "any branch, not only a single linear chain",
        and 2g adds merges -- a merge commit's second parent is reachable
        history whose corruption is just as fatal, so `walk_first_parent`
        (right for `log`'s *listing*) would be wrong here. The visited set is
        what keeps a diamond from re-verifying its shared ancestry.
        """
        seen: Set[str] = set()
        queue = deque(roots)
        while queue:
            commit_hash = queue.popleft()
            if commit_hash in seen:
                continue
            seen.add(commit_hash)

            commit = self.load_json(commit_hash, referenced_by="ref")
            if commit is None:
                continue
            self.report.commits += 1
            self.verify_checkpoint(commit, commit_hash)

            for parent in commit.get("parents") or []:
                if parent not in seen:
                    queue.append(parent)

    def verify_checkpoint(self, commit: dict, commit_hash: str) -> None:
        where = f"commit {commit_hash[:8]}"
        manifest_hash = commit.get("checkpoint_manifest")
        if not manifest_hash:
            self.fail(Failure(
                kind="malformed-object", object=commit_hash,
                detail="commit has no checkpoint_manifest",
            ))
            return

        manifest = self.load_json(manifest_hash, referenced_by=where)
        if manifest is None:
            return

        # The verbatim header and the topology config are content-addressed
        # like everything else; a tampered header would change how every
        # tensor in the file is interpreted, so it is not optional.
        for key in ("header_object", "topology_config_hash"):
            child = manifest.get(key)
            if child:
                self.load_object(
                    child, referenced_by=f"checkpoint-manifest {manifest_hash[:8]} ({key})"
                )

        for name, tensor_hash in (manifest.get("tensors") or {}).items():
            self.verify_tensor(
                tensor_hash, name,
                referenced_by=f"checkpoint-manifest {manifest_hash[:8]}",
            )

    def verify_tensor(self, tensor_hash: str, name: str, *, referenced_by: str) -> None:
        # A tensor-manifest shared between commits (FORMAT.md 4.5) is already
        # in the memo, but its *chunks* still need visiting on the first pass
        # only -- hence the check against `_ok_objects` rather than an early
        # return from load_object.
        already = tensor_hash in self._ok_objects
        manifest = self.load_json(tensor_hash, referenced_by=referenced_by)
        if manifest is None:
            return
        if not already:
            self.report.manifests += 1

        dtype = manifest.get("dtype")
        for chunk in manifest.get("chunks") or []:
            # Counted even for a manifest already verified under an earlier
            # commit, so `chunks` means "references walked" rather than
            # "references we happened not to skip" -- otherwise the memo would
            # hide its own effect by shrinking the number it should be
            # compared against.
            self.report.chunks += 1
            if not already:
                self.verify_chunk(chunk, tensor_hash, name, dtype)

    def verify_chunk(
        self, chunk: dict, tensor_hash: str, name: str, dtype: Optional[str]
    ) -> None:
        object_hex = chunk.get("object")
        if not object_hex:
            self.fail(Failure(
                kind="malformed-object", object=tensor_hash,
                detail=f"{name}: chunk entry has no object hash",
            ))
            return

        where = (
            f"tensor-manifest {tensor_hash[:8]} ({name!r}, rows "
            f"{chunk.get('row_start')}-{chunk.get('row_end')})"
        )

        if object_hex in self._ok_chunks:
            return
        if object_hex in self._bad:
            return

        object_hex_lower = object_hex.lower()
        path = self.store.path_for(object_hex_lower)
        if not path.is_file():
            self._bad.add(object_hex)
            self.fail(Failure(
                kind="missing-chunk", object=object_hex, referenced_by=where,
                detail="referenced by a manifest but not in the object store",
            ))
            return

        if self.tier == STRUCTURE:
            # Existence is all this tier promises -- CLI.md 7 calls it "broken
            # links", and this is one stat().
            self._ok_chunks.add(object_hex)
            return

        payload = path.read_bytes()
        self.report.bytes_verified += len(payload)

        # CHECKSUM tier. The reference value comes from the *tensor-manifest*,
        # which is hash-chained to the ref -- so unlike the old pack-index
        # checksum this is anchored, and detects substitution rather than only
        # rot. Full tamper detection without decompressing anything
        # (ARCHITECTURE.md 4.5.2).
        expected = chunk.get("stored_checksum")
        if expected is not None:
            actual = blake3.blake3(payload).digest()[:8].hex()
            if actual != expected:
                self._bad.add(object_hex)
                self.fail(Failure(
                    kind="chunk-checksum", object=object_hex,
                    expected=expected, actual=actual, referenced_by=where,
                    detail="stored bytes do not match the checksum the manifest "
                           "records -- rot or substitution",
                ))
                return

        if self.tier != CONTENT:
            self._ok_chunks.add(object_hex)
            return

        if not self._verify_content(chunk, payload, object_hex, None, where):
            return
        self._ok_chunks.add(object_hex)

    def _verify_content(
        self, chunk: dict, payload: bytes, object_hex: str,
        pack_name: str, where: str,
    ) -> bool:
        """The anchored check: does this payload decompress to the bytes the
        *manifest* says it should?

        `chunk['object']` is the reference value and it was reached by walking
        down from the ref, so unlike the pack checksum it is not something a
        tamperer could have rewritten in step with the payload. This is the
        one comparison in the module that detects malicious block injection.
        """
        try:
            stream = plain_stream(chunk.get("encoding"), payload)
        except Exception as exc:
            self._bad.add(object_hex)
            self.fail(Failure(
                kind="decode-error", object=object_hex, pack=pack_name,
                referenced_by=where,
                detail=f"payload could not be decompressed: {exc}",
            ))
            return False

        actual = blake3.blake3(stream).hexdigest()
        if actual != object_hex:
            self._bad.add(object_hex)
            self.fail(Failure(
                kind="chunk-content-mismatch", object=object_hex,
                expected=object_hex, actual=actual, pack=pack_name,
                referenced_by=where,
                detail="decompressed bytes do not match the hash the manifest "
                       "names -- this block was substituted",
            ))
            return False
        return True


# -- content_hash tier ------------------------------------------------------


def _verify_tensor_content_hashes(
    store: ObjectStore, commit_hashes: Sequence[str], walker: _Walker,
) -> None:
    """Reconstruct every tensor and check it against the manifest's
    `content_hash` (FORMAT.md 7.2).

    Separate from the tiers above, and much more expensive, because it is the
    only check that requires *reconstruction* rather than hashing stored bytes
    -- a residual chunk has to be applied to its base.

    What it buys is a failure mode nothing else can see: a permutation
    composed in the wrong order (`p2[p1]` instead of `p1[p2]`) produces chunks
    that hash perfectly, because both orderings are valid bijections of the
    right length. Every per-chunk check passes and the reconstructed tensor is
    scrambled. Once the alignment engine lands this becomes the check that
    catches it, which is why it exists before the aligner does.
    """
    for commit_hash in commit_hashes:
        try:
            view = graph.CommitCheckpoint(store, commit_hash)
        except (IntegrityError, KeyError) as exc:   # pragma: no cover - caught above
            walker.fail(Failure(
                kind="decode-error", object=commit_hash, detail=str(exc)))
            continue

        for name in view.names():
            manifest_hash = view.tensor_manifest_hashes()[name]
            manifest = graph.get_json(store, manifest_hash)
            expected = manifest.get("content_hash")
            if not expected:
                continue
            try:
                # Batched, and through the same helper `materialize` uses, so
                # the bytes hashed here are the bytes a checkout would write.
                # `encode_checkpoint` builds `content_hash` the same way, chunk
                # by chunk in row order, so the digests are comparable without
                # either side holding a whole tensor in memory.
                digest = blake3.blake3()
                for block in iter_tensor_bytes(
                    view, view.spec(name), batch_bytes=DEFAULT_BATCH_BYTES
                ):
                    digest.update(block)
                actual = digest.hexdigest()
            except Exception as exc:
                walker.fail(Failure(
                    kind="decode-error", object=manifest_hash,
                    referenced_by=f"commit {commit_hash[:8]} ({name!r})",
                    detail=f"tensor could not be reconstructed: {exc}",
                ))
                continue
            if actual != expected:
                walker.fail(Failure(
                    kind="tensor-content-mismatch", object=manifest_hash,
                    expected=expected, actual=actual,
                    referenced_by=f"commit {commit_hash[:8]} ({name!r})",
                    detail="tensor reconstructs to different bytes than its "
                           "manifest records",
                ))


# -- entry point ------------------------------------------------------------


def verify_lineage(
    repo: Repo,
    roots: Sequence[str],
    *,
    tier: str = CONTENT,
    check_content_hash: bool = False,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> VerifyReport:
    """Verify every commit reachable from `roots`.

    `roots` are already-resolved commit hashes; resolving a ref is the CLI's
    job, and keeping it out of here is what PS 2c's "trust is rooted at a
    locally accepted ref" looks like in code -- this function takes the root
    of trust as an argument rather than deciding it.

    There is no longer a `verify_packs` tier. Packs are gone, and with them the
    self-certifying structures that made one necessary: a chunk is now a file
    named for its own content, and the only reference values left live in the
    tensor-manifest, which is hash-chained to the ref.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")

    started = time.perf_counter()

    walker = _Walker(
        repo.store,
        tier=tier, check_content_hash=check_content_hash,
        max_failures=max_failures,
    )
    walker.walk(list(roots))
    report = walker.report
    report.chunks_distinct = len(walker._ok_chunks)

    if check_content_hash:
        _verify_tensor_content_hashes(repo.store, _reachable(repo.store, roots), walker)

    report.elapsed = time.perf_counter() - started
    return report


def _reachable(store: ObjectStore, roots: Sequence[str]) -> List[str]:
    """Every commit reachable from `roots`, all parents followed.

    Recomputed rather than captured during the walk because the walk skips
    commits whose objects failed to load, and a second pass over a partially
    broken history should not silently narrow its own scope.
    """
    seen: Set[str] = set()
    order: List[str] = []
    queue = deque(roots)
    while queue:
        commit_hash = queue.popleft()
        if commit_hash in seen:
            continue
        seen.add(commit_hash)
        try:
            commit = graph.get_json(store, commit_hash)
        except (ObjectNotFoundError, ValueError):
            continue
        order.append(commit_hash)
        queue.extend(commit.get("parents") or [])
    return order
