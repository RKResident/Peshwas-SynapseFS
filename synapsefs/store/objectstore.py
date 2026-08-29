"""Content-addressed object storage.

`ObjectStore` is intentionally the *only* thing in SynapseFS that knows how
to turn bytes into a hash and a path. It has no concept of "commit",
"branch", or even "object kind" (chunk vs. header vs. manifest, per
FileFormat.md ~3-4) -- every object kind is just bytes to this layer.
That's deliberate: it means crash-safety and content-addressing correctness
only ever need to be proven once, here, instead of once per object kind.

Packed storage (FileFormat.md ~5-6) is a later, separate optimization for
chunk objects specifically and is out of scope for this module -- everything
here is a "loose object": one file per hash, sharded by the first two hex
characters of the hash so that one directory never ends up holding hundreds
of thousands of entries.
"""

from __future__ import annotations

from pathlib import Path

import blake3

from synapsefs.errors import ObjectNotFoundError
from synapsefs.store.atomic import atomic_write, gc_tmp_dir


class ObjectStore:
    """Content-addressed put/get/has over `<repo>/.synapse/objects/`.

    Layout (FileFormat.md ~3)::

        objects/
        |-- tmp/<random>       write staging; GC'd unconditionally on open
        `-- <hh>/<hash>        loose objects, sharded by first hash byte
    """

    def __init__(self, objects_dir: Path):
        self.objects_dir = Path(objects_dir)
        self.tmp_dir = self.objects_dir / "tmp"
        # Anything left here means a previous write never completed --
        # atomic_write only ever leaves a file in tmp/ mid-write, renaming
        # it out on success. Safe to delete unconditionally; see
        # atomic.gc_tmp_dir's docstring for the full argument.
        gc_tmp_dir(self.tmp_dir)

    def path_for(self, object_hash: str) -> Path:
        """Sharded on-disk path for a hex hash: `objects/<ab>/<cd>/<60 hex>`.

        **Two shard levels, and the same scheme for every object kind** --
        commits, manifests, headers, configs and chunk payloads all live here
        (ARCHITECTURE.md 3.3). An earlier draft gave chunks two levels and
        everything else one, on the assumption that chunks vastly outnumber
        objects; measured, they do not (1,002 objects against 810 chunks over
        25 commits), and the split bought nothing but an "is this a chunk?"
        ambiguity at every call site.

        The filename is the hash **minus** the shard prefix, matching the
        loose-object convention in `docs/kris_docs.md`. The single source of
        truth for the layout -- `put`, `get` and `has` all route through here.
        """
        return self.objects_dir / object_hash[:2] / object_hash[2:4] / object_hash[4:]

    def has(self, object_hash: str) -> bool:
        """Whether an object with this hash is already stored.

        A path-existence check only -- deliberately does not read or
        re-hash the file. That's `verify`'s job (the tiered checks in
        FileFormat.md), not this layer's; `has()` needs to stay cheap
        because `put()` calls it on every single object to get
        content-addressed dedup for free.
        """
        return self.path_for(object_hash).exists()

    def put(self, data: bytes) -> str:
        """Store `data`, returning its BLAKE3 hex hash.

        If an object with this hash already exists, this is a stat, not a
        write -- this check *is* the dedup mechanism. Every object write in
        the system (chunks, manifests, commits, ...) is expected to funnel
        through this one method, so identical content anywhere in the
        system is only ever written to disk once.
        """
        object_hash = blake3.blake3(data).hexdigest()
        target = self.path_for(object_hash)
        if target.exists():
            return object_hash
        atomic_write(target, data, tmp_dir=self.tmp_dir)
        return object_hash

    def put_at(self, object_hash: str, data: bytes) -> str:
        """Store `data` under a hash the caller already computed.

        For **chunk payloads**, whose identity is `blake3` of the *decompressed
        stream* rather than of the bytes being written (FORMAT.md 2.1). Hashing
        `data` here would compute the wrong name -- and would also undo the
        property that a chunk's identity survives recompression at a different
        level.

        The caller owns the correctness of `object_hash`; `verify --deep` is
        what re-establishes it, by decompressing and re-hashing against the
        name the manifest gave.
        """
        target = self.path_for(object_hash)
        if target.exists():
            return object_hash
        atomic_write(target, data, tmp_dir=self.tmp_dir)
        return object_hash

    def get(self, object_hash: str) -> bytes:
        """Read back the bytes stored under `object_hash`.

        Raises ObjectNotFoundError (not a bare FileNotFoundError) so
        callers up the stack can catch one exception type regardless of
        whether the miss came from this store, a pack index, or anywhere
        else objects get looked up later.
        """
        path = self.path_for(object_hash)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(object_hash) from exc

