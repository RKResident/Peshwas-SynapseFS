"""Three-way merge of two branches, resolving conflicts by averaging.

**Merge here is more than git's.** Git reconciles *history*: if only one side
touched a file, take that side; if both did, hand the conflict to a human.
That works because a text conflict has a correct resolution a person can see.
Two tensors that both changed have no such resolution -- nobody can read a
4 MiB float array and pick.

So conflicts are resolved by **averaging after alignment**, and that choice
needs stating plainly:

* Averaging two independently-initialised networks elementwise produces a
  broken model. They occupy different basins of the loss landscape; the same
  functional unit sits at a different index in each, so the mean of two good
  models is near-chance. This is the central result of the Git Re-Basin line
  of work, and the reason `align/` exists.
* After permuting one side into the other's basis, `(A + B) / 2` *tends* to
  retain performance -- "linear mode connectivity modulo permutation". Tends:
  wide networks do well, narrow ones keep a real loss barrier.
* **This module cannot verify any of that.** Checking that a merge preserved
  accuracy means running the network on data, which a storage system does not
  have. What is guaranteed is that the merge is deterministic, content-
  addressed, and reconstructs byte-identically. Whether it is a *useful* model
  is a claim only evaluation can make, and the CLI says so.

Classification uses each tensor-manifest's `content_hash`, which is why that
field exists (FORMAT.md 2.1). A manifest hash would be wrong here: two
branches can hold byte-identical weights whose manifests differ because they
were diffed against different bases, and comparing manifests would report a
conflict on a tensor nobody touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from synapsefs import graph
from synapsefs.codec.checkpoint import apply_col_perm
from synapsefs.errors import ConflictError
from synapsefs.safetensors_io import TensorSpec
from synapsefs.store.objectstore import ObjectStore

__all__ = ["TensorDecision", "MergePlan", "plan_merge", "MergedCheckpoint"]

#: How a tensor was resolved.
TAKE_OURS = "ours"
TAKE_THEIRS = "theirs"
UNCHANGED = "unchanged"
AVERAGED = "averaged"
ONLY_OURS = "only-ours"
ONLY_THEIRS = "only-theirs"


@dataclass(frozen=True)
class TensorDecision:
    name: str
    action: str
    detail: str = ""


@dataclass
class MergePlan:
    base: Optional[str]
    ours: str
    theirs: str
    decisions: Dict[str, TensorDecision] = field(default_factory=dict)

    @property
    def conflicts(self) -> List[str]:
        return [n for n, d in self.decisions.items() if d.action == AVERAGED]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self.decisions.values():
            out[d.action] = out.get(d.action, 0) + 1
        return out


def _content_hashes(store: ObjectStore, commit_hash: str) -> Dict[str, str]:
    commit = graph.get_json(store, commit_hash)
    manifest = graph.get_json(store, commit["checkpoint_manifest"])
    out = {}
    for name, tensor_hash in manifest["tensors"].items():
        out[name] = graph.get_json(store, tensor_hash).get("content_hash")
    return out


def plan_merge(store: ObjectStore, ours: str, theirs: str,
               base: Optional[str]) -> MergePlan:
    """Classify every tensor. Pure: reads hashes, touches no weights."""
    plan = MergePlan(base=base, ours=ours, theirs=theirs)
    h_ours = _content_hashes(store, ours)
    h_theirs = _content_hashes(store, theirs)
    h_base = _content_hashes(store, base) if base else {}

    for name in sorted(set(h_ours) | set(h_theirs)):
        o, t = h_ours.get(name), h_theirs.get(name)
        b = h_base.get(name)
        if o is None:
            plan.decisions[name] = TensorDecision(name, ONLY_THEIRS,
                                                  "absent from ours")
        elif t is None:
            plan.decisions[name] = TensorDecision(name, ONLY_OURS,
                                                  "absent from theirs")
        elif o == t:
            plan.decisions[name] = TensorDecision(name, UNCHANGED,
                                                  "both sides identical")
        elif b is not None and o == b:
            plan.decisions[name] = TensorDecision(name, TAKE_THEIRS,
                                                  "only they changed it")
        elif b is not None and t == b:
            plan.decisions[name] = TensorDecision(name, TAKE_OURS,
                                                  "only we changed it")
        else:
            plan.decisions[name] = TensorDecision(
                name, AVERAGED, "both sides changed it")
    return plan


def _to_float(bits: np.ndarray, dtype: str) -> np.ndarray:
    flat = np.ascontiguousarray(bits).reshape(-1)
    if dtype == "BF16":
        return (flat.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype == "F16":
        return flat.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return flat.view(np.float32)
    raise ConflictError(f"cannot average dtype {dtype!r}")


def _from_float(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        # Round to nearest rather than truncate: dropping 16 bits by shifting
        # alone biases every averaged weight downward.
        u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        return (((u + np.uint32(0x8000)) >> np.uint32(16))).astype(np.uint16)
    if dtype == "F16":
        return values.astype(np.float16).view(np.uint16)
    if dtype == "F32":
        return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    raise ConflictError(f"cannot average dtype {dtype!r}")


class MergedCheckpoint:
    """The merge result, readable exactly like a checkpoint.

    Implements `names()`, `spec()` and `rows()` -- the same three methods
    `SafetensorsFile` and `CommitCheckpoint` expose -- so `materialize` writes
    it out with no new writer, and the result is byte-exact by the same code
    path everything else uses.

    `alignment` maps tensor name -> `TensorPermutation`, solved by the caller
    for THEIRS against OURS. Averaging without it is the failure the module
    docstring describes.
    """

    def __init__(self, store: ObjectStore, plan: MergePlan,
                 alignment: Optional[dict] = None):
        self.store = store
        self.plan = plan
        self.alignment = alignment or {}
        self.ours = graph.CommitCheckpoint(store, plan.ours)
        self.theirs = graph.CommitCheckpoint(store, plan.theirs)

    @property
    def header_bytes(self) -> bytes:
        """Ours, verbatim. Both sides share an architecture, so the shapes and
        dtypes agree; taking one side's header keeps its `__metadata__` and
        padding rather than synthesising something neither side had."""
        return self.ours.header_bytes

    def names(self) -> List[str]:
        return self.ours.names()

    def spec(self, name: str) -> TensorSpec:
        d = self.plan.decisions.get(name)
        source = self.theirs if d is not None and d.action == ONLY_THEIRS else self.ours
        return source.spec(name)

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        d = self.plan.decisions[name]
        if d.action in (UNCHANGED, TAKE_OURS, ONLY_OURS):
            return self.ours.rows(name, start, stop)
        if d.action in (TAKE_THEIRS, ONLY_THEIRS):
            return self._their_rows(name, start, stop)

        spec = self.ours.spec(name)
        ours = self.ours.rows(name, start, stop)
        theirs = self._their_rows(name, start, stop)
        if spec.dtype not in ("F16", "BF16", "F32"):
            # Integer buffers -- BatchNorm's num_batches_tracked is the only
            # one in practice. The mean of two counters is not a counter, so
            # keep ours rather than inventing a value.
            return ours
        mean = (_to_float(ours, spec.dtype) + _to_float(theirs, spec.dtype)) / 2.0
        return _from_float(mean, spec.dtype).reshape(ours.shape)

    def _their_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        """Their rows, brought into our basis.

        `perm.row[i]` is THEIR index corresponding to OUR index `i`, so the
        gather is `theirs[perm.row]` -- the same direction the codec uses, and
        never the inverse (ARCHITECTURE.md 2.4).
        """
        perm = self.alignment.get(name)
        if perm is None or perm.row is None:
            block = self.theirs.rows(name, start, stop)
        else:
            block = self.theirs.gather_rows(name, perm.row[start:stop])
        if perm is not None and perm.col is not None:
            block = apply_col_perm(block, perm.col, perm.col_block_size)
        return block
