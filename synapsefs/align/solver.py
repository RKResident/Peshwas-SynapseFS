"""The front door: two checkpoints in, one TensorAlignment per tensor out.

Residual measurement and the not-alignable threshold live in residual.py; this
module only decides WHEN to ask.

Alignment is solved per GROUP -- one shared ordering -- but the codec works per
TENSOR, so this module fans the answer out. One group's permutation lands on
several tensors and on different axes: a hidden group is the ROW permutation of
its own weight and bias, and the COLUMN permutation of the next layer's weight.
Getting that fan-out right is most of what this file does.

Nothing here mutates a checkpoint. The permutations travel as arrays; the codec
applies them once, during subtraction.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .coordinate_descent import DEFAULT_MAX_SWEEPS, DescentResult, descend
from .IR import GroupId, Topology
from .objective import MatrixPair, apply_col_perm, apply_row_perm
from .reader import SafetensorsReader
from .residual import (NOT_ALIGNABLE_THRESHOLD, Assessment, assess,
                       group_helped, summarize)


@dataclass
class TensorAlignment:
    pi_row: "np.ndarray | None" = None
    pi_col: "np.ndarray | None" = None
    col_block_size: int = 1
    alignable: bool = True
    residual_pre: float = float("nan")
    residual_post: float = float("nan")

    @property
    def identity(self) -> bool:
        return self.pi_row is None and self.pi_col is None


@dataclass
class AlignmentResult:
    tensors: dict[str, TensorAlignment] = field(default_factory=dict)
    not_alignable: list[str] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    missing_from_base: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)
    identity: bool = True
    groups: int = 0
    group_sizes: dict[GroupId, int] = field(default_factory=dict)
    sweeps: int = 0
    converged: bool = True
    unsolved_groups: list[GroupId] = field(default_factory=list)
    #: Groups the solver permuted but whose permutation did not reduce the
    #: residual, so it was dropped and every member left at identity.
    #: Distinct from `unsolved_groups`: those could not be solved at all.
    rejected_groups: list[GroupId] = field(default_factory=list)
    #: True when --no-align skipped the solve entirely. Identity is then
    #: ASSUMED, not found, and the two must never read the same in the
    #: output -- a fast commit whose log says 'identity detected' is
    #: otherwise indistinguishable from one that was never solved at all.
    disabled: bool = False
    wall_clock_s: float = 0.0
    seed: int = 0
    assessments: dict[str, Assessment] = field(default_factory=dict)

    def residual_summary(self):
        return summarize(self.assessments)

    def __getitem__(self, name: str) -> TensorAlignment:
        return self.tensors[name]

    @property
    def solved(self) -> bool:
        """False when a group was left at identity because it could not be
        solved, rather than because identity was the answer. The two look the
        same in the output and must never be conflated -- one is the fine-tune
        fast path, the other is alignment silently doing nothing."""
        return not self.unsolved_groups and self.converged

    def as_json(self) -> dict:
        """The 'alignment' block of CLI.md 3.1, plus degradation reporting."""
        return {
            "groups": self.groups,
            "identity": self.identity,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "not_alignable": list(self.not_alignable),
            "unsolved_groups": list(self.unsolved_groups),
            "rejected_groups": list(self.rejected_groups),
            "disabled": self.disabled,
            "converged": self.converged,
            "sweeps": self.sweeps,
        }

    def summary(self) -> str:
        if self.disabled:
            return (f"{self.groups} permutation groups, alignment disabled "
                    f"(--no-align) in {self.wall_clock_s:.2f}s")
        if not self.solved:
            head = (f"{self.groups} permutation groups, "
                    f"{len(self.unsolved_groups)} could NOT be solved "
                    f"(left at identity)")
        elif self.identity:
            head = f"{self.groups} permutation groups, identity permutation detected"
        else:
            moved = sum(1 for a in self.tensors.values() if not a.identity)
            head = (f"{self.groups} permutation groups, {moved} of "
                    f"{len(self.tensors)} tensors permuted")
        if self.solved and not self.converged:
            head += f" (stopped at the {self.sweeps}-sweep cap)"
        return f"{head} in {self.wall_clock_s:.2f}s"

PathLike = (str, os.PathLike)


def _open(x):
    return SafetensorsReader(os.fspath(x)) if isinstance(x, PathLike) else x

def _shapes(x) -> dict[str, tuple[int, ...]]:
    if hasattr(x, "refs"):
        return {n: r.shape for n, r in x.refs().items()}
    return {n: np.asarray(v).shape for n, v in x.items()}


def _unusable(topo: Topology, base_shapes, target_shapes) -> set[GroupId]:
    """Groups with a member the base cannot supply. Solving them would crash."""
    bad = {n for n, s in target_shapes.items()
           if base_shapes.get(n) != s}
    out: set[GroupId] = set()
    for gid, g in topo.groups.items():
        members = set(g.row_members) | {c.name for c in g.col_members}
        if members & (bad | (set(topo.tensors) - set(target_shapes))):
            out.add(gid)
    return out


def _aligned_base(base: np.ndarray, a: TensorAlignment) -> np.ndarray:
    out = apply_row_perm(base, a.pi_row)
    if a.pi_col is not None and out.ndim > 1:
        out = apply_col_perm(out, a.pi_col, a.col_block_size)
    return out


def _members(topo: Topology, gids) -> set[str]:
    """Every tensor these groups touch, on either axis."""
    out: set[str] = set()
    for gid in gids:
        g = topo.groups[gid]
        out.update(g.row_members)
        out.update(c.name for c in g.col_members)
    return out


def _measure(res: "AlignmentResult", src, names, threshold: float) -> None:
    """Score each tensor's residual against its currently planned permutation."""
    for name in names:
        a = res.tensors[name]
        t, b = src.target(name), src.base(name)
        verdict = assess(t, b, _aligned_base(b, a), threshold)
        res.assessments[name] = verdict
        a.residual_pre = verdict.pre
        a.residual_post = verdict.post
        a.alignable = verdict.alignable


def _reject_unhelpful(topo: Topology, perms: dict[GroupId, "np.ndarray | None"],
                      assessments) -> list[GroupId]:
    """Drop, in place, each group's permutation that did not pay for itself.

    The solver maximises the weight-matching objective, which is not the same
    thing as minimising the residual we store. Early in a training run the
    weights move enough that some other matching scores higher on the
    objective while making the delta LARGER -- measured at 83.24% against
    76.44% for identity on epoch 1 -> 2 of the 92M benchmark.

    The test is per GROUP because a permutation is per group. Testing each
    tensor separately and keeping the winners looks like a refinement and is
    not: it applies one axis's ordering to a BatchNorm buffer while leaving
    the kernel that produces those very channels unpermuted. See
    `residual.group_helped`.

    ATTRIBUTION IS JOINT. A tensor sitting between two permuted groups has one
    residual reflecting both, so a group's score includes its neighbours'
    contributions. Rejecting one group therefore leaves its neighbours' scores
    slightly stale, and the caller re-measures rather than trusting them. In
    the two cases that occur in practice -- a fine-tune where every group is
    identity, and a genuine re-basin where every group helps together -- the
    coupling does not change the verdict.
    """
    rejected = []
    for gid, p in perms.items():
        if p is None:
            continue
        scored = [assessments[n] for n in _members(topo, [gid])
                  if n in assessments]
        if not group_helped(scored):
            perms[gid] = None
            rejected.append(gid)
    return rejected


def plan(topo: Topology, perms: dict[GroupId, "np.ndarray | None"],
         names) -> dict[str, TensorAlignment]:
    """Fan a per-group answer out to per-tensor row/column assignments."""
    out: dict[str, TensorAlignment] = {}
    for name in names:
        a = TensorAlignment()
        row = topo.group_for_tensor(name, "row")
        if row is not None:
            a.pi_row = perms.get(row.id)
        col = topo.group_for_tensor(name, "col")
        if col is not None:
            a.pi_col = perms.get(col.id)
            a.col_block_size = topo.col_block_size(col.id, name) or 1
        out[name] = a
    return out


def align_checkpoints(base, target, topo: Topology, *,
                      seed: int = 0,
                      no_align: bool = False,
                      max_sweeps: int = DEFAULT_MAX_SWEEPS,
                      threshold: float = NOT_ALIGNABLE_THRESHOLD,
                      measure: bool = True,
                      on_sweep=None) -> AlignmentResult:
    """Solve, fan out, and measure. base/target are paths, readers, or dicts."""
    started = time.perf_counter()
    opened = [x for x in (base, target) if isinstance(x, PathLike)]
    rb, rt = _open(base), _open(target)
    try:
        src = MatrixPair(rb, rt)
        res = AlignmentResult(seed=seed)
        res.group_sizes = {g.id: g.size for g in topo.groups.values()}
        base_shapes, target_shapes = _shapes(rb), _shapes(rt)
        unusable = _unusable(topo, base_shapes, target_shapes)

        res.disabled = no_align
        if no_align:
            descent = DescentResult(perms={g.id: None
                                           for g in topo.solvable_groups()},
                                    seed=seed)
        else:
            descent = descend(topo, src, max_sweeps=max_sweeps, seed=seed,
                              skip=unusable, on_sweep=on_sweep)
        res.groups = len(topo.groups)
        res.sweeps = descent.sweeps
        res.converged = descent.converged
        res.unsolved_groups = list(descent.unsolved)

        names = list(target_shapes)
        res.tensors = plan(topo, descent.perms, names)
        res.unassigned = sorted(
            n for n in names
            if topo.group_for_tensor(n, "row") is None
            and topo.group_for_tensor(n, "col") is None)

        measurable = []
        for name in names:
            a = res.tensors[name]
            if name not in base_shapes:
                a.alignable = False
                res.missing_from_base.append(name)
                res.not_alignable.append(name)
                continue
            if base_shapes[name] != target_shapes[name]:
                a.alignable = False
                res.shape_mismatch.append(name)
                res.not_alignable.append(name)
                continue
            measurable.append(name)

        if measure:
            _measure(res, src, measurable, threshold)
            rejected = _reject_unhelpful(topo, descent.perms, res.assessments)
            if rejected:
                # Re-fan the surviving permutations. Only the tensors a
                # rejected group touched can have changed, and only those are
                # re-measured -- the pass costs a float64 norm over every
                # tensor, which is 42s on a 176 MiB checkpoint.
                touched = _members(topo, rejected)
                res.tensors.update(plan(topo, descent.perms, sorted(touched)))
                _measure(res, src, [n for n in measurable if n in touched],
                         threshold)
                res.rejected_groups = sorted(rejected)
            res.not_alignable.extend(
                n for n in measurable if not res.tensors[n].alignable)

        res.identity = all(a.identity for a in res.tensors.values())
        res.wall_clock_s = time.perf_counter() - started
        return res
    finally:
        for x, r in zip((base, target), (rb, rt)):
            if x in opened:
                r.close()