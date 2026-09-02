"""Algorithm 1: PermutationCoordinateDescent.

Every group's cost depends on its neighbours' permutations, and its neighbours'
costs depend on it, so there is no order that solves each group once with
correct inputs. Instead: start everyone at identity, solve one group at a time
against whatever the others currently claim, and sweep until a full pass
changes nothing.

Two properties this buys:

  Fast path. The paper already initialises P <- I, so a first sweep that
  changes nothing means the checkpoints were never permuted -- the fine-tuning
  case. That is not a special case bolted on, it is the ordinary termination
  condition firing on sweep 1.

  Monotonicity. Solving a group maximises exactly those terms of the global
  objective that involve it, holding the rest fixed, so the objective cannot
  decrease. Tracked only when asked: it costs a pass over every tensor.

Only neighbours are re-solved. If nothing adjacent to a group moved, its cost
matrix is identical to last sweep and so is its answer, so the sweep skips it.
That is an optimisation, not a different algorithm -- test_coordinate_descent
asserts it agrees with the naive version group for group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .Error import NotAlignable
from .IR import GroupId, Topology
from .lap import same, solve
from .objective import MatrixSource, group_cost, objective_value

DEFAULT_MAX_SWEEPS = 25


@dataclass
class DescentResult:
    perms: dict[GroupId, "np.ndarray | None"] = field(default_factory=dict)
    sweeps: int = 0
    converged: bool = True
    identity: bool = True
    changed_per_sweep: list[int] = field(default_factory=list)
    solved_per_sweep: list[int] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    unsolved: list[GroupId] = field(default_factory=list)
    seed: int = 0

    @property
    def groups(self) -> int:
        return len(self.perms)

    def summary(self) -> str:
        how = "converged" if self.converged else "hit the sweep cap"
        what = "identity" if self.identity else f"{self.nontrivial} permuted"
        line = (f"{self.groups} groups, {what}, {how} after "
                f"{self.sweeps} sweep{'s' if self.sweeps != 1 else ''}")
        if self.unsolved:
            line += f"; {len(self.unsolved)} group(s) left at identity"
        return line

    @property
    def nontrivial(self) -> int:
        return sum(1 for p in self.perms.values() if p is not None)


def neighbours(topo: Topology) -> dict[GroupId, set[GroupId]]:
    """Groups that share a tensor. Moving one changes the other's cost matrix."""
    out: dict[GroupId, set[GroupId]] = {g: set() for g in topo.groups}
    for name in topo.tensors:
        row = topo.group_for_tensor(name, "row")
        col = topo.group_for_tensor(name, "col")
        if row is not None and col is not None and row.id != col.id:
            out[row.id].add(col.id)
            out[col.id].add(row.id)
    return out


def descend(topo: Topology, src: MatrixSource,
            max_sweeps: int = DEFAULT_MAX_SWEEPS,
            seed: int = 0,
            shuffle: bool = True,
            track_objective: bool = False,
            skip_clean: bool = True,
            skip: "set[GroupId] | None" = None,
            on_sweep=None) -> DescentResult:
    skipped = set(skip or ())
    solvable = [g.id for g in topo.solvable_groups()]
    order = [g for g in solvable if g not in skipped]
    res = DescentResult(perms={g: None for g in solvable}, seed=seed)
    res.unsolved = sorted(g for g in solvable if g in skipped)
    if not order:
        return res

    rng = np.random.default_rng(seed)
    adjacency = neighbours(topo)
    dirty: set[GroupId] = set(order)

    if track_objective:
        res.objective.append(objective_value(topo, src, res.perms))

    for sweep in range(1, max_sweeps + 1):
        seq = list(order)
        if shuffle:
            rng.shuffle(seq)
        visited = [g for g in seq if not skip_clean or g in dirty]

        changed: set[GroupId] = set()
        for gid in visited:
            try:
                p = solve(group_cost(topo, gid, src, res.perms))
            except NotAlignable:
                if gid not in res.unsolved:
                    res.unsolved.append(gid)
                p = None
            if not same(res.perms[gid], p):
                changed.add(gid)
                res.perms[gid] = p

        res.sweeps = sweep
        res.changed_per_sweep.append(len(changed))
        res.solved_per_sweep.append(len(visited))
        if track_objective:
            res.objective.append(objective_value(topo, src, res.perms))
        if on_sweep is not None:
            on_sweep(res)

        dirty = set().union(*(adjacency[g] for g in changed)) if changed else set()
        dirty -= set(res.unsolved)
        if not dirty:
            break
    else:
        res.converged = False

    res.identity = all(p is None for p in res.perms.values())
    return res