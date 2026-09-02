from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .IR import Topology

MAX_SWEEPS = 10


class Solution(NamedTuple):
    perms: dict[str, "np.ndarray | None"]
    sweeps: int
    converged: bool


def gather_rows(mat: np.ndarray, p: np.ndarray | None) -> np.ndarray:
    return mat if p is None else mat[p]


def gather_cols(mat: np.ndarray, p: np.ndarray | None, block: int) -> np.ndarray:
    if p is None:
        return mat
    rows, cols = mat.shape
    return mat.reshape(rows, cols // block, block)[:, p, :].reshape(rows, cols)


def _blocks(mat: np.ndarray, n: int, block: int) -> np.ndarray:
    """[rows, n*block] -> [n, rows*block], so one GEMM scores every block pair."""
    rows = mat.shape[0]
    return mat.reshape(rows, n, block).transpose(1, 0, 2).reshape(n, rows * block)


def cost(topo: Topology, gid: str, base, target, perms: dict) -> np.ndarray:
    """[n, n] float32. C[i, j] = affinity of target unit i to base unit j."""
    group = topo.groups[gid]
    n = group.size
    out = np.zeros((n, n), dtype=np.float32)

    for name in group.rows:
        a, t = base.matrix(name), target.matrix(name)
        col = topo.col_owner.get(name)
        if col is not None:
            a = gather_cols(a, perms.get(col.group), col.block)
        out += t @ a.T

    for name in group.cols:
        a, t = base.matrix(name), target.matrix(name)
        row = topo.row_owner.get(name)
        if row is not None:
            a = gather_rows(a, perms.get(row))
        block = topo.col_owner[name].block
        if block == 1:
            out += t.T @ a
        else:
            out += _blocks(t, n, block) @ _blocks(a, n, block).T

    # A NaN weight contributes no similarity, which is the right meaning. The
    # tensor is flagged separately by its residual.
    return np.nan_to_num(out, copy=False)


def solve(topo: Topology, base, target, seed: int = 0,
          max_sweeps: int = MAX_SWEEPS, skip=()) -> Solution:
    order = [g.id for g in topo.solvable() if g.id not in skip]
    perms = {gid: np.arange(topo.groups[gid].size) for gid in order}
    rng = np.random.default_rng(seed)

    sweeps, converged = 0, True
    for sweeps in range(1, max_sweeps + 1):
        rng.shuffle(order)
        changed = False
        for gid in order:
            new = linear_sum_assignment(cost(topo, gid, base, target, perms),
                                        maximize=True)[1]
            changed |= not np.array_equal(new, perms[gid])
            perms[gid] = new
        if not changed:
            break
    else:
        converged = False

    return Solution(
        {gid: None if np.array_equal(p, np.arange(p.size)) else p.astype(np.int32)
         for gid, p in perms.items()},
        sweeps, converged)