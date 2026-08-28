"""Algorithm 1's LAP cost matrix, one group at a time.

Git Re-Basin's update for a group has two terms, because a permutation of a
group's units moves both the rows of every tensor the group produces and the
columns of every tensor that consumes it:

    C = sum over row members   W^target @ W^base.T          [n, n]
      + sum over col members   W^target.T @ W^base          (blockwise)

Only the pair together pins the permutation down. Rows alone tie whenever two
units have identical incoming weights; columns alone tie whenever two units are
read identically downstream.

DIRECTION. C is target-major: C[i, j] scores target unit i against base unit j.
scipy's assignment on that returns col_ind[i] = j, which is exactly
FileFormat.md 4.2's p[i] -- "the base index that target index i was diffed
against" -- with no inversion step anywhere. Neighbouring groups' permutations
are applied the same way round, by gathering the BASE into target order
(base[p]), which is the same gather 9's reconstructor performs. Nothing in this
module ever inverts a permutation; if you find yourself needing to, the
convention has been broken upstream.

Identity is None, never arange(n): a None perm skips the gather entirely.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from .Error import TopologyError
from .IR import GroupId, Topology

Perms = Mapping[GroupId, "np.ndarray | None"]


class MatrixSource(Protocol):
    def base(self, name: str) -> np.ndarray: ...
    def target(self, name: str) -> np.ndarray: ...


def as_matrix(arr: np.ndarray) -> np.ndarray:
    """Fold to the logical 2D [rows, cols] the solver works in."""
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.ndim == 2:
        return a
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


class MatrixPair:
    """Adapts a pair of SafetensorsReaders, or a pair of {name: array} dicts.

    Uncached by default: a 7B group holds several hundred MB of float32 and the
    solver revisits every group each sweep. Cache only for small models.
    """

    def __init__(self, base: object, target: object, cache: bool = False) -> None:
        self._base, self._target = base, target
        self._cache: dict[tuple[str, str], np.ndarray] | None = {} if cache else None

    def _fetch(self, side: str, src: object, name: str) -> np.ndarray:
        if self._cache is not None and (side, name) in self._cache:
            return self._cache[(side, name)]
        getter = getattr(src, "as_float", None)
        if getter is not None:
            mat = as_matrix(getter(name))
        else:
            try:
                mat = as_matrix(src[name])  # type: ignore[index]
            except KeyError:
                raise KeyError(f"{side} source has no tensor '{name}'") from None
        if self._cache is not None:
            self._cache[(side, name)] = mat
        return mat

    def base(self, name: str) -> np.ndarray:
        return self._fetch("base", self._base, name)

    def target(self, name: str) -> np.ndarray:
        return self._fetch("target", self._target, name)


def apply_row_perm(mat: np.ndarray, p: np.ndarray | None) -> np.ndarray:
    """Reindex rows into target order. None is identity."""
    return mat if p is None else mat[p]


def apply_col_perm(mat: np.ndarray, p: np.ndarray | None, block: int = 1) -> np.ndarray:
    """Reindex column blocks into target order. None is identity."""
    if p is None:
        return mat
    rows, cols = mat.shape
    return mat.reshape(rows, cols // block, block)[:, p, :].reshape(rows, cols)


def _perm_for(perms: Perms | None, gid: GroupId | None) -> np.ndarray | None:
    if perms is None or gid is None:
        return None
    return perms.get(gid)


def _pair(src: MatrixSource, name: str) -> tuple[np.ndarray, np.ndarray]:
    t, a = as_matrix(src.target(name)), as_matrix(src.base(name))
    if t.shape != a.shape:
        raise TopologyError(
            f"'{name}': target shape {t.shape} != base shape {a.shape}"
        )
    return t, a


def cost_terms(topo: Topology, gid: GroupId) -> list[tuple[str, str]]:
    """(tensor, axis) pairs contributing to this group's cost -- for -v output."""
    g = _group(topo, gid)
    return ([(n, "row") for n in g.row_members]
            + [(c.name, "col") for c in g.col_members])


def _group(topo: Topology, gid: GroupId):
    g = topo.groups.get(gid)
    if g is None:
        raise TopologyError(f"no such group '{gid}'")
    if g.is_empty:
        raise TopologyError(f"group '{gid}' has no members; nothing to solve")
    return g


def group_cost(topo: Topology, gid: GroupId, src: MatrixSource,
               perms: Perms | None = None) -> np.ndarray:
    """[n, n] float32. C[i, j] = affinity of target unit i to base unit j."""
    g = _group(topo, gid)
    n = g.size
    cost = np.zeros((n, n), dtype=np.float32)

    for name in g.row_members:
        t, a = _pair(src, name)
        if t.shape[0] != n:
            raise TopologyError(
                f"'{name}': {t.shape[0]} rows != group '{gid}' size {n}"
            )
        owner = topo.group_for_tensor(name, "col")
        if owner is not None:
            block = topo.col_block_size(owner.id, name) or 1
            a = apply_col_perm(a, _perm_for(perms, owner.id), block)
        cost += t @ a.T

    for c in g.col_members:
        t, a = _pair(src, c.name)
        rows, cols = t.shape
        if cols != n * c.col_block_size:
            raise TopologyError(
                f"'{c.name}': {cols} columns != group '{gid}' size {n} x "
                f"col_block_size {c.col_block_size}"
            )
        owner = topo.group_for_tensor(c.name, "row")
        if owner is not None:
            a = apply_row_perm(a, _perm_for(perms, owner.id))
        if c.col_block_size == 1:
            cost += t.T @ a
        else:
            k = c.col_block_size
            tb = t.reshape(rows, n, k).transpose(1, 0, 2).reshape(n, rows * k)
            ab = a.reshape(rows, n, k).transpose(1, 0, 2).reshape(n, rows * k)
            cost += tb @ ab.T

    return cost


def assignment_value(cost: np.ndarray, p: np.ndarray | None) -> float:
    """<P, C> for the assignment p. Identity is None."""
    n = cost.shape[0]
    if p is None:
        return float(np.trace(cost))
    return float(cost[np.arange(n), p].sum())


def member_names(topo: Topology) -> list[str]:
    seen: dict[str, None] = {}
    for g in topo.groups.values():
        for name in g.row_members:
            seen.setdefault(name, None)
        for c in g.col_members:
            seen.setdefault(c.name, None)
    return list(seen)


def objective_value(topo: Topology, src: MatrixSource,
                    perms: Perms | None = None,
                    names: Sequence[str] | None = None) -> float:
    """Global weight-matching objective: <W^target, gathered W^base> summed.

    Each tensor counts once here; a tensor appears in two group costs (row and
    column), so sum(<P_g, C_g>) is twice this. Coordinate descent must never
    decrease it.
    """
    total = 0.0
    for name in (member_names(topo) if names is None else names):
        t, a = _pair(src, name)
        row_owner = topo.group_for_tensor(name, "row")
        if row_owner is not None:
            a = apply_row_perm(a, _perm_for(perms, row_owner.id))
        col_owner = topo.group_for_tensor(name, "col")
        if col_owner is not None:
            block = topo.col_block_size(col_owner.id, name) or 1
            a = apply_col_perm(a, _perm_for(perms, col_owner.id), block)
        total += float(np.einsum("ij,ij->", t, a))
    return total