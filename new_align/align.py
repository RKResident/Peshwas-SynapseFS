from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .match import gather_cols, gather_rows, solve
from .reader import SafetensorsReader
from .topology import parse

THRESHOLD = 0.5


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
    group_sizes: dict[str, int] = field(default_factory=dict)
    not_alignable: list[str] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    unsolved_groups: list[str] = field(default_factory=list)
    identity: bool = True
    sweeps: int = 0
    converged: bool = True
    wall_clock_s: float = 0.0
    seed: int = 0

    @property
    def groups(self) -> int:
        return len(self.group_sizes)

    @property
    def solved(self) -> bool:
        """Distinguishes identity-because-correct from identity-because-we-could-
        not-try. The two look the same in the output and must never be conflated."""
        return not self.unsolved_groups and self.converged

    def __getitem__(self, name: str) -> TensorAlignment:
        return self.tensors[name]


def _norm(a: np.ndarray) -> float:
    """Accumulated in float64: a float32 dot over ten million elements drifts
    past 1e-6, and this is the number a threshold decision is made on."""
    flat = a.reshape(-1)
    return float(np.sqrt(np.einsum("i,i->", flat, flat, dtype=np.float64)))


def residual(target: np.ndarray, base: np.ndarray) -> float:
    scale = _norm(target)
    if scale == 0.0:
        return 0.0 if np.array_equal(target, base) else float("inf")
    value = _norm(target - base) / scale
    return value if np.isfinite(value) else float("inf")


def _aligned(base: np.ndarray, a: TensorAlignment) -> np.ndarray:
    out = gather_rows(base, a.pi_row)
    return out if a.pi_col is None else gather_cols(out, a.pi_col, a.col_block_size)


def align_checkpoints(base_path, target_path, *, seed: int = 0,
                      no_align: bool = False,
                      order: "list[str] | None" = None) -> AlignmentResult:
    started = time.perf_counter()
    with SafetensorsReader(base_path) as base:
       with SafetensorsReader(target_path) as target:
        shapes = target.shapes
        topo = parse(shapes, order)
        res = AlignmentResult(seed=seed,
                              group_sizes={g.id: g.size for g in topo.groups.values()})

        # A tensor the base cannot supply would be read mid-sweep, after real
        # work, and fail there. Its groups are excluded before the search starts.
        base_shapes = base.shapes
        bad = {n for n, s in shapes.items() if base_shapes.get(n) != s}
        unusable = {g.id for g in topo.groups.values()
                    if bad & (set(g.rows) | set(g.cols))}
        res.unsolved_groups = sorted(unusable & {g.id for g in topo.solvable()})

        if no_align:
            perms: dict = {}
        else:
            perms, res.sweeps, res.converged = solve(topo, base, target, seed=seed,
                                                     skip=unusable)

        res.unassigned = list(topo.unassigned)
        for name in shapes:
            col = topo.col_owner.get(name)
            a = TensorAlignment(
                pi_row=perms.get(topo.row_owner.get(name)),
                pi_col=perms.get(col.group) if col else None,
                col_block_size=col.block if col else 1,
            )
            res.tensors[name] = a
            if name in bad:
                a.alignable = False
                res.not_alignable.append(name)
                continue
            t, b = target.matrix(name), base.matrix(name)
            a.residual_pre = residual(t, b)
            a.residual_post = residual(t, _aligned(b, a))
            a.alignable = a.residual_post < THRESHOLD
            if not a.alignable:
                res.not_alignable.append(name)

        res.identity = all(a.identity for a in res.tensors.values())
        res.wall_clock_s = time.perf_counter() - started
        return res