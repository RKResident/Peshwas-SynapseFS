"""Topology IR: the frozen schema the alignment solver codes against.

A PermutationGroup is one permutable axis, shared by every tensor that touches
it. Solving alignment means choosing one permutation per non-pinned group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod

GroupId = str


class LayerKind(str, Enum):
    LINEAR = "linear"
    CONV = "conv"
    NORM = "norm"
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    OTHER = "other"


@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def cols(self) -> int:
        return prod(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class ColMember:
    """A tensor whose column axis a group permutes, in blocks of col_block_size."""

    name: str
    col_block_size: int = 1


@dataclass
class LayerNode:
    name: str
    kind: LayerKind
    weight: TensorRef
    bias: TensorRef | None = None
    in_group: GroupId | None = None
    out_group: GroupId | None = None
    kernel_hw: tuple[int, int] | None = None
    in_channels: int | None = None
    out_channels: int | None = None


@dataclass
class PermutationGroup:
    id: GroupId
    size: int
    pinned: bool = False
    row_members: list[str] = field(default_factory=list)
    col_members: list[ColMember] = field(default_factory=list)

    def add_row(self, name: str) -> None:
        if name not in self.row_members:
            self.row_members.append(name)

    def add_col(self, name: str, col_block_size: int = 1) -> None:
        if not any(c.name == name for c in self.col_members):
            self.col_members.append(ColMember(name, col_block_size))

    @property
    def is_empty(self) -> bool:
        return not self.row_members and not self.col_members


@dataclass
class Topology:
    tensors: dict[str, TensorRef] = field(default_factory=dict)
    nodes: list[LayerNode] = field(default_factory=list)
    groups: dict[GroupId, PermutationGroup] = field(default_factory=dict)
    input_groups: set[GroupId] = field(default_factory=set)
    output_groups: set[GroupId] = field(default_factory=set)
    unassigned: list[str] = field(default_factory=list)

    def group(self, gid: GroupId) -> PermutationGroup:
        return self.groups[gid]

    def solvable_groups(self) -> list[PermutationGroup]:
        return [g for g in self.groups.values() if not g.pinned and not g.is_empty]

    def pin(self, gid: GroupId) -> None:
        self.groups[gid].pinned = True

    def group_for_tensor(self, name: str, axis: str) -> PermutationGroup | None:
        for g in self.groups.values():
            if axis == "row" and name in g.row_members:
                return g
            if axis == "col" and any(c.name == name for c in g.col_members):
                return g
        return None

    def col_block_size(self, gid: GroupId, name: str) -> int | None:
        for c in self.groups[gid].col_members:
            if c.name == name:
                return c.col_block_size
        return None


def validate(topo: Topology) -> list[str]:
    """Return human-readable complaints. Empty list means the IR is well-formed."""
    problems: list[str] = []
    seen_row: dict[str, GroupId] = {}
    seen_col: dict[str, GroupId] = {}

    for gid, g in topo.groups.items():
        if gid != g.id:
            problems.append(f"group '{gid}' keyed under mismatched id '{g.id}'")
        if g.size < 1:
            problems.append(f"group '{gid}' has non-positive size {g.size}")
        if g.is_empty:
            problems.append(f"group '{gid}' has no members")
        if not g.row_members and g.col_members and not g.pinned:
            problems.append(f"group '{gid}' has column members but no row members")

        for name in g.row_members:
            ref = topo.tensors.get(name)
            if ref is None:
                problems.append(f"group '{gid}' row member '{name}' not in tensor table")
                continue
            if ref.rows != g.size:
                problems.append(
                    f"'{name}' rows {ref.rows} != group '{gid}' size {g.size}"
                )
            if name in seen_row:
                problems.append(
                    f"'{name}' row axis claimed by '{seen_row[name]}' and '{gid}'"
                )
            else:
                seen_row[name] = gid

        for c in g.col_members:
            ref = topo.tensors.get(c.name)
            if ref is None:
                problems.append(
                    f"group '{gid}' col member '{c.name}' not in tensor table"
                )
                continue
            if c.col_block_size < 1:
                problems.append(
                    f"'{c.name}' col_block_size {c.col_block_size} must be >= 1"
                )
            elif ref.cols % c.col_block_size:
                problems.append(
                    f"'{c.name}' cols {ref.cols} not divisible by "
                    f"col_block_size {c.col_block_size}"
                )
            elif ref.cols // c.col_block_size != g.size:
                problems.append(
                    f"'{c.name}' has {ref.cols // c.col_block_size} column blocks "
                    f"!= group '{gid}' size {g.size}"
                )
            if c.name in seen_col:
                problems.append(
                    f"'{c.name}' col axis claimed by '{seen_col[c.name]}' and '{gid}'"
                )
            else:
                seen_col[c.name] = gid

    for gid in topo.input_groups | topo.output_groups:
        if gid not in topo.groups:
            problems.append(f"boundary group '{gid}' not in groups")
        elif not topo.groups[gid].pinned:
            problems.append(f"boundary group '{gid}' is not pinned")

    for node in topo.nodes:
        for gid, axis in ((node.in_group, "in"), (node.out_group, "out")):
            if gid is not None and gid not in topo.groups:
                problems.append(f"node '{node.name}' {axis}_group '{gid}' undefined")

    for name in topo.unassigned:
        if name not in topo.tensors:
            problems.append(f"unassigned tensor '{name}' not in tensor table")
        elif name in seen_row or name in seen_col:
            problems.append(f"'{name}' is both unassigned and a group member")

    return problems