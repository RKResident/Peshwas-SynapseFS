"""Build PermutationGroups from the layer graph.

A node's out_group owns its weight's ROW axis; its in_group owns the
COLUMN axis. That is an axis-level statement, not a data-flow one -- an
embedding [vocab, hidden] has out_group = the vocab axis and in_group =
the residual stream, even though data flows the other way.

The parser assigns seed ids; axes that must share one permutation -- a
transformer residual stream, a skip connection -- are merged here by
union-find rather than by the parser getting the naming right up front.

Boundary groups are detected structurally: an axis produced but never
consumed is an output, consumed but never produced is an input. Both are
pinned to identity.
"""

from __future__ import annotations

from dataclasses import replace

from .axes import col_block_size_for, permutes_columns, producer_size
from .Error import TopologyError
from .IR import GroupId, LayerKind, LayerNode, PermutationGroup, TensorRef, Topology


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[GroupId, GroupId] = {}

    def add(self, x: GroupId) -> GroupId:
        self._parent.setdefault(x, x)
        return x

    def find(self, x: GroupId) -> GroupId:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: GroupId, b: GroupId) -> GroupId:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        root, other = (ra, rb) if ra <= rb else (rb, ra)
        self._parent[other] = root
        return root

    def members(self) -> dict[GroupId, list[GroupId]]:
        out: dict[GroupId, list[GroupId]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return {k: sorted(v) for k, v in out.items()}


def _resolve(nodes: list[LayerNode], shared: list[tuple[GroupId, GroupId]]
             ) -> tuple[UnionFind, dict[int, GroupId], dict[int, GroupId]]:
    uf = UnionFind()
    for n in nodes:
        if n.out_group is not None:
            uf.add(n.out_group)
        if n.in_group is not None:
            uf.add(n.in_group)
    for a, b in shared:
        uf.union(a, b)
    ins = {id(n): uf.find(n.in_group) for n in nodes if n.in_group is not None}
    outs = {id(n): uf.find(n.out_group) for n in nodes if n.out_group is not None}
    return uf, ins, outs


def _producers(nodes: list[LayerNode], outs: dict[int, GroupId]
               ) -> dict[GroupId, LayerNode]:
    found: dict[GroupId, LayerNode] = {}
    for n in nodes:
        gid = outs.get(id(n))
        if gid is None or n.kind is LayerKind.NORM:
            continue
        prev = found.get(gid)
        if prev is None:
            found[gid] = n
        elif producer_size(prev) != producer_size(n):
            raise TopologyError(
                f"group '{gid}' produced by '{prev.name}' (size "
                f"{producer_size(prev)}) and '{n.name}' (size "
                f"{producer_size(n)})"
            )
    return found


def _axis_size(gid: GroupId, producer: LayerNode | None,
               consumer: LayerNode) -> int:
    if producer is not None:
        return producer_size(producer)
    if consumer.in_channels is not None:
        return consumer.in_channels
    return consumer.weight.cols


def build_groups(nodes: list[LayerNode],
                 tensors: dict[str, TensorRef] | None = None,
                 shared: list[tuple[GroupId, GroupId]] | None = None,
                 pin: list[GroupId] | None = None) -> Topology:
    shared = list(shared or [])
    uf, ins, outs = _resolve(nodes, shared)
    producers = _producers(nodes, outs)

    topo = Topology()
    topo.nodes = [
        replace(n, in_group=ins.get(id(n)), out_group=outs.get(id(n)))
        for n in nodes
    ]
    topo.tensors = dict(tensors) if tensors else {}
    if not topo.tensors:
        for n in nodes:
            topo.tensors[n.weight.name] = n.weight
            if n.bias is not None:
                topo.tensors[n.bias.name] = n.bias

    def ensure(gid: GroupId, size: int) -> PermutationGroup:
        g = topo.groups.get(gid)
        if g is None:
            g = topo.groups[gid] = PermutationGroup(gid, size)
        elif g.size != size:
            raise TopologyError(
                f"group '{gid}' size {g.size} contradicted by {size}"
            )
        return g

    for n in nodes:
        gid = outs.get(id(n))
        if gid is None:
            continue
        if n.kind is LayerKind.NORM:
            size = n.weight.rows
        else:
            size = producer_size(n)
        g = ensure(gid, size)
        g.add_row(n.weight.name)
        if n.bias is not None:
            g.add_row(n.bias.name)

    for n in nodes:
        gid = ins.get(id(n))
        if gid is None or n.kind is LayerKind.NORM:
            continue
        if not permutes_columns(n.kind):
            continue
        size = _axis_size(gid, producers.get(gid), n)
        block = col_block_size_for(n.weight.cols, size, n.weight.name)
        ensure(gid, size).add_col(n.weight.name, block)

    produced = {outs[id(n)] for n in nodes
                if id(n) in outs and n.kind is not LayerKind.NORM}
    consumed = {ins[id(n)] for n in nodes if id(n) in ins}

    topo.output_groups = {g for g in produced - consumed if g in topo.groups}
    topo.input_groups = {g for g in consumed - produced if g in topo.groups}
    for gid in topo.output_groups | topo.input_groups | set(pin or ()):
        if gid not in topo.groups:
            raise TopologyError(f"cannot pin unknown group '{gid}'")
        topo.pin(gid)

    claimed = {name for g in topo.groups.values()
               for name in g.row_members} | {
        c.name for g in topo.groups.values() for c in g.col_members}
    topo.unassigned = sorted(n for n in topo.tensors if n not in claimed)
    return topo


def merged_view(nodes: list[LayerNode],
                shared: list[tuple[GroupId, GroupId]] | None = None
                ) -> dict[GroupId, list[GroupId]]:
    uf, _, _ = _resolve(nodes, list(shared or []))
    return uf.members()