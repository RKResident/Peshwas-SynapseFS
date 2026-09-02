from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class ColRef(NamedTuple):
    group: str
    block: int


@dataclass
class Group:
    id: str
    size: int
    rows: list[str] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    pinned: bool = False


@dataclass
class Topology:
    groups: dict[str, Group] = field(default_factory=dict)
    row_owner: dict[str, str] = field(default_factory=dict)
    col_owner: dict[str, ColRef] = field(default_factory=dict)
    unassigned: list[str] = field(default_factory=list)

    def solvable(self) -> list[Group]:
        return [g for g in self.groups.values() if not g.pinned]

    def add(self, gid: str, size: int) -> Group:
        g = self.groups.get(gid)
        if g is None:
            g = self.groups[gid] = Group(gid, size)
        return g

    def own_rows(self, gid: str, *names: str) -> None:
        g = self.groups[gid]
        for name in names:
            g.rows.append(name)
            self.row_owner[name] = gid

    def own_cols(self, gid: str, name: str, block: int) -> None:
        self.groups[gid].cols.append(name)
        self.col_owner[name] = ColRef(gid, block)