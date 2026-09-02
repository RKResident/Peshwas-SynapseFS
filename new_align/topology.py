from __future__ import annotations

import re
from typing import NamedTuple

from .Error import TopologyError
from .IR import Topology

INPUT = "g.input"
SUFFIXES = ("weight", "bias", "running_mean", "running_var")
_DIGITS = re.compile(r"(\d+)")


class Layer(NamedTuple):
    stem: str
    rank: int
    width: int          # units this layer produces
    cols: int           # its weight's column count, flattened
    in_units: int       # units it reads, before any block expansion
    weight: str
    names: list[str]    # every tensor sitting on this layer's output axis


def natural_key(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in _DIGITS.split(name))

def _prefix_share(a: str, b: str) -> int:
    """How many leading dot-separated components two stems share, e.g.
    'layer1.0.bn1' and 'layer1.0.conv1' share 2 ('layer1', '0')."""
    n = 0
    for x, y in zip(a.split("."), b.split(".")):
        if x != y:
            break
        n += 1
    return n

def col_block(cols: int, size: int, name: str) -> int:
    if cols % size:
        raise TopologyError(
            f"'{name}' reads {cols} columns, not divisible by the {size} units "
            f"before it; the layer order is wrong "
        )
    return cols // size


def layers(shapes: dict[str, tuple[int, ...]], order: list[str] | None = None) -> list[Layer]:
    grouped: dict[str, dict[str, str]] = {}
    for name in shapes:
        stem, _, suffix = name.rpartition(".")
        if stem and suffix in SUFFIXES:
            grouped.setdefault(stem, {})[suffix] = name

    out = []
    for stem, parts in grouped.items():
        weight = parts.get("weight")
        if weight is None:
            continue
        shape = shapes[weight]
        if not shape or len(shape) == 3 or len(shape) > 4:
            continue
        cols = 1
        for d in shape[1:]:
            cols *= d
        out.append(Layer(stem, len(shape), shape[0], cols,
                         shape[1] if len(shape) > 1 else 1, weight,
                         [parts[s] for s in SUFFIXES if s in parts]))

    rank = {stem: i for i, stem in enumerate(order)} if order else None
    if rank is not None:
        missing = [l.stem for l in out if l.stem not in rank]
        if missing:
            raise TopologyError(f"order omits {missing}")
    return sorted(out, key=lambda l: rank[l.stem] if rank else natural_key(l.stem))


def parse(shapes: dict[str, tuple[int, ...]],
          order: list[str] | None = None) -> Topology:
    found = layers(shapes, order)
    backbone = [l for l in found if l.rank > 1]
    if not backbone:
        raise TopologyError("no conv or linear layers found")

    topo = Topology()
    axis, size = INPUT, backbone[0].in_units
    for layer in backbone:
        topo.add(axis, size)
        topo.own_cols(axis, layer.weight, col_block(layer.cols, size, layer.weight))
        axis, size = f"g.{layer.stem}", layer.width
        topo.add(axis, size)
        topo.own_rows(axis, *layer.names)

    for i, norm in enumerate(found):
        if norm.rank > 1:
            continue
        behind = [l for l in found[:i] if l.rank > 1 and l.width == norm.width]
        ahead = [l for l in found[i + 1:] if l.rank > 1 and l.width == norm.width]
        if not behind and not ahead:
            raise TopologyError(
                f"norm '{norm.stem}' has {norm.width} channels, which no conv or "
                f"linear layer produces; this is not a straight chain"
            )
        
        best = max(_prefix_share(norm.stem, l.stem) for l in behind + ahead)
        best_behind = [l for l in behind if _prefix_share(norm.stem, l.stem) == best]
        best_ahead = [l for l in ahead if _prefix_share(norm.stem, l.stem) == best]
        host = best_behind[-1] if best_behind else best_ahead[0]
        
        topo.own_rows(f"g.{host.stem}", *norm.names)

    # In a chain the input is consumed but never produced, and the last layer's
    # axis is produced but never consumed. Both are boundaries: pin to identity.
    topo.groups[INPUT].pinned = True
    topo.groups[axis].pinned = True

    claimed = set(topo.row_owner) | set(topo.col_owner)
    topo.unassigned = sorted(n for n in shapes if n not in claimed)

    return topo