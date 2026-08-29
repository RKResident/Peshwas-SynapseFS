"""Checkpoint -> Topology.

Shapes are normative here; config.json only corroborates. Everything the IR
needs is already in the safetensors header: a 4d weight [out_ch, in_ch, kh, kw]
gives kernel geometry and channel counts, and per axes.py the block size falls
out of consumer.cols // producer.size without any spatial hint. config.json key
names vary by architecture and cannot be relied on for that; what it is good
for is disagreeing with the shapes, which means the wrong config was passed.

ORDERING IS THE TRAP. FileFormat.md 1.2 says safetensors sorts keys
alphabetically, so header order gives 'features.10' before 'features.2', and
'classifier.*' before 'features.*'. Sorting naturally fixes the first and not
the second, so the chain is type-checked afterwards: each layer's column count
must be a whole multiple of the previous layer's output size. If it is not, the
order is wrong and we raise instead of wiring a plausible-looking graph that
silently aligns nothing. Pass order=[...] to override.

NORM layers do not advance the chain, and they are attached by WIDTH, not by
position. 'bn1, bn2, conv1, conv2, fc' sorts with both norms ahead of every
conv, so position would put them on the input axis and collide. A norm's width
is always its producer's output width, so the host is the nearest layer -- back
first, then forward -- that produces exactly that many channels. Its params,
and for BatchNorm its running_mean and running_var, join that host's group.
Those two buffers are per output channel and MUST move with the channel
permutation; leaving them behind is invisible under identity permutations and
wrong under every real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .Error import TopologyError, UnsupportedArchitecture
from .IR import GroupId, LayerKind, LayerNode, TensorRef, Topology, validate
from .group import build_groups

INPUT_GROUP: GroupId = "g.input"

WEIGHT_SUFFIXES = ("weight", "gamma")
BIAS_SUFFIXES = ("bias", "beta")
ROW_BUFFERS = ("running_mean", "running_var")
IGNORED_SUFFIXES = ("num_batches_tracked",)

UNSUPPORTED_HINTS = ("embed", "attn", "attention", "q_proj", "k_proj", "v_proj",
                     "rotary", "rope", "lora")

_NUM = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """Sort 'features.2' before 'features.10'. Alphabetical order does not."""
    return tuple(int(p) if p.isdigit() else p for p in _NUM.split(name))


@dataclass
class Layer:
    stem: str
    kind: LayerKind
    weight: TensorRef
    bias: TensorRef | None = None
    buffers: list[TensorRef] = field(default_factory=list)

    @property
    def out_size(self) -> int:
        return self.weight.shape[0] if self.weight.shape else 1


def split_suffix(name: str) -> tuple[str, str]:
    stem, _, suffix = name.rpartition(".")
    return (stem, suffix) if stem else (name, "")


def classify(ref: TensorRef) -> LayerKind:
    if ref.ndim == 4:
        return LayerKind.CONV
    if ref.ndim == 2:
        return LayerKind.LINEAR
    if ref.ndim == 1:
        return LayerKind.NORM
    return LayerKind.OTHER


def collect(tensors: dict[str, TensorRef]) -> tuple[list[Layer], list[str]]:
    """Group parameters by stem. Returns (layers, tensors we did not place)."""
    parts: dict[str, dict[str, TensorRef]] = {}
    loose: list[str] = []
    for name, ref in tensors.items():
        stem, suffix = split_suffix(name)
        if suffix in IGNORED_SUFFIXES:
            loose.append(name)
        elif suffix in WEIGHT_SUFFIXES + BIAS_SUFFIXES + ROW_BUFFERS:
            parts.setdefault(stem, {})[suffix] = ref
        else:
            loose.append(name)

    layers: list[Layer] = []
    for stem in sorted(parts, key=natural_key):
        got = parts[stem]
        weight = next((got[s] for s in WEIGHT_SUFFIXES if s in got), None)
        if weight is None:
            loose.extend(r.name for r in got.values())
            continue
        for hint in UNSUPPORTED_HINTS:
            if hint in stem.lower():
                raise UnsupportedArchitecture(
                    f"'{stem}' looks like a {hint} layer; this parser wires "
                    "straight conv/linear/norm chains only"
                )
        kind = classify(weight)
        if kind is LayerKind.OTHER:
            loose.extend(r.name for r in got.values())
            continue
        layers.append(Layer(
            stem, kind, weight,
            next((got[s] for s in BIAS_SUFFIXES if s in got), None),
            [got[s] for s in ROW_BUFFERS if s in got],
        ))
    return layers, sorted(loose)


def backbone(layers: list[Layer]) -> list[Layer]:
    return [l for l in layers if l.kind is not LayerKind.NORM]


def check_chain(layers: list[Layer]) -> None:
    """Every consumer's columns must be a whole multiple of its producer."""
    chain = backbone(layers)
    for producer, consumer in zip(chain, chain[1:]):
        cols, size = consumer.weight.cols, producer.out_size
        if cols % size:
            raise UnsupportedArchitecture(
                f"'{consumer.stem}' has {cols} input columns, not divisible by "
                f"the {size} outputs of '{producer.stem}'; pass order=[...] "
                "if these layers are not consecutive"
            )


def infer_order(layers: list[Layer]) -> list[Layer] | None:
    """Recover the layer chain from shapes when the names do not give it.

    ORDERING IS THE TRAP (see the module docstring): safetensors sorts keys
    alphabetically, so `head` precedes `stem` and no amount of natural-sorting
    fixes it. `order=[...]` exists for that, but requiring the caller to supply
    it means alignment silently does nothing on any model whose layer names do
    not happen to sort topologically -- which is most of them.

    So infer it. The only structural fact available is the one `check_chain`
    already tests: a consumer's column count must be a whole multiple of its
    producer's output size. That gives a "can follow" relation over the
    backbone, and the chain is a path through it.

        stem   27 cols (3*3*3), out 64     <- 27 is divisible by no layer's
        conv  576 cols (64*3*3), out 64       output size, so nothing can
        head   64 cols,          out 100      precede it: it is the head

    Greedy from that unique start, breaking ties by natural name order.
    Returns None when the shapes do not determine a chain, in which case the
    caller keeps its existing behaviour and raises with the actionable message.
    """
    chain = backbone(layers)
    if len(chain) < 2:
        return None

    def can_follow(consumer: Layer, producer: Layer) -> bool:
        return (consumer is not producer
                and producer.out_size > 0
                and consumer.weight.cols % producer.out_size == 0)

    starts = [l for l in chain if not any(can_follow(l, p) for p in chain)]
    if len(starts) != 1:
        return None                      # ambiguous or cyclic; do not guess

    ordered = [starts[0]]
    remaining = [l for l in chain if l is not starts[0]]
    while remaining:
        nxt = [l for l in remaining if can_follow(l, ordered[-1])]
        if not nxt:
            return None
        pick = min(nxt, key=lambda l: natural_key(l.stem))
        ordered.append(pick)
        remaining.remove(pick)

    # Norms do not advance the chain and are attached by width later, so they
    # can sit anywhere; keep them in natural order after their host's position.
    norms = [l for l in layers if l.kind is LayerKind.NORM]
    return ordered + sorted(norms, key=lambda l: natural_key(l.stem))


def host_of(norm: Layer, layers: list[Layer]) -> Layer:
    """The layer whose output axis this norm sits on. Width decides, not order."""
    at = layers.index(norm)
    behind = [l for l in layers[:at]
              if l.kind is not LayerKind.NORM and l.out_size == norm.out_size]
    if behind:
        return behind[-1]
    ahead = [l for l in layers[at + 1:]
             if l.kind is not LayerKind.NORM and l.out_size == norm.out_size]
    if ahead:
        return ahead[0]
    produced = sorted({l.out_size for l in backbone(layers)})
    raise UnsupportedArchitecture(
        f"norm '{norm.stem}' has {norm.out_size} channels, which no conv or "
        f"linear layer produces (widths present: {produced})"
    )


def to_nodes(layers: list[Layer]) -> list[LayerNode]:
    groups = {l.stem: f"g.{l.stem}" for l in backbone(layers)}
    for norm in (l for l in layers if l.kind is LayerKind.NORM):
        groups[norm.stem] = groups[host_of(norm, layers).stem]

    nodes: list[LayerNode] = []
    current: GroupId = INPUT_GROUP
    for layer in backbone(layers):
        group = groups[layer.stem]
        kw = {}
        if layer.kind is LayerKind.CONV:
            kw = {"kernel_hw": (layer.weight.shape[2], layer.weight.shape[3]),
                  "in_channels": layer.weight.shape[1],
                  "out_channels": layer.weight.shape[0]}
        nodes.append(LayerNode(layer.stem, layer.kind, layer.weight, layer.bias,
                               current, group, **kw))
        current = group

    for layer in (l for l in layers if l.kind is LayerKind.NORM):
        group = groups[layer.stem]
        nodes.append(LayerNode(layer.stem, LayerKind.NORM, layer.weight,
                               layer.bias, group, group))
        for buf in layer.buffers:
            nodes.append(LayerNode(f"{layer.stem}:{buf.name.rsplit('.', 1)[-1]}",
                                   LayerKind.NORM, buf, None, group, group))
    return nodes


def corroborate(config: dict | None, layers: list[Layer]) -> None:
    """config.json must not contradict the shapes. It never overrides them."""
    if not config:
        return
    widths = [l.out_size for l in layers if l.kind is not LayerKind.NORM]
    for key in ("hidden_sizes", "widths", "channels"):
        declared = config.get(key)
        if isinstance(declared, list) and declared:
            if not set(declared) <= set(widths):
                raise TopologyError(
                    f"config '{key}' = {declared} does not match the widths the "
                    f"checkpoint actually has ({widths}); wrong config.json?"
                )
    for key in ("num_hidden_layers", "num_layers", "depth"):
        declared = config.get(key)
        if isinstance(declared, int) and declared not in (len(widths), len(layers)):
            raise TopologyError(
                f"config '{key}' = {declared} but the checkpoint holds "
                f"{len(widths)} weight layers; wrong config.json?"
            )


def parse(tensors: dict[str, TensorRef],
          config: dict | None = None, *,
          order: list[str] | None = None,
          shared: list[tuple[GroupId, GroupId]] | None = None,
          pin: list[GroupId] | None = None) -> Topology:
    layers, loose = collect(tensors)
    if not layers:
        raise UnsupportedArchitecture("no conv, linear or norm layers found")
    if order is not None:
        index = {stem: i for i, stem in enumerate(order)}
        missing = [l.stem for l in layers if l.stem not in index]
        if missing:
            raise TopologyError(f"order omits {missing}")
        layers.sort(key=lambda l: index[l.stem])
        check_chain(layers)
    else:
        try:
            check_chain(layers)
        except UnsupportedArchitecture:
            # Alphabetical order is not the chain. Recover it from shapes.
            inferred = infer_order(layers)
            if inferred is None:
                raise
            layers = inferred
            check_chain(layers)
    corroborate(config, layers)

    topo = build_groups(to_nodes(layers), tensors=dict(tensors),
                        shared=shared, pin=pin)
    problems = validate(topo)
    if problems:
        raise TopologyError("; ".join(problems))
    return topo


def from_checkpoint(reader, config: dict | None = None, **kw) -> Topology:
    """reader is a SafetensorsReader, or any object exposing refs()."""
    return parse(reader.refs(), config, **kw)


def describe(topo: Topology) -> str:
    """The -v line: group count and sizes, for eyeballing before a long commit."""
    solvable = topo.solvable_groups()
    pinned = [g for g in topo.groups.values() if g.pinned]
    sizes = ", ".join(f"{g.id}={g.size}" for g in solvable) or "none"
    line = (f"{len(topo.groups)} groups ({len(solvable)} solvable, "
            f"{len(pinned)} pinned): {sizes}")
    if topo.unassigned:
        line += f"\n  unassigned ({len(topo.unassigned)}): " \
                + ", ".join(topo.unassigned[:8]) \
                + (" ..." if len(topo.unassigned) > 8 else "")
    return line