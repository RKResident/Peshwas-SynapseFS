"""Axis geometry.

Rows always move one at a time. Columns sometimes move in contiguous blocks:
after a conv->linear flatten, output channel c owns columns
[c*block, (c+1)*block). This module derives that block size.

One rule covers every case:

    block = consumer.cols // producer.size

  linear -> linear   cols = in_features,        size = in_features   -> 1
  conv   -> conv     cols = in_ch * kh * kw,    size = in_ch         -> kh*kw
  conv   -> linear   cols = in_ch * H * W,      size = in_ch         -> H*W
"""

from __future__ import annotations

from math import prod

from .Error import TopologyError
from .IR import LayerKind, LayerNode, TensorRef

NO_COLUMN_AXIS = frozenset({LayerKind.NORM})


def row_axis_length(shape: tuple[int, ...]) -> int:
    return shape[0] if shape else 1


def col_axis_length(shape: tuple[int, ...]) -> int:
    return prod(shape[1:]) if len(shape) > 1 else 1


def permutes_columns(kind: LayerKind) -> bool:
    return kind not in NO_COLUMN_AXIS


def assert_block_divides(cols: int, block: int, name: str = "<tensor>") -> None:
    if block < 1:
        raise TopologyError(f"{name}: col_block_size {block} must be >= 1")
    if cols % block:
        raise TopologyError(
            f"{name}: {cols} columns not divisible by col_block_size {block}"
        )


def col_block_size_for(consumer_cols: int, producer_size: int,
                       name: str = "<tensor>") -> int:
    """Columns per producer unit. Raises rather than defaulting to 1."""
    if producer_size < 1:
        raise TopologyError(f"{name}: producer size {producer_size} must be >= 1")
    if consumer_cols % producer_size:
        raise TopologyError(
            f"{name}: {consumer_cols} columns not divisible by producer size "
            f"{producer_size}; cannot derive col_block_size"
        )
    return consumer_cols // producer_size


def producer_size(node: LayerNode) -> int:
    if node.out_channels is not None:
        return node.out_channels
    return node.weight.rows


def col_block_size_from_nodes(consumer: LayerNode, producer: LayerNode) -> int:
    if not permutes_columns(consumer.kind):
        return 1

    size = producer_size(producer)
    block = col_block_size_for(consumer.weight.cols, size, consumer.weight.name)

    if consumer.kind is LayerKind.CONV and consumer.kernel_hw is not None:
        expected = consumer.kernel_hw[0] * consumer.kernel_hw[1]
        if block != expected:
            raise TopologyError(
                f"{consumer.weight.name}: derived col_block_size {block} != "
                f"kernel {consumer.kernel_hw[0]}x{consumer.kernel_hw[1]} = {expected}"
            )
    return block


def spatial_cells(consumer_cols: int, producer_channels: int) -> int:
    """H*W of the feature map entering a linear layer after a conv flatten."""
    return col_block_size_for(consumer_cols, producer_channels)


def block_columns(block_index: int, block: int) -> range:
    return range(block_index * block, (block_index + 1) * block)


def block_count(ref: TensorRef, block: int) -> int:
    assert_block_divides(ref.cols, block, ref.name)
    return ref.cols // block