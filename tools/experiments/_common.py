"""Shared loading for the measurement scripts.

Every script here reproduces a number that appears in `docs/ARCHITECTURE.md`.
They read real checkpoint pairs rather than synthetic data, because the whole
point of the measurements is that the answers depend on the actual weight
distribution -- several conclusions flipped when tested on a different epoch
range, and one flipped on zstd level alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synapsefs.codec.chunk import dtype_spec
from synapsefs.safetensors_io import SafetensorsFile

DEFAULT_CHECKPOINTS = Path("tools/checkpoints")


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS,
                    help="directory of epochNN.safetensors files")
    ap.add_argument("--newest", type=int, default=25,
                    help="epoch to measure against (the target)")
    return ap


def path_for(root: Path, epoch: int) -> Path:
    return root / f"epoch{epoch:02d}.safetensors"


def load_pair(root: Path, lo: int, hi: int, dtype: str = "F16"):
    """`(target_bits, base_bits, raw_bytes)` concatenated over all tensors.

    Raw *bit patterns*, never floats -- there is no numpy dtype for bf16, and
    the codec never converts. `dtype=None` takes every tensor.
    """
    T, B, raw = [], [], 0
    with SafetensorsFile(path_for(root, lo)) as b, \
         SafetensorsFile(path_for(root, hi)) as t:
        for name in t.names():
            spec = t.spec(name)
            if dtype is not None and spec.dtype != dtype:
                continue
            tb = t.rows(name, 0, spec.num_rows).ravel()
            bb = b.rows(name, 0, spec.num_rows).ravel()
            T.append(tb.copy())
            B.append(bb.copy())
            raw += tb.nbytes
    if not T:
        raise SystemExit(f"no {dtype} tensors found in {path_for(root, hi)}")
    return np.concatenate(T), np.concatenate(B), raw


def load_streams(root: Path, lo: int, hi: int):
    """Per-tensor `(target, base, width)`, for schemes that must not mix
    tensors -- compressing a concatenation flatters any scheme that benefits
    from long runs."""
    out = []
    with SafetensorsFile(path_for(root, lo)) as b, \
         SafetensorsFile(path_for(root, hi)) as t:
        for name in t.names():
            spec = t.spec(name)
            width, _ = dtype_spec(spec.dtype)
            out.append((t.rows(name, 0, spec.num_rows).ravel().copy(),
                        b.rows(name, 0, spec.num_rows).ravel().copy(), width))
    return out


def header(title: str, subtitle: str = "") -> None:
    print(title)
    if subtitle:
        print(subtitle)
    print()
