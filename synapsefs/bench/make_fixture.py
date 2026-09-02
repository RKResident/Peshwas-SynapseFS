"""Ground-truth fixtures: pairs of checkpoints with a known permutation.

The target is built by walking the same Topology the solver walks, which is
circular -- a bug in the IR would produce a fixture wrong in the way that hides
it. So every permuted fixture is also checked FUNCTIONALLY: a numpy forward
pass over both checkpoints must produce the same outputs on the same input.
That is the actual definition of the problem, and it does not go through the
IR, so it catches a mis-derived col_block_size or a buffer left behind.

Four variants per architecture:

  permuted    every solvable group shuffled, weights otherwise identical
  noisy       shuffled, plus gaussian noise -- two real runs never match bitwise
  finetune    no shuffle, small noise. Identity is the right answer.
  unrelated   fresh weights. Not alignable at all.

Written as real .safetensors (F16 by default, BF16 and F32 available) so the
whole stack is exercised, mmap and dtype widening included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synapsefs.align.config_parser import parse
from synapsefs.align.IR import TensorRef
from synapsefs.align.reader import SafetensorsReader

VARIANTS = ("permuted", "noisy", "finetune", "unrelated")


def encode(arr: np.ndarray, dtype: str) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if dtype == "F32":
        return a.tobytes()
    if dtype == "F16":
        return a.astype(np.float16).tobytes()
    if dtype == "BF16":
        bits = a.view(np.uint32)
        return (((bits + 0x8000 + ((bits >> 16) & 1)) >> 16)
                .astype(np.uint16).tobytes())
    raise ValueError(f"unsupported fixture dtype {dtype}")


def write_safetensors(path: str, arrays: dict[str, np.ndarray],
                      dtype: str = "F16", metadata: dict | None = None) -> str:
    blob: dict = {}
    if metadata:
        blob["__metadata__"] = metadata
    body, off = b"", 0
    for name in sorted(arrays):
        payload = encode(arrays[name], dtype)
        blob[name] = {"dtype": dtype, "shape": list(np.shape(arrays[name])),
                      "data_offsets": [off, off + len(payload)]}
        body += payload
        off += len(payload)
    js = json.dumps(blob, separators=(",", ":")).encode("utf-8")
    js += b" " * (-(8 + len(js)) % 8)
    with open(path, "wb") as f:
        f.write(len(js).to_bytes(8, "little") + js + body)
    return path


def build_mlp(widths: list[int], rng) -> tuple[dict, dict]:
    """widths = [in, h1, ..., out]."""
    arrays = {}
    for i, (fan_in, fan_out) in enumerate(zip(widths, widths[1:]), start=1):
        scale = np.float32(np.sqrt(2.0 / fan_in))
        arrays[f"l{i}.weight"] = (rng.standard_normal((fan_out, fan_in))
                                  * scale).astype(np.float32)
        arrays[f"l{i}.bias"] = (rng.standard_normal(fan_out)
                                * 0.1).astype(np.float32)
    return arrays, {"model_type": "mlp", "hidden_sizes": widths[1:-1]}


def build_cnn(channels: list[int], classes: int, rng, bn: bool = True,
              kernel: int = 3, out_hw: int = 5) -> tuple[dict, dict]:
    """channels = [in_ch, c1, c2, ...]; the map ends at out_hw x out_hw."""
    arrays = {}
    for i, (cin, cout) in enumerate(zip(channels, channels[1:]), start=1):
        scale = np.float32(np.sqrt(2.0 / (cin * kernel * kernel)))
        arrays[f"conv{i}.weight"] = (
            rng.standard_normal((cout, cin, kernel, kernel)) * scale).astype(np.float32)
        arrays[f"conv{i}.bias"] = (rng.standard_normal(cout) * 0.1).astype(np.float32)
        if bn:
            arrays[f"bn{i}.weight"] = (1.0 + 0.1 * rng.standard_normal(cout)).astype(np.float32)
            arrays[f"bn{i}.bias"] = (0.1 * rng.standard_normal(cout)).astype(np.float32)
            arrays[f"bn{i}.running_mean"] = (0.1 * rng.standard_normal(cout)).astype(np.float32)
            arrays[f"bn{i}.running_var"] = (1.0 + 0.1 * rng.standard_normal(cout) ** 2).astype(np.float32)
    flat = channels[-1] * out_hw * out_hw
    arrays["fc.weight"] = (rng.standard_normal((classes, flat))
                           * np.float32(np.sqrt(2.0 / flat))).astype(np.float32)
    arrays["fc.bias"] = (rng.standard_normal(classes) * 0.1).astype(np.float32)
    config = {"model_type": "cnn", "channels": channels[1:],
              "kernel_size": kernel, "num_classes": classes}
    return arrays, config


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    out_ch, in_ch, kh, kw = w.shape
    _, h, wd = x.shape
    oh, ow = h - kh + 1, wd - kw + 1
    flat = w.reshape(out_ch, -1)
    out = np.zeros((out_ch, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j] = flat @ x[:, i:i + kh, j:j + kw].reshape(-1)
    if b is not None:
        out += b[:, None, None]
    return out


def batchnorm(x: np.ndarray, g, b, mean, var, eps=1e-5) -> np.ndarray:
    shape = (-1, 1, 1) if x.ndim == 3 else (-1,)
    return ((x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
            * g.reshape(shape) + b.reshape(shape))


def forward(arrays: dict[str, np.ndarray], x: np.ndarray, kind: str) -> np.ndarray:
    """A plain numpy forward pass. Deliberately does not consult the IR."""
    if kind == "mlp":
        depth = sum(1 for k in arrays if k.startswith("l") and k.endswith(".weight"))
        for i in range(1, depth + 1):
            x = arrays[f"l{i}.weight"] @ x + arrays[f"l{i}.bias"]
            if i < depth:
                x = np.maximum(x, 0)
        return x
    convs = sum(1 for k in arrays if k.startswith("conv") and k.endswith(".weight"))
    for i in range(1, convs + 1):
        x = conv2d(x, arrays[f"conv{i}.weight"], arrays[f"conv{i}.bias"])
        if f"bn{i}.weight" in arrays:
            x = batchnorm(x, arrays[f"bn{i}.weight"], arrays[f"bn{i}.bias"],
                          arrays[f"bn{i}.running_mean"], arrays[f"bn{i}.running_var"])
        x = np.maximum(x, 0)
    return arrays["fc.weight"] @ x.reshape(-1) + arrays["fc.bias"]


def topology_of(arrays: dict[str, np.ndarray], config: dict | None = None):
    return parse({n: TensorRef(n, np.shape(a), "F32") for n, a in arrays.items()},
                 config)


def permute(arrays: dict[str, np.ndarray], topo, rng) -> tuple[dict, dict]:
    """Shuffle every solvable group. Returns (target, {group: permutation})."""
    truth = {g.id: rng.permutation(g.size) for g in topo.solvable_groups()}
    out = {k: np.array(v, copy=True) for k, v in arrays.items()}
    for gid, p in truth.items():
        group = topo.groups[gid]
        for name in group.row_members:
            out[name] = out[name][p]
        for member in group.col_members:
            a = out[member.name]
            shape = a.shape
            flat = a.reshape(shape[0], -1)
            k = member.col_block_size
            out[member.name] = (flat.reshape(shape[0], -1, k)[:, p, :]
                                .reshape(shape))
    return out, truth


def jitter(arrays: dict[str, np.ndarray], sigma: float, rng) -> dict:
    """Noise relative to each tensor's own spread -- absolute sigma would be
    negligible on a narrow layer and overwhelming on a wide one."""
    out = {}
    for k, v in arrays.items():
        scale = float(np.std(v)) or 1.0
        out[k] = (v + sigma * scale * rng.standard_normal(np.shape(v))).astype(np.float32)
    return out


def check_equivalent(base: dict, target: dict, kind: str, rng,
                     shape, tol: float = 2e-2) -> float:
    """The permuted net must compute the same function. Independent of the IR."""
    x = rng.standard_normal(shape).astype(np.float32)
    a, b = forward(base, x, kind), forward(target, x, kind)
    scale = float(np.abs(a).max()) or 1.0
    gap = float(np.abs(a - b).max()) / scale
    if gap > tol:
        raise AssertionError(
            f"permuted fixture is not function-preserving: max relative "
            f"difference {gap:.4f} over {a.size} outputs"
        )
    return gap


SPECS = {
    "mlp-small": dict(kind="mlp", widths=[16, 32, 24, 10]),
    "mlp-medium": dict(kind="mlp", widths=[128, 256, 256, 10]),
    "mlp-large": dict(kind="mlp", widths=[512, 1024, 512, 10]),
    "cnn-small": dict(kind="cnn", channels=[1, 4, 8], classes=10, bn=True),
    "cnn-medium": dict(kind="cnn", channels=[3, 16, 32], classes=10, bn=True),
    "cnn-nobn": dict(kind="cnn", channels=[3, 8, 16], classes=10, bn=False),
}
LARGE = {"mlp-xl": dict(kind="mlp", widths=[2048, 4096, 2048, 10])}


def build(spec: dict, rng):
    if spec["kind"] == "mlp":
        arrays, config = build_mlp(spec["widths"], rng)
        return arrays, config, (spec["widths"][0],)
    arrays, config = build_cnn(spec["channels"], spec["classes"], rng,
                               bn=spec.get("bn", True))
    convs = len(spec["channels"]) - 1
    side = 5 + 2 * convs
    return arrays, config, (spec["channels"][0], side, side)


def make(name: str, spec: dict, variant: str, out_dir: str, seed: int,
         dtype: str, sigma: float) -> dict:
    rng = np.random.default_rng(seed)
    base, config, in_shape = build(spec, rng)
    topo = topology_of(base, config)

    if variant == "unrelated":
        target = build(spec, np.random.default_rng(seed + 9973))[0]
        truth = None
    else:
        if variant == "finetune":
            target, truth = dict(base), {}
        else:
            target, truth = permute(base, topo, rng)
        if variant in ("noisy", "finetune"):
            target = jitter(target, sigma, rng)

    gap = None
    if variant == "permuted":
        gap = check_equivalent(base, target, spec["kind"], rng, in_shape)

    where = os.path.join(out_dir, f"{name}-{variant}")
    os.makedirs(where, exist_ok=True)
    write_safetensors(os.path.join(where, "base.safetensors"), base, dtype,
                      {"format": "pt"})
    write_safetensors(os.path.join(where, "target.safetensors"), target, dtype,
                      {"format": "pt"})
    with open(os.path.join(where, "config.json"), "w") as f:
        json.dump(config, f, separators=(",", ":"), sort_keys=True)

    meta = {
        "name": name, "variant": variant, "kind": spec["kind"], "seed": seed,
        "dtype": dtype, "noise": sigma if variant in ("noisy", "finetune") else 0.0,
        "input_shape": list(in_shape),
        "tensors": len(base),
        "bytes": sum(os.path.getsize(os.path.join(where, f))
                     for f in ("base.safetensors", "target.safetensors")),
        "groups": {g.id: g.size for g in topo.solvable_groups()},
        "permutations": (None if truth is None
                         else {k: [int(x) for x in v] for k, v in truth.items()}),
        "alignable": variant != "unrelated",
        "identity_expected": variant == "finetune",
        "function_gap": gap,
    }
    with open(os.path.join(where, "truth.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build alignment benchmark fixtures")
    ap.add_argument("--out", default="bench/fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="F16", choices=("F16", "BF16", "F32"))
    ap.add_argument("--noise", type=float, default=0.02,
                    help="gaussian sigma, as a fraction of each tensor's std")
    ap.add_argument("--only", action="append")
    ap.add_argument("--variants", action="append", choices=VARIANTS)
    ap.add_argument("--large", action="store_true", help="add the slow big one")
    args = ap.parse_args(argv)

    specs = dict(SPECS, **(LARGE if args.large else {}))
    if args.only:
        specs = {k: v for k, v in specs.items() if k in args.only}
        if not specs:
            print(f"no fixture named {args.only}", file=sys.stderr)
            return 2
    variants = args.variants or list(VARIANTS)

    os.makedirs(args.out, exist_ok=True)
    index = []
    for name, spec in specs.items():
        for variant in variants:
            meta = make(name, spec, variant, args.out, args.seed,
                        args.dtype, args.noise)
            index.append(meta)
            gap = ("" if meta["function_gap"] is None
                   else f"  function gap {meta['function_gap']:.2e}")
            print(f"  {name}-{variant:<10} {meta['tensors']:3d} tensors  "
                  f"{meta['bytes'] / 2**20:7.2f} MiB  "
                  f"{len(meta['groups'])} groups{gap}", file=sys.stderr)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
    print(f"{len(index)} fixtures in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())