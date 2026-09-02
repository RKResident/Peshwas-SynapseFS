
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from align.align import THRESHOLD, align_checkpoints
from align.Error import AlignError
from align.match import gather_cols, gather_rows
from align.reader import SafetensorsReader
from align.topology import parse

VARIANTS = ("permuted", "noisy", "finetune", "unrelated")
NOISE = 0.02


# ---------------------------------------------------------------- checkpoints

def load(path):
    """Read a real checkpoint into memory. Only bench does this; the pipeline
    reads from the mapping a tensor at a time."""
    with SafetensorsReader(path) as r:
        shapes = r.shapes
        return {n: r.matrix(n).reshape(shapes[n] or (1,)) for n in shapes}


def synth(spec, rng):
    """'mlp:64,128,10' or 'cnn:3,16,32'. Returns (checkpoint, input shape)."""
    kind, _, raw = spec.partition(":")
    dims = [int(d) for d in raw.split(",") if d]
    if kind == "mlp":
        a = {}
        for i, (fan_in, fan_out) in enumerate(zip(dims, dims[1:]), 1):
            a[f"l{i}.weight"] = rng.standard_normal((fan_out, fan_in)) * np.sqrt(2 / fan_in)
            a[f"l{i}.bias"] = rng.standard_normal(fan_out) * 0.1
        shape = (dims[0],)
    elif kind == "cnn":
        a = {}
        for i, (cin, cout) in enumerate(zip(dims, dims[1:]), 1):
            a[f"conv{i}.weight"] = rng.standard_normal((cout, cin, 3, 3)) * np.sqrt(2 / (cin * 9))
            a[f"conv{i}.bias"] = rng.standard_normal(cout) * 0.1
            a[f"bn{i}.weight"] = 1 + 0.1 * rng.standard_normal(cout)
            a[f"bn{i}.bias"] = 0.1 * rng.standard_normal(cout)
            a[f"bn{i}.running_mean"] = 0.1 * rng.standard_normal(cout)
            a[f"bn{i}.running_var"] = 1 + 0.1 * rng.standard_normal(cout) ** 2
        a["fc.weight"] = rng.standard_normal((10, dims[-1] * 25)) * np.sqrt(2 / (dims[-1] * 25))
        a["fc.bias"] = rng.standard_normal(10) * 0.1
        a["bn1.num_batches_tracked"] = np.array(100.0)
        side = 5 + 2 * (len(dims) - 1)
        shape = (dims[0], side, side)
    else:
        raise ValueError(f"--make wants mlp:... or cnn:..., got '{spec}'")
    return {k: np.asarray(v, np.float32) for k, v in a.items()}, shape


def write(path, arrays):
    """A real .safetensors file: metadata first, keys sorted, header padded."""
    header, body, offset = {"__metadata__": {"format": "pt"}}, b"", 0
    for name in sorted(arrays):
        payload = np.asarray(arrays[name], np.float32).astype(np.float16).tobytes()
        header[name] = {"dtype": "F16", "shape": list(np.shape(arrays[name])),
                        "data_offsets": [offset, offset + len(payload)]}
        body += payload
        offset += len(payload)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-(8 + len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little") + blob + body)
    return path


# ---------------------------------------------------------------- the target

def permute(arrays, topo, rng):
    """Shuffle every solvable group. Returns (target, {group: permutation})."""
    truth = {g.id: rng.permutation(g.size) for g in topo.solvable()}
    out = {k: np.array(v, copy=True) for k, v in arrays.items()}
    for gid, p in truth.items():
        for name in topo.groups[gid].rows:
            out[name] = gather_rows(out[name], p)
        for name in topo.groups[gid].cols:
            shape = out[name].shape
            flat = out[name].reshape(shape[0], -1)
            out[name] = gather_cols(flat, p, topo.col_owner[name].block).reshape(shape)
    return out, truth


def jitter(arrays, rng, sigma=NOISE):
    """Noise relative to each tensor's own spread, so it means the same thing on
    a 16-wide layer and a 1024-wide one."""
    return {k: (v + sigma * (np.std(v) or 1) * rng.standard_normal(np.shape(v))
                ).astype(np.float32) for k, v in arrays.items()}


def scramble(arrays, rng):
    """Fresh weights of the same shapes: an unrelated checkpoint."""
    return {k: (rng.standard_normal(np.shape(v)) * (np.std(v) or 1)).astype(np.float32)
            for k, v in arrays.items()}


# ------------------------------------------------- the check that skips the IR

def forward(arrays, x):
    """A plain numpy forward pass. Deliberately never consults the Topology."""
    convs = sum(1 for k in arrays if k.startswith("conv") and k.endswith(".weight"))
    if not convs:
        depth = sum(1 for k in arrays if k.endswith(".weight"))
        for i in range(1, depth + 1):
            x = arrays[f"l{i}.weight"] @ x + arrays[f"l{i}.bias"]
            x = np.maximum(x, 0) if i < depth else x
        return x

    for i in range(1, convs + 1):
        w, b = arrays[f"conv{i}.weight"], arrays[f"conv{i}.bias"]
        out_ch, _, kh, kw = w.shape
        oh, ow = x.shape[1] - kh + 1, x.shape[2] - kw + 1
        y = np.zeros((out_ch, oh, ow), np.float32)
        for r in range(oh):
            for c in range(ow):
                y[:, r, c] = w.reshape(out_ch, -1) @ x[:, r:r + kh, c:c + kw].reshape(-1)
        x = y + b[:, None, None]
        m, v = arrays[f"bn{i}.running_mean"], arrays[f"bn{i}.running_var"]
        x = (x - m[:, None, None]) / np.sqrt(v[:, None, None] + 1e-5)
        x = x * arrays[f"bn{i}.weight"][:, None, None] + arrays[f"bn{i}.bias"][:, None, None]
        x = np.maximum(x, 0)
    return arrays["fc.weight"] @ x.reshape(-1) + arrays["fc.bias"]


def function_gap(base, target, shape, rng):
    """How differently the two checkpoints behave. Must be ~0 after a shuffle."""
    x = rng.standard_normal(shape).astype(np.float32)
    a, b = forward(base, x), forward(target, x)
    return float(np.abs(a - b).max() / (np.abs(a).max() or 1))


# ---------------------------------------------------------------- one fixture

def make(label, base, shape, variant, out_dir, rng, order=None):
    topo = parse({k: np.shape(v) for k, v in base.items()}, order)

    if variant == "unrelated":
        target, truth = scramble(base, rng), None
    elif variant == "finetune":
        target, truth = jitter(base, rng), {}
    else:
        target, truth = permute(base, topo, rng)
        if variant == "noisy":
            target = jitter(target, rng)

    gap = None
    if variant == "permuted" and shape is not None:
        gap = function_gap(base, target, shape, rng)
        if gap > 2e-2:
            raise AssertionError(f"{label}: the shuffled checkpoint computes a "
                                 f"different function (gap {gap:.3f})")

    where = os.path.join(out_dir, f"{label}-{variant}")
    os.makedirs(where, exist_ok=True)
    paths = (write(os.path.join(where, "base.safetensors"), base),
             write(os.path.join(where, "target.safetensors"), target))
    if truth:
        with open(os.path.join(where, "truth.json"), "w") as f:
            json.dump({g: p.tolist() for g, p in truth.items()}, f, indent=1)
    return {"label": f"{label}-{variant}", "variant": variant, "paths": paths,
            "truth": truth, "gap": gap, "order": order}


# ---------------------------------------------------------------- scoring

def recovery(result, base_path, truth, order=None):
    """Fraction of units placed on their ground-truth partner, size-weighted."""
    with SafetensorsReader(base_path) as r:
        topo = parse(r.shapes, order)
    hits = total = 0
    for group in topo.solvable():
        want = np.asarray(truth.get(group.id, range(group.size)))
        got = result[group.rows[0]].pi_row
        if got is None:
            got = np.arange(group.size)
        hits += int((got == want).sum())
        total += group.size
    return hits / total if total else None


def score(fixture):
    base, target = fixture["paths"]
    order = fixture["order"]
    result = align_checkpoints(base, target, order=order)
    start = time.perf_counter()
    align_checkpoints(base, target, order=order)
    seconds = time.perf_counter() - start

    found = (None if fixture["truth"] is None
             else recovery(result, base, fixture["truth"], order))
    expected = {"permuted": found == 1.0, "noisy": found == 1.0,
                "finetune": result.identity,
                "unrelated": bool(result.not_alignable)}[fixture["variant"]]
    posts = [a.residual_post for a in result.tensors.values()]
    return {"label": fixture["label"], "groups": result.groups,
            "sweeps": result.sweeps, "recovered": found,
            "pre": float(np.mean([a.residual_pre for a in result.tensors.values()])),
            "post": float(np.mean(posts)), "worst": float(np.max(posts)),
            "alignable": fixture["variant"] != "unrelated",
            "seconds": seconds, "ok": expected}


# ---------------------------------------------------------------- output

def show(rows):
    head = (f"{'fixture':<26}{'grp':>5}{'swp':>5}{'recovered':>11}"
            f"{'residual pre':>14}{'post':>9}{'seconds':>9}   ")
    print(head)
    print("-" * (len(head) + 4))
    for r in rows:
        rec = "    --" if r["recovered"] is None else f"{r['recovered'] * 100:5.1f}%"
        print(f"{r['label']:<26}{r['groups']:>5}{r['sweeps']:>5}{rec:>11}"
              f"{r['pre']:>14.3f}{r['post']:>9.3f}{r['seconds']:>9.3f}   "
              f"{'ok' if r['ok'] else 'FAIL'}")


def calibrate(rows):
    good = [r["worst"] for r in rows if r["alignable"]]
    bad = [r["worst"] for r in rows if not r["alignable"]]
    if not good or not bad:
        return "threshold: nothing to calibrate against"
    inside = max(good) < THRESHOLD <= min(bad)
    return (f"threshold: alignable pairs stay under {max(good):.3f}, unrelated "
            f"never fall below {min(bad):.3f}\n"
            f"           {THRESHOLD} sits {'inside' if inside else 'OUTSIDE'} that gap")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoints", nargs="*",
                    help="real .safetensors files to shuffle and re-align")
    ap.add_argument("--make", action="append", metavar="SPEC",
                    help="synthesise one instead, e.g. cnn:3,16,32")
    ap.add_argument("--out", default="fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", metavar="STEMS",
                    help="comma-separated layer order, when the tensor names do "
                         "not sort into execution order")
    args = ap.parse_args(argv)
    order = args.order.split(",") if args.order else None

    rng = np.random.default_rng(args.seed)
    inputs = []
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"no such file: {path}", file=sys.stderr)
            return 2
        inputs.append((os.path.basename(path).removesuffix(".safetensors"),
                       load(path), None))
    for spec in args.make or ([] if args.checkpoints else ["cnn:1,4,8"]):
        try:
            arrays, shape = synth(spec, rng)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        inputs.append((spec.replace(":", "-"), arrays, shape))

    rows = []
    for label, arrays, shape in inputs:
        if shape is None:
            print(f"  {label}: supplied checkpoint, so the forward-pass check "
                  f"is skipped", file=sys.stderr)
        for variant in VARIANTS:
            fixture = make(label, arrays, shape, variant, args.out, rng, order)
            if fixture["gap"] is not None:
                print(f"  {fixture['label']}: shuffled checkpoint computes the "
                      f"same function (gap {fixture['gap']:.1e})", file=sys.stderr)
            rows.append(score(fixture))

    print()
    show(rows)
    print()
    print(calibrate(rows))
    print(f"\nfixtures in {args.out}/")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)