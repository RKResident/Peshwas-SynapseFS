"""The loss barrier between two independently trained models, aligned or not.

Re-Basin's claim is about a BARRIER, not a distance. It says that after
permuting one network's units to match the other's, the straight line between
them in weight space stays low-loss -- the two solutions are the same basin
wearing different labels. It does not say the aligned weights are numerically
close, and those are different quantities: a basin can be wide and flat, so two
points in it can have near-identical loss all along the path between them and
still sit a full weight-norm apart.

That distinction is the whole question for a delta codec. Our MNIST pair aligns
to a residual of 1.21 against a break-even of 1.0, and extrapolating the
residual-vs-epoch curve gives an asymptote of 1.198 -- no amount of training
reaches it. This measures whether that is because alignment FAILED or because
alignment SUCCEEDED at something we cannot use.

Interpolates `theta(a) = (1-a) * A + a * B` and evaluates at each step, once
with B permuted to match A and once with B left alone. If the aligned curve is
flat where the naive one humps, the method works and the conclusion is that
linear mode connectivity does not imply small weight distance.

BATCHNORM MUST BE RECALIBRATED, and this is not a detail. `running_mean` and
`running_var` describe the activation statistics of the endpoints; an averaged
network has different activations, so its inherited statistics are simply
wrong, and an un-recalibrated midpoint reads as a huge barrier that is an
artifact of stale buffers rather than of the loss landscape. Re-Basin resets
and re-estimates them. Both are reported here, because the gap between them is
large enough to invert the conclusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, "tools")

from synapsefs.align import config_parser, solver
from synapsefs.align.IR import TensorRef                    # noqa: E402
from synapsefs.align.objective import apply_col_perm, apply_row_perm  # noqa: E402
from synapsefs.align.reader import SafetensorsReader        # noqa: E402
from synapsefs.safetensors_io import SafetensorsFile        # noqa: E402


def load_state(path: str) -> dict:
    """Every tensor as float32 (or int64 for the counters), by name."""
    out = {}
    with SafetensorsFile(path) as f:
        for name in f.names():
            s = f.spec(name)
            bits = f.rows(name, 0, s.num_rows).ravel()
            if s.dtype == "BF16":
                wide = bits.astype(np.uint32) << 16
                arr = wide.view(np.float32)
            elif s.dtype == "F16":
                arr = bits.view(np.float16).astype(np.float32)
            elif s.dtype == "F32":
                arr = bits.view(np.float32)
            else:
                arr = bits.astype(np.int64)
            out[name] = torch.from_numpy(
                np.ascontiguousarray(arr).reshape(s.shape if s.shape else ()))
    return out


def align(a_path: str, b_path: str, config: Path):
    """Per-tensor permutations taking B into A's unit ordering."""
    with SafetensorsReader(a_path) as ra:
        refs = {n: TensorRef(name=n, shape=tuple(r.shape), dtype=r.dtype)
                for n, r in ra.refs().items()}
    cfg = json.loads(config.read_text()) if config.is_file() else None
    topo = config_parser.parse(refs, cfg)
    res = solver.align_checkpoints(b_path, a_path, topo, max_sweeps=5)
    return res


def permute(state: dict, result) -> dict:
    """Apply the solved permutation to every tensor of B."""
    out = {}
    for name, t in state.items():
        ta = result.tensors.get(name)
        if ta is None or ta.identity or t.ndim == 0:
            out[name] = t
            continue
        rows = t.shape[0]
        flat = t.reshape(rows, -1).numpy()
        flat = apply_row_perm(flat, ta.pi_row)
        if ta.pi_col is not None and flat.shape[1] > 1:
            flat = apply_col_perm(flat, ta.pi_col, ta.col_block_size)
        out[name] = torch.from_numpy(np.ascontiguousarray(flat)).reshape(t.shape)
    return out


def lerp(a: dict, b: dict, alpha: float) -> dict:
    out = {}
    for k, va in a.items():
        vb = b[k]
        if va.is_floating_point():
            out[k] = (1.0 - alpha) * va + alpha * vb
        else:
            out[k] = va          # num_batches_tracked: not a thing to average
    return out


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    loss = correct = total = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss / total, correct / total * 100


@torch.no_grad()
def recalibrate_bn(model, loader, dev, batches: int):
    """Reset BatchNorm buffers and re-estimate them from training data."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None          # cumulative average over what we show it
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= batches:
            break
        model(x.to(dev))
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", choices=["mnist", "cifar100"], required=True)
    ap.add_argument("--data", default="tools/tools/data")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--bn-batches", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "mnist":
        from train_mnist_pair import MnistMLP
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(args.data, train=True, download=True, transform=tf)
        test = datasets.MNIST(args.data, train=False, download=True, transform=tf)
        cfg = json.loads(args.config.read_text())
        make = lambda: MnistMLP(widths=tuple(cfg["widths"]))
    else:
        from train_benchmark_mlp import PlainMLP
        norm = transforms.Normalize((0.5071, 0.4865, 0.4409),
                                    (0.2673, 0.2564, 0.2762))
        tf = transforms.Compose([transforms.ToTensor(), norm])
        train = datasets.CIFAR100(args.data, train=True, download=True, transform=tf)
        test = datasets.CIFAR100(args.data, train=False, download=True, transform=tf)
        cfg = json.loads(args.config.read_text())
        make = lambda: PlainMLP(hidden=tuple(cfg["widths"]))

    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2)
    test_dl = DataLoader(test, batch_size=args.batch * 2, num_workers=2)

    A = load_state(args.a)
    B = load_state(args.b)
    res = align(args.a, args.b, args.config)
    print(f"alignment: {res.summary()}")
    s = res.residual_summary()
    print(f"           {s.line()}\n")
    B_aligned = permute(B, res)

    model = make().to(dev)
    alphas = np.linspace(0.0, 1.0, args.steps)

    for recal in (False, True):
        tag = "BN recalibrated" if recal else "BN inherited (stale)"
        print(f"  {tag}")
        print(f"  {'alpha':>6} {'naive loss':>11} {'naive acc':>10} "
              f"{'aligned loss':>13} {'aligned acc':>12}")
        curves = {}
        for label, Bx in (("naive", B), ("aligned", B_aligned)):
            row = []
            for a in alphas:
                model.load_state_dict({k: v.to(dev) for k, v in lerp(A, Bx, a).items()})
                if recal:
                    recalibrate_bn(model, train_dl, dev, args.bn_batches)
                row.append(evaluate(model, test_dl, dev))
            curves[label] = row
        for a, n, al in zip(alphas, curves["naive"], curves["aligned"]):
            print(f"  {a:6.2f} {n[0]:11.4f} {n[1]:9.2f}% {al[0]:13.4f} {al[1]:11.2f}%")
        for label in ("naive", "aligned"):
            lo, hi = curves[label][0][0], curves[label][-1][0]
            worst = max(curves[label], key=lambda r: r[0])[0]
            base = max(lo, hi)
            print(f"    {label:>8} barrier: worst loss {worst:.4f} vs "
                  f"endpoints {lo:.4f}/{hi:.4f}  ->  +{worst - base:.4f}")
        print()


if __name__ == "__main__":
    main()
