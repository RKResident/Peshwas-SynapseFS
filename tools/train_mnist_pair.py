"""Two differently-seeded MNIST classifiers, trained to convergence, for
testing whether alignment can relate independent lineages.

Our first attempt at this failed: two 40M CIFAR-100 MLPs from different inits
aligned from 1.4103 to 1.3204, leaving 0.1% of weights below the storage
threshold. Break-even is 1.0, so that is not close. Git Re-Basin reports
success on the same question, and the difference in setup is what this script
exists to remove.

Two things differ, and only one of them is the one usually blamed.

**Training length.** The CIFAR-100 pair ran 20 epochs to ~33% validation
accuracy -- nowhere near converged. Re-Basin's models are trained out. Weights
that are still moving fast have not settled into a basin yet, so there may
simply be no stable correspondence to find. MNIST converges in single-digit
epochs, so 100 epochs is far past the point where this excuse applies.

**Width.** This is the one that gets forgotten, and it may matter more.
Re-Basin's central empirical claim is that the loss barrier between aligned
solutions shrinks as models get WIDER, and that narrow models retain large
barriers no matter how long they train. So "very lightweight" pulls against
the thing being tested. `--widths` therefore defaults to 512x512 rather than
something truly tiny -- on MNIST that still trains 100 epochs in minutes -- and
the flag exists so the width sweep can be run deliberately rather than
stumbled into.

**Checkpoints are saved on a log schedule, not just at the end.** The claim
under test is that alignment improves with training. One final checkpoint
cannot show that; a curve of aligned residual against epoch can, and it costs
nothing to save 11 checkpoints instead of 1.

Trains both seeds in one run so the recipe is identical by construction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms

SAVE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

#: Roughly log-spaced, so the early epochs -- where the weights move most and
#: any correspondence would be forming -- are sampled densely.
DEFAULT_SCHEDULE = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100)


class MnistMLP(nn.Module):
    """linear -> bn -> relu chain, then a linear head. No residual connections:
    `align/config_parser.py` recovers the layer chain from shape divisibility
    and handles straight chains only."""

    def __init__(self, widths=(512, 512), num_classes=10, in_features=28 * 28):
        super().__init__()
        layers = []
        prev = in_features
        for w in widths:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            prev = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        return self.head(self.features(x.flatten(1)))


def cast_state_dict(model: nn.Module, dtype: torch.dtype) -> dict:
    """Floating-point tensors to `dtype`; integer buffers untouched, because
    `num_batches_tracked` is int64 and casting it corrupts BatchNorm."""
    return {
        k: (v.to(dtype) if v.is_floating_point() else v).contiguous().cpu()
        for k, v in model.state_dict().items()
    }


def train_one(seed: int, out: Path, args, train_dl, test_dl, dev) -> float:
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    model = MnistMLP(widths=tuple(args.widths)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # config.json is the only metadata the aligner may read, and it must NOT
    # record the seed -- the two lineages have to be indistinguishable to it by
    # anything except their weights.
    (out / "config.json").write_text(json.dumps({
        "architecture": "MnistMLP", "widths": list(args.widths),
        "num_classes": 10, "in_features": 28 * 28,
    }, indent=2))

    print(f"\nseed {seed}: MnistMLP {tuple(args.widths)}  {n_params/1e6:.2f}M params "
          f"-> {out}", flush=True)
    schedule = set(args.schedule) | {args.epochs}
    val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = total = 0
        bar = tqdm(train_dl, desc=f"  seed {seed} epoch {epoch:>3}/{args.epochs}",
                   leave=False, unit="batch")
        for x, y in bar:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        bar.close()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                vc += (model(x).argmax(1) == y).sum().item()
                vt += y.size(0)
        val = vc / vt * 100

        if epoch in schedule:
            save_file(cast_state_dict(model, SAVE_DTYPES[args.dtype]),
                      str(out / f"epoch{epoch:03d}.safetensors"))
        tqdm.write(f"  seed {seed} epoch {epoch:>3}/{args.epochs}  "
                   f"loss {loss_sum/total:.4f}  train {correct/total*100:5.2f}%  "
                   f"val {val:5.2f}%  {time.time()-t0:4.1f}s"
                   + ("  saved" if epoch in schedule else ""))
    return val


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tools/mnist-pair")
    ap.add_argument("--data", default="tools/data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="MNIST converges in single digits at 1e-3; the point "
                         "of 100 epochs is to be far past that, not to crawl")
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 512],
                    help="hidden widths. Re-Basin's barriers shrink with width, "
                         "so this is the knob to sweep if alignment fails")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dtype", choices=sorted(SAVE_DTYPES), default="bf16")
    ap.add_argument("--schedule", type=int, nargs="+", default=list(DEFAULT_SCHEDULE),
                    help="epochs to checkpoint, for the residual-vs-epoch curve")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    test = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=args.batch * 4, num_workers=args.workers,
                         pin_memory=True)

    root = Path(args.out)
    print(f"device {dev}   MNIST {len(train)} train / {len(test)} test   "
          f"{args.epochs} epochs   dtype {args.dtype}")
    finals = {}
    for seed in args.seeds:
        finals[seed] = train_one(seed, root / f"seed{seed}", args,
                                 train_dl, test_dl, dev)

    a, b = (root / f"seed{s}" for s in args.seeds[:2])
    last = f"epoch{args.epochs:03d}.safetensors"
    print("\nfinal validation accuracy: "
          + "  ".join(f"seed {s} {v:.2f}%" for s, v in finals.items()))
    print(f"\nDoes alignment improve as the models converge?\n")
    print(f"  PYTHONPATH=. python tools/experiments/cross_basin.py \\")
    print(f"      --config {a}/config.json --sweeps 5 \\")
    for e in args.schedule:
        print(f"      ep{e:03d}={b}/epoch{e:03d}.safetensors:{a}/epoch{e:03d}.safetensors \\")
    print(f"\n  A permuted positive control belongs in that list too -- without one,")
    print(f"  a flat curve cannot be told apart from a broken measurement.\n")
    print(f"Then the merge, as two lineages in one repo:\n")
    print(f"  synapsefs -C {root} init")
    print(f"  synapsefs -C {root} commit {a}/{last} -m 'seed {args.seeds[0]}'")
    print(f"  synapsefs -C {root} branch other")
    print(f"  synapsefs -C {root} checkout other")
    print(f"  synapsefs -C {root} commit {b}/{last} -m 'seed {args.seeds[1]}'")
    print(f"  synapsefs -C {root} checkout main")
    print(f"  synapsefs -C {root} merge other --average")


if __name__ == "__main__":
    main()
