"""Train a ~40M-parameter fully-connected net on CIFAR-100, saving every epoch.

The second storage benchmark. Every compression conclusion in
`docs/ARCHITECTURE.md` §4.1 was measured on `train_benchmark.py`'s 92M CNN, and
a conclusion drawn from one architecture is a conclusion about that
architecture until it is checked on another. This model exists to check them.

What changes, from the codec's point of view:

**No spatial axis.** A conv kernel is `[out, in, 3, 3]`, so its nine taps sit
adjacent in memory and share an exponent range. A dense weight is a flat
`[out, in]` matrix with no such grouping. If the byte shuffle's advantage came
partly from spatial locality rather than from the exponent/mantissa split, it
shows up here as a worse ratio.

**Fewer, larger tensors.** The CNN spreads 176 MiB over 11 kernels; this
spreads 76 MiB over 4 matrices, the largest being 4096x4096 fp16 = 32 MiB.
Chunk boundaries therefore fall inside a tensor far more often.

**Different drift.** A dense layer's every weight sees every input feature, so
gradients are denser and less structured than a convolution's shared-weight
updates. Whether that raises or lowers the residual is the open question.

`BatchNorm1d` is used rather than `LayerNorm` deliberately: it keeps
`running_mean`, `running_var` and `num_batches_tracked` in the state dict, so
the per-class table `tools/experiments/layer_deltas.py` prints is directly
comparable between the two benchmarks. Switching to LayerNorm would remove the
buffers and make the two models' checkpoints structurally different for a
reason unrelated to the dense/conv question.

Accuracy will be poor -- a fully-connected net on raw pixels plateaus somewhere
around 25-30% top-1 on CIFAR-100, against the CNN's much higher figure. That is
expected and irrelevant: this is a benchmark for storage behaviour, and what
matters is that the weights move the way real training moves them.

Trains and saves only; committing is a separate step:

    python tools/train_benchmark_mlp.py
    for f in tools/benchmark-mlp/epoch*.safetensors; do
        synapsefs -C tools/benchmark-mlp commit "$f" -m "$(basename "$f" .safetensors)"
    done
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
from torchinfo import summary
from torchvision import datasets, transforms

# 3072 -> 4096 -> 4096 -> 2048 -> 1024 -> 100, which lands at 39.98M:
#   3072*4096 + 4096 = 12,587,008     4096*2048 + 2048 =  8,390,656
#   4096*4096 + 4096 = 16,781,312     2048*1024 + 1024 =  2,098,176
#   1024*100  +  100 =    102,500     BatchNorm1d scales/biases = 22,528
HIDDEN = (4096, 4096, 2048, 1024)


class PlainMLP(nn.Module):
    """Straight chain of linear -> bn -> relu, then a linear head.

    No residual connections, for the same reason `PlainCNN` has none:
    `align/config_parser.py` recovers the layer chain from shape divisibility
    and handles straight chains only. A branch would be wired into a
    plausible-looking but wrong graph.
    """

    def __init__(self, hidden=HIDDEN, num_classes=100, in_features=3 * 32 * 32):
        super().__init__()
        self.hidden = tuple(hidden)
        layers = []
        prev = in_features
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            prev = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x):
        return self.head(self.features(x.flatten(1)))


def fp16_state_dict(model: nn.Module) -> dict:
    """Cast floating-point tensors to fp16; leave integer buffers alone.

    `num_batches_tracked` is int64 and must stay that way -- casting it would
    silently corrupt BatchNorm on reload, and it is the reason the codec needs
    an integer dtype at all.
    """
    return {
        k: (v.half() if v.is_floating_point() else v).contiguous().cpu()
        for k, v in model.state_dict().items()
    }


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tools/benchmark-mlp",
                    help="where epochNN.safetensors and config.json are written")
    ap.add_argument("--data", default="data", help="dataset directory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed. Two lineages with different seeds are the "
                         "cross-basin experiment: same architecture, same data, "
                         "unrelated weights.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 2000 images with a 1/8-width model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    train = datasets.CIFAR100(args.data, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm]))
    test = datasets.CIFAR100(args.data, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), norm]))

    if args.smoke:
        train = torch.utils.data.Subset(train, range(2000))
        test = torch.utils.data.Subset(test, range(1000))
        args.epochs = min(args.epochs, 2)

    hidden = tuple(w // 8 for w in HIDDEN) if args.smoke else HIDDEN
    model = PlainMLP(hidden=hidden).to(dev)
    summary(model, input_size=(32, 3, 32, 32))
    n_params = sum(p.numel() for p in model.parameters())

    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=args.batch * 2,
                         num_workers=args.workers, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))

    # config.json is the only metadata the aligner may read. Never
    # ground_truth.json -- that is the answer key, for scoring only.
    (out / "config.json").write_text(json.dumps({
        "architecture": "PlainMLP", "widths": list(hidden),
        "num_classes": 100, "in_features": 3 * 32 * 32,
    }, indent=2))
    # config.json deliberately does NOT record the seed: it is the only file
    # the aligner may read, and a seed would let it distinguish lineages by
    # metadata rather than by weights. Two lineages must look identical to it.

    print(f"model      PlainMLP {hidden}   seed {args.seed}")
    print(f"parameters {n_params/1e6:.2f}M   fp16 checkpoint ~{human(n_params * 2)}")
    print(f"device     {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    print(f"data       CIFAR-100, {len(train)} train / {len(test)} test, batch {args.batch}")
    print(f"optimizer  Adam lr={args.lr}, no weight decay, no scheduler")
    print(f"writing    {out}/epochNN.safetensors "
          f"({human(n_params * 2 * args.epochs)} total)\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = total = 0
        # leave=False so each epoch's bar is erased once done and only the
        # one-line summary below survives -- a 20-line log rather than 20 bars.
        bar = tqdm(train_dl, desc=f"epoch {epoch:>2}/{args.epochs}", leave=False,
                   unit="batch")
        for x, y in bar:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            bar.set_postfix(loss=f"{loss_sum/total:.3f}",
                            acc=f"{correct/total*100:.1f}%", refresh=False)
        bar.close()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in tqdm(test_dl, desc="  eval", leave=False, unit="batch"):
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                    vc += (model(x).argmax(1) == y).sum().item()
                vt += y.size(0)

        train_s = time.time() - t0
        path = out / f"epoch{epoch:02d}.safetensors"
        t1 = time.time()
        save_file(fp16_state_dict(model), str(path))
        tqdm.write(
            f"epoch {epoch:>2}/{args.epochs}  loss {loss_sum/total:.3f}  "
            f"train {correct/total*100:5.2f}%  val {vc/vt*100:5.2f}%  "
            f"{train_s:5.1f}s  save {time.time()-t1:4.1f}s  -> {path.name}")

    print(f"\n{args.epochs} checkpoints in {out}/. To commit them:\n")
    print(f'  for f in {out}/epoch*.safetensors; do')
    print(f'      synapsefs -C {out} commit "$f" -m "$(basename "$f" .safetensors)"')
    print(f'  done')


if __name__ == "__main__":
    main()
