"""Train a ~92M-parameter CNN on CIFAR-100, saving a checkpoint every epoch.

The benchmark SynapseFS is measured against. `train_stl10.py` produces a 3.2M
model whose tensors all fit in a single 4 MiB chunk, which hides every
behaviour that only appears at scale: multi-chunk tensors, partial reads that
touch a subset of chunks, gather cost under a row permutation, and object
counts large enough for lookup structure to matter. Here the largest tensor is
1792x1344x3x3 fp16 = 41 MiB, spanning 11 chunks.

This script only trains and saves. Committing is a separate step, so the two
can be re-run independently:

    python tools/train_benchmark.py
    for f in tools/benchmark/epoch*.safetensors; do
        synapsefs -C tools/benchmark commit "$f" -m "$(basename "$f" .safetensors)"
    done

Three choices exist for the storage side rather than the accuracy side:

**Plain CNN, no residual connections.** `align/config_parser.py` recovers the
layer chain from shape divisibility and handles straight conv/linear/norm
chains only. A ResNet's shortcut convolutions would produce a branch it wires
into a plausible-looking but wrong graph. Skip connections would test the
aligner's *limits*; a straight chain tests the *pipeline*, which is what a
benchmark is for.

**Adam, constant lr, no weight decay, no scheduler.** Matches how the STL-10
fixtures were produced, so the two are comparable. It is also the hardest case
for compression: Adam's normalised update moves every weight by roughly `lr`
per step regardless of gradient magnitude, so consecutive checkpoints differ
almost everywhere and there is little to deduplicate.

**fp16 state dict.** What the PS evaluates. `num_batches_tracked` stays int64
through `.half()`, so every checkpoint here is genuinely mixed-dtype -- which
is the case a codec assuming one width per file gets wrong.
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
from torchinfo import summary

# 10 conv layers in 5 stages, one 2x2 pool per stage. Tuned to land just over
# 90M parameters -- see the module docstring for why size is the point.
WIDTHS = (224, 224, 448, 448, 896, 896, 1344, 1344, 1792, 1792)


class PlainCNN(nn.Module):
    """VGG-style straight chain: conv -> bn -> relu, pooling every 2 layers."""

    def __init__(self, widths=WIDTHS, num_classes=100, in_ch=3, spatial=32):
        super().__init__()
        self.widths = tuple(widths)
        layers = []
        prev = in_ch
        for i, w in enumerate(widths):
            layers += [nn.Conv2d(prev, w, 3, padding=1), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True)]
            if i % 2 == 1 and i < 8:              # 4 pools: 32 -> 16 -> 8 -> 4 -> 2
                layers.append(nn.MaxPool2d(2))
            prev = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev * (spatial // 16) ** 2, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


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
    ap.add_argument("--out", default="tools/benchmark",
                    help="where epochNN.safetensors and config.json are written")
    ap.add_argument("--data", default="data", help="dataset directory")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 2000 images with a 1/8-width model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    train = datasets.CIFAR100(args.data, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm]))
    test = datasets.CIFAR100(args.data, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), norm]))

    print(next(iter(train))[0].shape)

    if args.smoke:
        train = torch.utils.data.Subset(train, range(2000))
        test = torch.utils.data.Subset(test, range(1000))
        args.epochs = min(args.epochs, 2)

    widths = tuple(w // 8 for w in WIDTHS) if args.smoke else WIDTHS
    model = PlainCNN(widths=widths).to(dev).to(memory_format=torch.channels_last)
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
        "architecture": "PlainCNN", "widths": list(widths),
        "num_classes": 100, "kernel_size": 3,
    }, indent=2))

    print(f"model      PlainCNN {widths}")
    print(f"parameters {n_params/1e6:.2f}M   fp16 checkpoint ~{human(n_params * 2)}")
    print(f"device     {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    print(f"data       CIFAR-100, {len(train)} train / {len(test)} test, batch {args.batch}")
    print(f"optimizer  Adam lr={args.lr}, no weight decay, no scheduler")
    print(f"writing    {out}/epochNN.safetensors "
          f"({human(n_params * 2 * args.epochs)} total)\n", flush=True)

    summary(model, input_size=(32, 3, 32, 32))
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = total = 0
        # leave=False so each epoch's bar is erased once done and only the
        # one-line summary below survives -- a 25-line log rather than 25 bars.
        bar = tqdm(train_dl, desc=f"epoch {epoch:>2}/{args.epochs}", leave=False,
                   unit="batch")
        for x, y in bar:
            x = x.to(dev, non_blocking=True, memory_format=torch.channels_last)
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
                x = x.to(dev, non_blocking=True, memory_format=torch.channels_last)
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
