#!/usr/bin/env python
"""Train a small CNN on STL-10 and save one checkpoint per epoch.

Produces exactly what `synapsefs commit` wants to eat: a directory of
`.safetensors` files that differ by one epoch of training, plus a single
`config.json` beside them (commit looks for `<checkpoint>.parent/config.json`,
so one file serves every epoch).

    PYTHONPATH= .venv/bin/python tools/train_stl10.py
    PYTHONPATH= .venv/bin/python tools/train_stl10.py --smoke   # no download, random data

Then, from the checkpoint directory:

    synapsefs init .
    synapsefs commit epoch01.safetensors -m "epoch 1"
    ...

Two choices here are about SynapseFS rather than about training:

* **Only floating-point tensors are cast to fp16.** The PS grades checkpoints
  "in fp16/bf16 precision (not fp32)", but `.half()` on a whole state_dict
  would also mangle BatchNorm's `num_batches_tracked`, which is an int64
  scalar. Leaving integer buffers alone is both correct and the case the codec
  specifically supports (F16 weights + an I64 0-d buffer).
* **The default is ~3.2M parameters** (widths 96/192/448), which is the point:
  a checkpoint around 6 MB rather than 570 KB, so residual ratios and pack
  sizes mean something. Override with `--widths`.
* **The architecture is conv / batch-norm / linear**, matching the PS's
  Architecture Scope. BatchNorm matters beyond realism: its `running_mean` and
  `running_var` shift every epoch while the conv weights move only slightly,
  which is exactly the mix of changed and near-frozen tensors that makes a
  residual codec worth testing.

Note the STL-10 archive is ~2.5 GB (it bundles 100k unlabeled images even
though only the 5k labeled train split is used here). Use `--smoke` to check
the pipeline end to end without it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torchinfo import summary
from tqdm import tqdm
import safetensors

IMAGE_SIZE = 96
NUM_CLASSES = 10
# STL-10's published channel statistics.
MEAN = (0.4467, 0.4398, 0.4066)
STD = (0.2603, 0.2566, 0.2713)


class ConvBlock(nn.Module):
    """conv -> bn -> relu, twice, then halve the resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.max_pool2d(x, 2)


class SmallCNN(nn.Module):
    def __init__(self, widths=(96, 192, 448), num_classes: int = NUM_CLASSES):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.blocks = nn.Sequential(
            *[ConvBlock(chans[i], chans[i + 1]) for i in range(len(widths))]
        )
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def fp16_state_dict(model: nn.Module) -> dict:
    """State dict with floats narrowed to fp16 and integer buffers left alone.

    `model.half()` would turn `num_batches_tracked` (int64) into fp16, which is
    both wrong and would hide the mixed-dtype case the codec handles.
    """
    out = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.half()
        out[name] = tensor.contiguous().clone()
    return out


def write_config(model: nn.Module, path: Path, widths) -> None:
    """Minimal topology description, in the shape an alignment engine wants:
    the layer graph, not the weights.

    The PS says evaluated checkpoints come with "standard config.json files
    defining the graph topology", and that the alignment engine must work from
    those. It must never read ground_truth.json -- that is the answer key.
    """
    layers = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind == "Conv2d":
            layers.append({"name": name, "type": "conv2d",
                           "in_channels": module.in_channels,
                           "out_channels": module.out_channels,
                           "kernel_size": list(module.kernel_size)})
        elif kind == "BatchNorm2d":
            layers.append({"name": name, "type": "batchnorm2d",
                           "num_features": module.num_features})
        elif kind == "Linear":
            layers.append({"name": name, "type": "linear",
                           "in_features": module.in_features,
                           "out_features": module.out_features})
    path.write_text(json.dumps(
        {"architecture": "SmallCNN", "dataset": "stl10",
         "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
         "widths": list(widths), "num_classes": NUM_CLASSES,
         "layers": layers},
        indent=2,
    ))


def build_loaders(args):
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.STL10(args.data, split="train", download=True, transform=train_tf)
    test = datasets.STL10(args.data, split="test", download=True, transform=eval_tf)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    return (
        torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True,
                                    drop_last=True, **common),
        torch.utils.data.DataLoader(test, batch_size=args.batch * 2, **common),
    )


def build_smoke_loaders(args):
    """Random tensors shaped like STL-10, so the whole pipeline can be checked
    without the 2.5 GB download."""
    def fake(n):
        return torch.utils.data.TensorDataset(
            torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.randint(0, NUM_CLASSES, (n,)),
        )
    return (
        torch.utils.data.DataLoader(fake(256), batch_size=args.batch, shuffle=True, drop_last=True),
        torch.utils.data.DataLoader(fake(128), batch_size=args.batch * 2),
    )


def latest_checkpoint(out_dir: Path) -> Path:
    found = sorted(out_dir.glob("epoch*.safetensors"))
    if not found:
        raise SystemExit(f"--resume: no epoch*.safetensors in {out_dir}")
    return found[-1]


def load_for_resume(model, optimizer, args, out_dir: Path) -> int:
    """Restore weights (and optimizer moments if available); return the epoch
    to start from.

    The `.safetensors` files hold fp16 weights, because that is what gets
    committed; the live model is fp32, so floats are widened on the way back
    in. Integer buffers (`num_batches_tracked`) are left alone.

    Optimizer state lives in a separate `training_state.pt`, deliberately not
    in the checkpoint: AdamW's moments are training scaffolding, not model
    weights, and putting them in the `.safetensors` would inflate every commit
    with tensors nobody ever reconstructs. Resuming without them resets
    momentum, which shows up as an unusually large weight jump in the first
    resumed epoch -- and so an unusually poor residual ratio for that commit.
    """
    path = latest_checkpoint(out_dir) if args.resume == "auto" else Path(args.resume)
    state = load_file(str(path))
    model.load_state_dict(
        {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    )

    start = int(path.stem.replace("epoch", "")) + 1
    training_state = out_dir / "training_state.pt"
    if training_state.is_file():
        blob = torch.load(training_state, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(blob["optimizer"])
        start = blob["epoch"] + 1
        print(f"resuming from {path.name} + optimizer state -> epoch {start}")
    else:
        print(f"resuming from {path.name} (no optimizer state; momentum resets) "
              f"-> epoch {start}")
    return start


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data", default="./data", help="dataset download directory")
    p.add_argument("--out", default="./checkpoints", help="where the .safetensors go")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--widths", type=int, nargs="+", default=[96, 192, 448],
                   help="channels per conv block; the default is ~3.2M params, "
                        "which is the point -- a realistic checkpoint size, not a toy")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   metavar="CHECKPOINT",
                   help="continue training. Bare --resume picks the highest-numbered "
                        "epochNN.safetensors in --out; or name one explicitly. "
                        "--epochs is the TOTAL, so --resume --epochs 20 after a "
                        "7-epoch run trains epochs 8..20.")
    p.add_argument("--smoke", action="store_true",
                   help="random data instead of STL-10, no download -- checks the pipeline only")
    args = p.parse_args()

    if args.smoke:
        args.workers = 0

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    widths = tuple(args.widths)
    model = SmallCNN(widths).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"widths {widths} -> {params:,} parameters")
    summary(model, input_size=(args.batch, 3, IMAGE_SIZE, IMAGE_SIZE),
            col_names=("input_size", "output_size", "num_params"), depth=3)

    train_loader, test_loader = (
        build_smoke_loaders(args) if args.smoke else build_loaders(args)
    )

    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(model, out_dir / "config.json", widths)

    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_for_resume(model, optimizer, args, out_dir)
        if start_epoch > args.epochs:
            raise SystemExit(
                f"--epochs is the total: {args.epochs} already reached "
                f"(next would be {start_epoch}). Pass a larger --epochs."
            )
        # Rebuilt against the new total rather than restored, so the cosine
        # curve spans the whole extended run instead of the original one.
        # for _ in range(start_epoch - 1):
        #     scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
            bar.set_postfix(loss=f"{running / steps:.3f}")
        # scheduler.step()

        accuracy = evaluate(model, test_loader, device)
        path = out_dir / f"epoch{epoch:02d}.safetensors"
        save_file(fp16_state_dict(model), str(path))
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict()},
                   out_dir / "training_state.pt")
        print(f"epoch {epoch}/{args.epochs}  loss {running / max(steps, 1):.4f}  "
              f"test acc {accuracy:5.2f}%  -> {path.name} ({path.stat().st_size / 1024:.0f} KiB)")

    print(f"\nepochs {start_epoch}..{args.epochs} in {out_dir.resolve()}")
    print("next:  cd", out_dir, "&& synapsefs init . && "
          "synapsefs commit epoch01.safetensors -m 'epoch 1'")


if __name__ == "__main__":
    main()
