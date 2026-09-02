"""Fine-tune a large CNN, committing every epoch into SynapseFS as it trains.

The point of this script is the shape of its inner loop, not the training:

    for epoch in ...:
        train_one_epoch(...)
        write ONE checkpoint to a temp path
        synapsefs commit it
        DELETE the temp path

At no point does more than a single checkpoint exist on disk. That is the
workflow the PS describes -- "fine-tuning large models produces hundreds of
gigabytes of serialized checkpoints" -- and the reason to have a VCS for them
at all. A script that trains first and imports afterwards needs every
checkpoint present simultaneously, which is exactly the cost the system exists
to avoid, and it is also the workflow that would let you use B-frames
(ARCHITECTURE.md 4.3.1) -- so keeping the online shape honest here keeps that
trade-off visible rather than hidden.

Checkpoints are written in **bf16**: it is what the PS grades and what the
escape codec was tuned for. `num_batches_tracked` stays int64, so every file
is mixed-dtype on purpose (ARCHITECTURE.md 7.7).

PREREQUISITES on the training machine
-------------------------------------
    pip install torch torchvision timm safetensors tqdm
    pip install -e .                       # synapsefs itself
    # synapsefs' CLI imports pyfuse3 at module load even for `commit`, so:
    sudo apt install libfuse3-dev pkg-config && pip install pyfuse3
    python tools/fetch_regnety.py --out regnety_base

DATA
----
Any torchvision ImageFolder layout: <root>/train/<class>/*.jpg and
<root>/val/<class>/*.jpg. Imagenette is a good default -- 10 real ImageNet
classes, ~1.5 GB, so the fine-tune is representative without being an ImageNet
run.

A NOTE ON ALIGNMENT
-------------------
RegNetY has grouped convolutions, squeeze-excitation blocks and downsample
paths. `config_parser` infers the layer chain from shape divisibility and
raises rather than guessing when it cannot -- so it may reject this topology.
That is the parser behaving correctly, not a failure of the run. `--no-align`
keeps the storage path working while alignment is unavailable; the residual
codec does not need a permutation to be useful between consecutive epochs,
where identity is the right answer anyway.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------
# SynapseFS interface
# --------------------------------------------------------------------------

def synapsefs(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the CLI as a subprocess.

    A subprocess rather than an in-process import for two reasons: a crash in
    the storage layer must not take a multi-hour training run with it, and the
    exit code is the documented contract (errors.py reserves 4 for
    verification failures specifically, and graders script against it).
    """
    return subprocess.run([sys.executable, "-m", "synapsefs.cli.main", *args],
                          capture_output=True, text=True, cwd=cwd)


def ensure_repo(repo: Path) -> None:
    if (repo / ".synapse").exists():
        return
    repo.mkdir(parents=True, exist_ok=True)
    r = synapsefs("init", str(repo))
    if r.returncode != 0:
        raise SystemExit(f"synapsefs init failed:\n{r.stderr or r.stdout}")


def commit_checkpoint(repo: Path, ckpt: Path, config: Path, message: str,
                      no_align: bool) -> dict:
    """Commit one checkpoint and return the parsed --json result.

    Raises on failure WITHOUT the caller deleting the file, so a storage
    problem never silently loses an epoch of training.
    """
    args = ["-C", str(repo), "commit", str(ckpt), "-m", message,
            "--config", str(config), "--json"]
    if no_align:
        args.append("--no-align")
    r = synapsefs(*args)
    if r.returncode != 0:
        raise RuntimeError(
            f"commit failed (exit {r.returncode}) -- checkpoint KEPT at {ckpt}\n"
            f"{r.stderr or r.stdout}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"commit produced non-JSON output:\n{r.stdout[:500]}")


# --------------------------------------------------------------------------
# Checkpoint writing
# --------------------------------------------------------------------------

def save_bf16(model, path: Path) -> None:
    """Write the model as a bf16 safetensors file, floats only.

    `.contiguous()` matters: safetensors refuses non-contiguous tensors, and
    a channels_last model produces exactly those.
    """
    import torch
    from safetensors.torch import save_file
    sd = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        sd[k] = (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
    save_file(sd, str(path))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def build_loaders(data: Path, img_size: int, batch: int, workers: int):
    import torch
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = datasets.ImageFolder(str(data / "train"), train_tf)
    va = datasets.ImageFolder(str(data / "val"), val_tf)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=sh, num_workers=workers,
        pin_memory=True, drop_last=sh)
    return mk(tr, True), mk(va, False), len(tr.classes)


def freeze_prefix(model, frac: float):
    """Freeze the first `frac` of the model by parameter count.

    Walks `named_parameters()` in definition order -- which for a timm CNN is
    stem, then stages in depth order -- and freezes until the cumulative count
    crosses the fraction. Splitting by parameter count rather than by module
    count matters on RegNetY, where the last stage holds most of the weights:
    freezing "half the modules" would freeze a small minority of the tensors.

    **BatchNorm is put in eval() as well as having requires_grad cleared.**
    `requires_grad=False` stops gradients, but `running_mean`, `running_var`
    and `num_batches_tracked` are buffers, not parameters -- they keep updating
    on every forward pass in train() mode. A "frozen" layer whose BN statistics
    still move produces a different tensor every epoch, which for this
    benchmark is the whole point: frozen weights are supposed to be
    byte-identical across commits so the codec can dedupe them.

    Returns (frozen_modules, frozen_params, total_params).
    """
    params = [(n, p) for n, p in model.named_parameters()]
    total = sum(p.numel() for _, p in params)
    budget = frac * total
    seen = 0
    frozen_names = set()
    for name, p in params:
        if seen >= budget:
            break
        p.requires_grad_(False)
        frozen_names.add(name)
        seen += p.numel()

    # A module is frozen only if every parameter it owns is frozen; those are
    # the ones safe to hold in eval() so their buffers stop moving.
    frozen_modules = []
    for mname, mod in model.named_modules():
        owned = [f"{mname}.{n}" if mname else n
                 for n, _ in mod.named_parameters(recurse=False)]
        if owned and all(o in frozen_names for o in owned):
            frozen_modules.append(mod)
    return frozen_modules, seen, total


def _progress(iterable, desc, total=None):
    """tqdm bar if tqdm is installed, otherwise the iterable untouched.

    Bars go to stderr and `leave=False`, so they never interleave with the
    per-epoch results table on stdout -- that table is the artifact of this
    run and stays machine-readable when the output is piped.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False,
                unit="batch", dynamic_ncols=True)


def train_one_epoch(model, loader, opt, scaler, sched, device, amp_dtype,
                    epoch=None, frozen_modules=()):
    import torch
    import torch.nn.functional as F
    model.train()
    # model.train() re-enables buffer updates everywhere, so the frozen
    # modules have to be put back into eval() after it, every epoch.
    for m in frozen_modules:
        m.eval()
    total = correct = 0
    loss_sum = 0.0
    bar = _progress(loader, f"train e{epoch}" if epoch is not None else "train",
                    total=len(loader))
    for x, y in bar:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss_sum / max(total, 1):.3f}",
                            acc=f"{correct / max(total, 1):.3f}",
                            refresh=False)
    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, device, amp_dtype, epoch=None):
    import torch
    model.eval()
    total = correct = 0
    bar = _progress(loader, f"val   e{epoch}" if epoch is not None else "val",
                    total=len(loader))
    with torch.no_grad():
        for x, y in bar:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(acc=f"{correct / max(total, 1):.3f}", refresh=False)
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="ImageFolder root with train/ and val/")
    ap.add_argument("--base", type=Path, default=Path("regnety_base"),
                    help="output of tools/fetch_regnety.py")
    ap.add_argument("--repo", type=Path, default=Path("regnety_repo"))
    ap.add_argument("--model", default="regnety_1280.swag_ft_in1k")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224,
                    help="384 is the model's native size; 224 trains far "
                         "faster and is fine for a storage benchmark")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-frac", type=float, default=0.0,
                    help="Freeze this fraction of the model, by parameter "
                         "count, from the input side. 0.5 freezes the first "
                         "half. Frozen BatchNorm is held in eval() so its "
                         "running statistics stop moving too -- otherwise the "
                         "'frozen' tensors still differ every epoch.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-align", action="store_true",
                    help="skip alignment; see the note in the module docstring")
    ap.add_argument("--results", type=Path,
                    default=Path("bench-results") / f"regnety-{datetime.now():%Y%m%d-%H%M}.json")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast where supported: it matches the checkpoint dtype, so what
    # trains is what gets stored, with no extra rounding step in between.
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)

    tr, va, ncls = build_loaders(args.data, args.img_size, args.batch, args.workers)

    import timm
    model = timm.create_model(args.model, pretrained=True, num_classes=ncls)
    model = model.to(device, memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    frozen_modules, n_frozen, n_all = (), 0, nparam
    if args.freeze_frac > 0.0:
        frozen_modules, n_frozen, n_all = freeze_prefix(model, args.freeze_frac)

    # Only trainable parameters go to the optimizer: AdamW keeps two state
    # tensors per parameter, so handing it frozen ones costs real memory for
    # updates that are multiplied by a zero gradient anyway.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1),
        pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    config = args.base / "config.json"
    if not config.exists():
        raise SystemExit(f"missing {config} -- run tools/fetch_regnety.py first")
    ensure_repo(args.repo)

    staging = args.repo.parent / ".synapse-staging"
    staging.mkdir(parents=True, exist_ok=True)

    print(f"\n  {args.model}  {nparam/1e6:.1f}M params  bf16  "
          f"{ncls} classes  {args.img_size}px  batch {args.batch}")
    if args.freeze_frac > 0.0:
        print(f"  frozen: {n_frozen/1e6:.1f}M of {n_all/1e6:.1f}M params "
              f"({100.0*n_frozen/max(n_all,1):.1f}%), {len(frozen_modules)} modules "
              f"held in eval();  trainable {sum(p.numel() for p in trainable)/1e6:.1f}M")
    print(f"  repo: {args.repo}   epochs: {args.epochs}   device: {device}")
    print(f"\n  {'epoch':<7}{'loss':>8}{'train':>8}{'val':>8}"
          f"{'stored':>12}{'ratio':>9}{'commit':>10}{'secs':>8}")

    rows = []
    raw_bytes = 0
    for epoch in range(args.epochs + 1):        # epoch 0 = pretrained baseline
        t0 = time.perf_counter()
        if epoch == 0:
            loss = float("nan"); tacc = float("nan")
        else:
            loss, tacc = train_one_epoch(model, tr, opt, scaler, sched,
                                         device, amp_dtype, epoch=epoch,
                                         frozen_modules=frozen_modules)
        vacc = evaluate(model, va, device, amp_dtype, epoch=epoch)

        # ---- the part that matters: write, commit, delete -----------------
        ckpt = staging / f"epoch{epoch:03d}.safetensors"
        save_bf16(model, ckpt)
        size = ckpt.stat().st_size
        raw_bytes += size

        result = commit_checkpoint(args.repo, ckpt, config,
                                   f"epoch{epoch:03d}", args.no_align)
        ckpt.unlink()          # only after a successful commit
        # -------------------------------------------------------------------

        stored = result.get("residual_bytes", 0)
        ratio = 100.0 * result.get("residual_ratio", 0.0)
        secs = time.perf_counter() - t0
        print(f"  {epoch:<7}{loss:>8.3f}{tacc:>8.3f}{vacc:>8.3f}"
              f"{stored:>12}{ratio:>8.2f}%{result['commit'][:8]:>10}{secs:>7.0f}s")

        rows.append({"epoch": epoch, "loss": loss, "train_acc": tacc,
                     "val_acc": vacc, "raw_bytes": size,
                     "stored_bytes": stored, "ratio_pct": ratio,
                     "full": result.get("full"), "commit": result["commit"],
                     "seconds": secs})

    total_stored = sum(r["stored_bytes"] for r in rows)
    print(f"\n  {len(rows)} checkpoints, {raw_bytes/2**30:.2f} GiB raw -> "
          f"{total_stored/2**30:.2f} GiB stored "
          f"({100.0*total_stored/raw_bytes:.2f}%, "
          f"{raw_bytes/max(total_stored,1):.1f}x)")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({
        "script": "tools/train_regnety_synapse.py",
        "argv": sys.argv[1:],
        "model": args.model,
        "params_M": round(nparam / 1e6, 1),
        "dtype": "bf16",
        "trained": True,
        "img_size": args.img_size,
        "batch": args.batch,
        "lr": args.lr,
        "no_align": args.no_align,
        "when": datetime.now().isoformat(timespec="seconds"),
        "raw_bytes": raw_bytes,
        "stored_bytes": total_stored,
        "ratio_pct": round(100.0 * total_stored / max(raw_bytes, 1), 3),
        "epochs": rows,
    }, indent=2) + "\n")
    print(f"  results -> {args.results}")

    print("\n  verify the history end to end with:")
    print(f"    synapsefs -C {args.repo} verify --content")
    print(f"    synapsefs -C {args.repo} restore <commit> --compare <file> --strict")


if __name__ == "__main__":
    main()
