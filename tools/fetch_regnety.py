"""Download a large RegNetY backbone and write a bf16 baseline checkpoint.

Run this ONCE on the machine that will do the fine-tuning, before
`train_regnety_synapse.py`. It exists separately because the download is the
slow, network-bound, resumable part and the training script should not have to
own it.

Why RegNetY-128GF specifically: it is the largest pure-CNN backbone with
published safetensors weights (644.8M parameters), which puts it ~7x above the
92M STL-10 benchmark every number in ARCHITECTURE.md comes from. The PS scopes
grading to MLPs and CNNs, so a CNN is what the fixture has to be -- the LLM
checkpoint suites (Pythia, OLMo) are transformers and `config_parser`'s
UNSUPPORTED_HINTS rejects them outright.

The hub stores these weights as **F32**. We cast to bf16 on the way out,
because bf16 is what the PS grades and what the codec's escape encoding was
tuned for -- fp16 and bf16 behave very differently here (ARCHITECTURE.md
4.1.3: gap-1 ratios of 74.32% against 52.72%).

Requires (on the target machine, not here):
    pip install torch timm safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "regnety_1280.swag_ft_in1k"   # 644.8M params, 384x384


def build(model_name: str, num_classes: int):
    import timm
    # num_classes=0 would drop the head; we keep a real head so the fine-tune
    # has something to train and the topology has a pinned output group.
    model = timm.create_model(model_name, pretrained=True,
                              num_classes=num_classes)
    model.eval()
    return model


def state_dict_bf16(model) -> dict:
    """Contiguous bf16 tensors, floats only cast.

    `num_batches_tracked` is int64 and must stay int64 -- .half()/.bfloat16()
    on a whole state_dict silently leaves it alone in PyTorch, but doing the
    cast explicitly documents that the mixed-dtype file is intended, not an
    accident. ARCHITECTURE.md 7.7: every real reduced-precision checkpoint is
    mixed-dtype, and a codec assuming one width per file breaks on the first
    real model.
    """
    import torch
    out = {}
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            out[k] = v.detach().to(torch.bfloat16).contiguous()
        else:
            out[k] = v.detach().contiguous()
    return out


def topology_config(model, num_classes: int) -> dict:
    """A HuggingFace-shaped config.json for the alignment engine.

    `config_parser` treats SHAPES as normative and config only as
    corroboration -- it checks that the widths declared here are a subset of
    the widths actually present, and complains if they disagree. So this is
    deliberately minimal: emitting the distinct channel widths found in the
    conv stack is enough to corroborate, and inventing more risks a
    disagreement that makes the parser reject a checkpoint it could have read.
    """
    widths = []
    for name, p in model.named_parameters():
        if p.ndim == 4:                        # conv kernel [out, in, kh, kw]
            w = int(p.shape[0])
            if w not in widths:
                widths.append(w)
    return {
        "architecture": "RegNetY",
        "widths": sorted(widths),
        "num_classes": num_classes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-classes", type=int, default=10,
                    help="head width for the fine-tune task")
    ap.add_argument("--out", type=Path, default=Path("regnety_base"))
    args = ap.parse_args()

    from safetensors.torch import save_file

    print(f"building {args.model} (downloads weights on first run)...")
    model = build(args.model, args.num_classes)

    n = sum(p.numel() for p in model.parameters())
    print(f"  {n/1e6:.1f}M parameters")

    args.out.mkdir(parents=True, exist_ok=True)
    sd = state_dict_bf16(model)
    ckpt = args.out / "epoch000.safetensors"
    save_file(sd, str(ckpt))

    cfg = topology_config(model, args.num_classes)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    mb = ckpt.stat().st_size / 1048576
    print(f"  wrote {ckpt}  ({mb:.0f} MiB, bf16)")
    print(f"  wrote {args.out / 'config.json'}  widths={len(cfg['widths'])}")
    print(f"\n  next: python tools/train_regnety_synapse.py --base {args.out} ...")


if __name__ == "__main__":
    main()