"""How alignment scales with layer width, and how it degrades with noise.

Reproduces ARCHITECTURE.md 4.6.3.

Every other alignment number in the docs comes from one model: the 92M CNN
benchmark, whose widest permutation group is 1792 units. The PS evaluates up to
roughly 7B parameters, and the quantities that matter here grow with the *width*
of a layer, not with the parameter count -- the cost matrix is [n, n] and the
assignment solve is superlinear in n. A 7B CNN stays narrow (3x3 kernels
multiply channels by 9, so ~4000 channels gets you there); a 7B MLP does not
(~17800 hidden units). Those two land in very different places, and nothing in
the repo had ever been run wide enough to find out which.

So this sweeps width directly, on synthetic MLPs, with a known ground-truth
permutation. It answers three questions:

    1. what does alignment actually cost at n = 10000?
    2. how much noise can the solver take before it stops recovering?
    3. do width and noise add, or multiply?

Weights are synthetic Gaussians, not trained -- see "Caveats" at the bottom of
the output. The permutation convention copies tools/gen_fixtures.py exactly
(out[i] == w[p[i]]), which is the solver's own convention, so recovered
permutations compare to ground truth without inverting anything.

Run from the repository root:

    PYTHONPATH=. python tools/experiments/align_scaling.py             # both sweeps
    PYTHONPATH=. python tools/experiments/align_scaling.py --quick     # skip n>4096
    PYTHONPATH=. python tools/experiments/align_scaling.py --one --width 8192 --noise 1.0

Each configuration runs in its own subprocess, because `ru_maxrss` is a
high-water mark for the whole process: measure two configurations in one
interpreter and the second inherits the first's peak.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np

from synapsefs.align import config_parser, solver, lap
from synapsefs.align.IR import TensorRef

#: Widths to sweep. 10000 is the stress target; the curve is fitted on >= 2048,
#: below which a fixed ~80 MB interpreter baseline dominates the RSS reading.
WIDTHS = (512, 1024, 2048, 4096, 8192, 10000)

#: Relative noise levels: w += k * std(w) * N(0, 1). k = 1.0 means the
#: perturbation is as large as the weights themselves.
NOISE_LEVELS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


def build_mlp(in_dim, hidden, depth, out_dim, rng, dtype=np.float16):
    """layers.{i}.weight/bias plus norm affine on every hidden layer.

    Names match tools/gen_fixtures.py so config_parser sees the shape it
    expects. fp16 on purpose: it is what a real checkpoint holds, and it makes
    the solver pay the float32 widening that dominates its profile.
    """
    t = {}
    for i in range(depth):
        fan_in = in_dim if i == 0 else hidden
        fan_out = out_dim if i == depth - 1 else hidden
        scale = np.sqrt(2.0 / fan_in)
        t[f"layers.{i}.weight"] = (
            rng.standard_normal((fan_out, fan_in), dtype=np.float32) * scale
        ).astype(dtype)
        t[f"layers.{i}.bias"] = np.zeros(fan_out, dtype=dtype)
        if i < depth - 1:
            t[f"layers.{i}.norm.weight"] = rng.uniform(
                0.8, 1.2, size=fan_out).astype(dtype)
            t[f"layers.{i}.norm.bias"] = (
                rng.standard_normal(fan_out, dtype=np.float32) * 0.05).astype(dtype)
    return t


def permute_mlp(t, hidden, depth, in_dim, rng):
    """Permute every hidden group. Returns (permuted, {group: truth}).

    Input and output axes stay pinned -- permuting them would change the
    function as seen from outside, which is exactly what permutation symmetry
    does not do.
    """
    groups = {f"g_hidden_{i}": rng.permutation(hidden).astype(np.int32)
              for i in range(depth - 1)}
    ident_in = np.arange(in_dim, dtype=np.int32)
    out = {}
    for i in range(depth):
        p_in = ident_in if i == 0 else groups[f"g_hidden_{i - 1}"]
        p_out = groups.get(f"g_hidden_{i}")

        w = t[f"layers.{i}.weight"][:, p_in]
        if p_out is not None:
            w = w[p_out]
        out[f"layers.{i}.weight"] = np.ascontiguousarray(w)

        b = t[f"layers.{i}.bias"]
        out[f"layers.{i}.bias"] = b if p_out is None else b[p_out]

        if p_out is not None:
            for suf in ("norm.weight", "norm.bias"):
                out[f"layers.{i}.{suf}"] = t[f"layers.{i}.{suf}"][p_out]
    return out, groups


def add_noise(t, rel, rng):
    """w += rel * std(w) * N(0, 1), per tensor. rel <= 0 returns a copy."""
    if rel <= 0:
        return {k: v.copy() for k, v in t.items()}
    out = {}
    for k, v in t.items():
        f = v.astype(np.float32)
        s = float(f.std())
        if s > 0:
            f = f + rng.standard_normal(f.shape, dtype=np.float32) * (rel * s)
        out[k] = f.astype(v.dtype)
    return out


def run_one(width, depth, in_dim, out_dim, noise, seed, permute) -> dict:
    """One alignment. Returns the measurement as a plain dict."""
    rng = np.random.default_rng(seed)
    base = build_mlp(in_dim, width, depth, out_dim, rng)

    if permute:
        target, truth = permute_mlp(base, width, depth, in_dim, rng)
        target = add_noise(target, noise, rng)
    else:
        target = add_noise(base, noise, rng)
        truth = {f"g_hidden_{i}": np.arange(width, dtype=np.int32)
                 for i in range(depth - 1)}

    refs = {n: TensorRef(n, tuple(v.shape), "F16") for n, v in target.items()}
    topo = config_parser.parse(refs)

    t0 = time.perf_counter()
    res = solver.align_checkpoints(base, target, topo)
    align_s = time.perf_counter() - t0

    # Layer i's weight rows carry group g_hidden_i, so pi_row is directly
    # comparable to the ground truth for that group.
    accs = [lap.agreement(res.tensors[f"layers.{i}.weight"].pi_row,
                          truth[f"g_hidden_{i}"], width)
            for i in range(depth - 1)]
    post = [a.post for a in res.assessments.values() if a.post is not None]

    return {
        "width": width, "depth": depth, "noise": noise, "permuted": permute,
        "params_M": round(sum(v.size for v in base.values()) / 1e6, 2),
        "align_s": round(align_s, 2),
        "sweeps": res.sweeps,
        "accuracy": round(float(np.mean(accs)), 4),
        "not_alignable": len(res.not_alignable),
        "resid_post": round(float(np.mean(post)), 4) if post else None,
        "peak_rss_MB": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def spawn(args, **over) -> dict:
    """Run one configuration in a fresh interpreter, so ru_maxrss is its own."""
    cfg = {"width": args.width, "depth": args.depth, "in_dim": args.in_dim,
           "out_dim": args.out_dim, "noise": args.noise, "seed": args.seed,
           "permute": True}
    cfg.update(over)
    cmd = [sys.executable, __file__, "--one",
           "--width", str(cfg["width"]), "--depth", str(cfg["depth"]),
           "--in-dim", str(cfg["in_dim"]), "--out-dim", str(cfg["out_dim"]),
           "--noise", str(cfg["noise"]), "--seed", str(cfg["seed"])]
    if not cfg["permute"]:
        cmd.append("--no-permute")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def fit_exponent(xs, ys) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one", action="store_true",
                    help="run a single configuration and print one JSON line")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=3,
                    help="linear layers; depth-1 permutable groups")
    ap.add_argument("--in-dim", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-permute", action="store_true",
                    help="fine-tune case: identity is the right answer")
    ap.add_argument("--quick", action="store_true",
                    help="skip widths above 4096 (the slow half)")
    args = ap.parse_args()

    if args.one:
        print(json.dumps(run_one(args.width, args.depth, args.in_dim,
                                 args.out_dim, args.noise, args.seed,
                                 not args.no_permute)))
        return

    widths = [w for w in WIDTHS if not (args.quick and w > 4096)]

    print(f"\n  width sweep -- permuted, noise {args.noise}, depth {args.depth}"
          f" ({args.depth - 1} groups)")
    print(f"  {'width':>7}{'params':>10}{'align':>10}{'sweeps':>8}"
          f"{'peak RSS':>11}{'recovery':>10}")
    rows = []
    for w in widths:
        r = spawn(args, width=w)
        rows.append(r)
        print(f"  {r['width']:>7}{r['params_M']:>9.1f}M{r['align_s']:>9.2f}s"
              f"{r['sweeps']:>8}{r['peak_rss_MB']:>10.0f}M"
              f"{r['accuracy'] * 100:>9.1f}%")

    big = [r for r in rows if r["width"] >= 2048]
    if len(big) >= 2:
        xs = [r["width"] for r in big]
        kt = fit_exponent(xs, [r["align_s"] for r in big])
        kr = fit_exponent(xs, [r["peak_rss_MB"] for r in big])
        print(f"\n  fitted on n >= 2048:  time ~ n^{kt:.2f}   RSS ~ n^{kr:.2f}")
        print("  Not n^3. The Hungarian solve's worst case is a near-degenerate")
        print("  cost matrix; a clean permutation is its best case.")

    print(f"\n  noise sweep -- width {args.width}, depth {args.depth}")
    print(f"  {'noise':>7}{'recovery':>11}{'resid post':>13}"
          f"{'sweeps':>8}{'align':>10}{'n/align':>9}")
    for n in NOISE_LEVELS:
        r = spawn(args, noise=n)
        print(f"  {n:>7}{r['accuracy'] * 100:>10.1f}%{r['resid_post']:>13}"
              f"{r['sweeps']:>8}{r['align_s']:>9.2f}s{r['not_alignable']:>9}")

    print(f"\n  fine-tune fast path -- no permutation, noise {args.noise}")
    print(f"  {'width':>7}{'align':>10}{'sweeps':>8}")
    for w in widths:
        r = spawn(args, width=w, permute=False)
        print(f"  {r['width']:>7}{r['align_s']:>9.2f}s{r['sweeps']:>8}")

    print("\n  Caveats, because these numbers will be quoted:")
    print("  - Weights are synthetic Gaussians. Trained weights carry structure")
    print("    that could make the cost matrix more or less decisive.")
    print("  - Both models are passed as in-memory dicts, so peak RSS includes")
    print("    them. A real commit passes SafetensorsReader and mmaps instead;")
    print("    the n^2 driver is the float32 widening, the [n,n] cost matrix,")
    print("    and residual.py's temporaries.")
    print("  - depth 3 = 2 permutable groups. Wall clock scales with group")
    print("    count; peak RSS does not, since groups are solved one at a time.")
    print("  - One seed per configuration.")


if __name__ == "__main__":
    main()
