"""The alignment benchmark: did it recover the permutation, and how fast.

Two numbers per fixture that are easy to conflate:

  recovery   fraction of units placed on their ground-truth partner. Only
             defined where a ground truth exists.
  residual   ||B - A_aligned|| / ||B||, before and after. 'before' is exactly
             what --no-align would produce, so pre -> post is the evidence for
             row (d) of the codec table without running anything twice.

Timing excludes the residual pass. align_checkpoints(measure=True) walks every
tensor to compute norms, which is I/O and numpy, not search; reporting it as
alignment wall-clock would flatter nothing and mislead everything. Timing runs
use measure=False and take the best of --repeat.

The threshold section is the point of the unrelated fixtures: it measures the
gap between the worst residual among pairs that ARE alignable and the best
among pairs that are not, so NOT_ALIGNABLE_THRESHOLD can be a measurement
rather than a guess.

Human table to stderr, JSON to stdout, per CLI.md 1.2.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synapsefs.align.config_parser import from_checkpoint
from synapsefs.align.lap import agreement
from synapsefs.align.reader import SafetensorsReader
from synapsefs.align.residual import NOT_ALIGNABLE_THRESHOLD
from synapsefs.align.solver import align_checkpoints


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def topology_for(where: str):
    config_path = os.path.join(where, "config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else None
    with SafetensorsReader(os.path.join(where, "base.safetensors")) as r:
        return from_checkpoint(r, config)


def recovery(result, topo, truth: dict | None) -> float | None:
    """Size-weighted fraction of units placed correctly. None without truth."""
    if truth is None:
        return None
    total = hits = 0
    for gid, group in topo.groups.items():
        if group.pinned or not group.row_members:
            continue
        want = np.array(truth[gid]) if gid in truth else np.arange(group.size)
        got = result[group.row_members[0]].pi_row
        hits += agreement(got, want, group.size) * group.size
        total += group.size
    return None if total == 0 else hits / total


def evaluate(entry: dict, root: str, repeat: int, max_sweeps: int = 25) -> dict:
    where = os.path.join(root, f"{entry['name']}-{entry['variant']}")
    base = os.path.join(where, "base.safetensors")
    target = os.path.join(where, "target.safetensors")
    truth = entry.get("permutations")

    t0 = time.perf_counter()
    topo = topology_for(where)
    parse_s = time.perf_counter() - t0

    result = align_checkpoints(base, target, topo, seed=0, max_sweeps=max_sweeps)

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        align_checkpoints(base, target, topo, seed=0, measure=False,
                          max_sweeps=max_sweeps)
        best = min(best, time.perf_counter() - t0)

    summary = result.residual_summary()
    sizes = [g.size for g in topo.solvable_groups()]
    return {
        "name": entry["name"],
        "variant": entry["variant"],
        "kind": entry["kind"],
        "dtype": entry["dtype"],
        "mib": entry["bytes"] / 2 ** 20,
        "tensors": len(result.tensors),
        "groups": len(sizes),
        "widest_group": max(sizes) if sizes else 0,
        "sweeps": result.sweeps,
        "converged": result.converged,
        "identity": result.identity,
        "identity_expected": entry["identity_expected"],
        "recovery": recovery(result, topo, truth),
        "residual_pre": summary.mean_pre,
        "residual_post": summary.mean_post,
        "worst_post": max((a.residual_post for a in result.tensors.values()),
                          default=float("nan")),
        "not_alignable": len(result.not_alignable),
        "alignable_expected": entry["alignable"],
        "align_s": best,
        "parse_s": parse_s,
        "peak_rss_mib": peak_rss_mib(),
    }


def verdict(row: dict) -> str:
    if row["identity_expected"] and not row["identity"]:
        return "FAIL identity not detected"
    if not row["alignable_expected"]:
        return "ok" if row["not_alignable"] else "FAIL should be unalignable"
    if not row["converged"]:
        return "ok (hit sweep cap)"
    if row["recovery"] is not None and row["recovery"] < 0.999:
        return f"PARTIAL {row['recovery'] * 100:.1f}%"
    return "ok"


def table(rows: list[dict]) -> list[str]:
    head = (f"{'fixture':<22}{'MiB':>7}{'grp':>5}{'width':>7}{'swp':>5}"
            f"{'recov':>8}{'residual pre->post':>22}{'align s':>10}  verdict")
    out = [head, "-" * len(head)]
    for r in rows:
        rec = "  --  " if r["recovery"] is None else f"{r['recovery'] * 100:5.1f}%"
        out.append(
            f"{r['name'] + '-' + r['variant']:<22}"
            f"{r['mib']:7.2f}{r['groups']:5d}{r['widest_group']:7d}"
            f"{str(r['sweeps']) + ('' if r['converged'] else '+'):>5}{rec:>8}"
            f"{r['residual_pre']:11.3f} ->{r['residual_post']:8.3f}"
            f"{r['align_s']:10.3f}  {verdict(r)}"
        )
    return out


def scaling(rows: list[dict]) -> list[str]:
    """Wall clock against the widest group, which is what drives the n^3 cost."""
    ordered = sorted((r for r in rows if r["variant"] == "permuted"),
                     key=lambda r: r["widest_group"])
    out = [f"{'fixture':<22}{'widest':>8}{'sweeps':>8}{'align s':>10}"
           f"{'s / sweep':>12}  growth"]
    out.append("-" * len(out[0]))
    prev = None
    for r in ordered:
        per = r["align_s"] / max(r["sweeps"], 1)
        growth = ""
        if prev and prev[1] > 0:
            width_ratio = r["widest_group"] / prev[0]
            if width_ratio > 1:
                exponent = np.log(per / prev[1]) / np.log(width_ratio)
                growth = f"  n^{exponent:.1f}"
        out.append(f"{r['name']:<22}{r['widest_group']:8d}{r['sweeps']:8d}"
                   f"{r['align_s']:10.3f}{per:12.4f}{growth}")
        prev = (r["widest_group"], per)
    return out


def calibrate(rows: list[dict]) -> dict:
    """Measure the gap the not-alignable threshold has to sit in."""
    alignable = [r for r in rows if r["alignable_expected"]]
    unrelated = [r for r in rows if not r["alignable_expected"]]
    worst_ok = max((r["worst_post"] for r in alignable), default=float("nan"))
    best_bad = min((r["worst_post"] for r in unrelated), default=float("nan"))
    good = np.isfinite(worst_ok) and np.isfinite(best_bad) and worst_ok < best_bad
    return {
        "n_alignable": len(alignable),
        "n_unrelated": len(unrelated),
        "worst_alignable_residual": worst_ok,
        "best_unrelated_residual": best_bad,
        "separated": bool(good),
        "midpoint": float(np.sqrt(worst_ok * best_bad)) if good else None,
        "current_threshold": NOT_ALIGNABLE_THRESHOLD,
        "current_threshold_correct": bool(
            good and worst_ok < NOT_ALIGNABLE_THRESHOLD <= best_bad),
    }


def calibration_lines(cal: dict) -> list[str]:
    if not cal["n_unrelated"] or not cal["n_alignable"]:
        missing = "unrelated" if not cal["n_unrelated"] else "alignable"
        return [f"threshold: not calibrated -- this fixture set has no "
                f"{missing} pairs to measure against"]
    if not cal["separated"]:
        return ["threshold: the two populations OVERLAP -- no single threshold "
                "separates alignable from unrelated on this fixture set"]
    return [
        f"threshold: alignable pairs stay under {cal['worst_alignable_residual']:.3f}, "
        f"unrelated pairs never fall below {cal['best_unrelated_residual']:.3f}",
        f"           any threshold in that gap works; geometric midpoint is "
        f"{cal['midpoint']:.3f}, current is {cal['current_threshold']:.2f} "
        f"({'inside' if cal['current_threshold_correct'] else 'OUTSIDE'} the gap)",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="alignment recovery and wall clock")
    ap.add_argument("--fixtures", default="bench/fixtures")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--only", action="append")
    ap.add_argument("--max-sweeps", type=int, default=25,
                    help="cap; unrelated pairs otherwise burn every sweep")
    args = ap.parse_args(argv)

    index_path = os.path.join(args.fixtures, "index.json")
    if not os.path.exists(index_path):
        print(f"no index.json in {args.fixtures}; run make_fixtures.py first",
              file=sys.stderr)
        return 2
    index = json.load(open(index_path))
    if args.only:
        index = [e for e in index if e["name"] in args.only]

    rows = [evaluate(entry, args.fixtures, args.repeat, args.max_sweeps)
            for entry in index]
    cal = calibrate(rows)
    failures = [r for r in rows if verdict(r).startswith("FAIL")]

    for line in table(rows):
        print(line, file=sys.stderr)
    print("", file=sys.stderr)
    for line in scaling(rows):
        print(line, file=sys.stderr)
    print("", file=sys.stderr)
    for line in calibration_lines(cal):
        print(line, file=sys.stderr)
    print(f"peak RSS {peak_rss_mib():.0f} MiB, {len(failures)} failure(s)",
          file=sys.stderr)

    if args.json:
        json.dump({"fixtures": rows, "calibration": cal,
                   "failures": [r["name"] + "-" + r["variant"] for r in failures]},
                  sys.stdout, indent=1, sort_keys=True, default=float)
        sys.stdout.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())