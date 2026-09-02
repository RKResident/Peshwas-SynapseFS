"""Can one lineage's checkpoint serve as the delta base for another lineage's?

The question behind cross-branch storage: two runs from different random
inits, trained on the same data, are believed to reach the same loss basin up
to a permutation of hidden units (Git Re-Basin, linear mode connectivity
modulo permutation). If that is true in the sense the codec needs, aligning
one against the other should leave a residual small enough to store as a
delta.

It is worth being precise about what the paper claims, because the two
statements are easy to conflate. Re-Basin shows the LOSS BARRIER along the
interpolation path between two aligned solutions vanishes. It does not show
the aligned WEIGHTS are close. A basin can be wide and flat, so two points in
it can have near-identical loss everywhere between them and still sit a full
weight-norm apart. Only the second property would help compression.

Three pairs are measured, and the controls are the point:

  permuted    the same weights with a known permutation applied. Alignment
              MUST drive this to ~0; if it does not, nothing else here means
              anything, because the solver is broken rather than the theory.

  init        two different random inits, untrained. The floor: no shared
              training, so whatever alignment achieves here is what it
              achieves from architecture alone.

  trained     two lineages trained separately. The actual question. If this
              lands near `init`, shared training bought nothing recoverable
              by a permutation.

`--threshold` is drawn on the output because a residual only helps if it is
below it; above, the codec stores the tensor in full and the alignment was
wasted work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synapsefs.align import config_parser, solver
from synapsefs.align.IR import TensorRef
from synapsefs.align.reader import SafetensorsReader
from synapsefs.align.residual import NOT_ALIGNABLE_THRESHOLD


def topology_for(path: Path, config: Path | None):
    with SafetensorsReader(str(path)) as r:
        refs = {n: TensorRef(name=n, shape=tuple(t.shape), dtype=t.dtype)
                for n, t in r.refs().items()}
    cfg = json.loads(config.read_text()) if config and config.is_file() else None
    return config_parser.parse(refs, cfg)


def report(label: str, base: Path, target: Path, config: Path,
           sweeps: int, threshold: float) -> None:
    topo = topology_for(target, config)
    res = solver.align_checkpoints(base, target, topo, max_sweeps=sweeps,
                                   threshold=threshold)
    s = res.residual_summary()
    # Weight by element count: the same reason group_bit_delta does. A mean
    # over tensors lets ten tiny norm buffers outvote the matrices that are
    # the entire checkpoint.
    tot = sum(a.numel for a in res.assessments.values())
    wpre = sum(a.pre * a.numel for a in res.assessments.values()) / tot
    wpost = sum(a.post * a.numel for a in res.assessments.values()) / tot
    ok = sum(a.numel for a in res.assessments.values() if a.post < threshold)
    print(f"  {label:<10} {wpre:9.4f} {wpost:9.4f} {s.helped:5d}/{s.tensors:<5d} "
          f"{ok/tot*100:8.1f}% {res.sweeps:4d} {res.wall_clock_s:8.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sweeps", type=int, default=3,
                    help="cap; a 4096-unit group's LAP is O(n^3) per sweep")
    ap.add_argument("--threshold", type=float, default=NOT_ALIGNABLE_THRESHOLD)
    ap.add_argument("pairs", nargs="+",
                    help="label=base.safetensors:target.safetensors")
    args = ap.parse_args()

    print(f"residuals are element-count weighted; 'usable' is the fraction of")
    print(f"weights whose aligned residual falls below {args.threshold}\n")
    print(f"  {'pair':<10} {'pre':>9} {'post':>9} {'permuted':>11} "
          f"{'usable':>9} {'swp':>4} {'time':>9}")
    for spec in args.pairs:
        label, _, rest = spec.partition("=")
        b, _, t = rest.partition(":")
        report(label, Path(b), Path(t), args.config, args.sweeps, args.threshold)


if __name__ == "__main__":
    main()
