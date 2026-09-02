from __future__ import annotations

import sys

import numpy as np


def lines(result, verbose: bool = False) -> list[str]:
    out = [f"Aligning against base ({result.groups} permutation groups)"]

    if not result.solved:
        out.append(f"  {len(result.unsolved_groups)} group(s) could NOT be solved "
                   f"\u2014 those tensors are stored unaligned")
    elif result.identity:
        out.append("  identity permutation detected \u2014 fast path")
    else:
        moved = sum(1 for a in result.tensors.values() if not a.identity)
        out.append(f"  {moved} of {len(result.tensors)} tensors permuted, "
                   f"converged in {result.sweeps} sweep"
                   f"{'' if result.sweeps == 1 else 's'}")

    if verbose:
        sizes = ", ".join(f"{g}={n}" for g, n in sorted(result.group_sizes.items()))
        out.append(f"  sizes: {sizes}")
        width = max((len(n) for n in result.tensors), default=0)
        for name in sorted(result.tensors):
            a = result[name]
            axes = ",".join(x for x in ("row" if a.pi_row is not None else "",
                                        "col" if a.pi_col is not None else "") if x)
            block = f"block={a.col_block_size}" if a.col_block_size > 1 else ""
            out.append(f"  {name:<{width}}  {axes or 'identity':<8}{block:<10}"
                       f"residual {a.residual_pre:.3f} -> {a.residual_post:.3f}")
        if result.unassigned:
            out.append(f"  unassigned: {', '.join(result.unassigned)}")

    if result.unsolved_groups:
        out.append(f"warning: could not solve {', '.join(result.unsolved_groups)}; "
                   f"alignment did nothing for the tensors on those axes")
    if not result.converged:
        out.append(f"warning: stopped at the {result.sweeps}-sweep cap "
                   f"without converging")
    for name in result.not_alignable:
        shown = result[name].residual_post
        shown = "inf" if not np.isfinite(shown) else f"{shown:.2f}"
        out.append(f"warning: '{name}' not meaningfully alignable "
                   f"(residual norm {shown} of baseline)")
        out.append("         stored in full (raw-zstd), no delta applied")
    return out


def emit(result, verbose: bool = False, stream=None) -> None:
    for line in lines(result, verbose):
        print(line, file=sys.stderr if stream is None else stream)


def _clean(x: float):
    """json.dumps writes a bare NaN token, which no strict parser accepts."""
    return round(x, 6) if np.isfinite(x) else None


def to_json(result, verbose: bool = False) -> dict:
    out = {
        "groups": result.groups,
        "identity": result.identity,
        "wall_clock_s": round(result.wall_clock_s, 3),
        "not_alignable": list(result.not_alignable),
        "unsolved_groups": list(result.unsolved_groups),
        "sweeps": result.sweeps,
        "converged": result.converged,
    }
    if result.unassigned:
        out["unassigned"] = list(result.unassigned)
    if verbose:
        out["tensors"] = {
            name: {"pi_row": a.pi_row is not None,
                   "pi_col": a.pi_col is not None,
                   "col_block_size": a.col_block_size,
                   "alignable": a.alignable,
                   "residual_pre": _clean(a.residual_pre),
                   "residual_post": _clean(a.residual_post)}
            for name, a in sorted(result.tensors.items())
        }
    return out