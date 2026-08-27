"""What the CLI prints, and what it serialises.

CLI.md 1.2: stdout is results only -- under --json exactly one document and
nothing else -- because the benchmark harness parses it. Every line this module
produces is progress or diagnosis, so it all goes to stderr. emit() has no way
to write to stdout at all; the JSON goes back to the caller as a dict and the
CLI is what prints it.

--quiet suppresses PROGRESS. It does not suppress not-alignable warnings or
unsolved-group warnings: the PS grades explicit reporting over silent
degradation, so a flag that hides degradation would defeat the point. Only -q
plus a clean run is silent.

Rendering is pure. render() returns lines; emit() writes them. Tests read the
lines rather than capturing a stream, and the stdout guarantee is testable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .config_parser import describe
from .residual import warning_lines

RESET = "\033[0m"
YELLOW = "\033[33m"
DIM = "\033[2m"


@dataclass
class ReportOptions:
    verbose: bool = False
    quiet: bool = False
    color: bool | None = None  # None = auto-detect

    def coloured(self, stream) -> bool:
        if self.color is not None:
            return self.color
        return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def short(ref: str | None, n: int = 6) -> str:
    return "?" if not ref else (ref[:n] if len(ref) > n else ref)


def header_line(result, base_ref: str | None = None) -> str:
    plural = "" if result.groups == 1 else "s"
    return (f"Aligning against {short(base_ref)} "
            f"({result.groups} permutation group{plural})")


def status_lines(result) -> list[str]:
    if not result.solved:
        return [f"  {len(result.unsolved_groups)} group(s) could NOT be solved "
                f"— those tensors are stored unaligned"]
    if result.identity:
        return ["  identity permutation detected \u2014 fast path"]
    moved = sum(1 for a in result.tensors.values() if not a.identity)
    sweeps = f"{result.sweeps} sweep{'' if result.sweeps == 1 else 's'}"
    how = "converged in" if result.converged else "stopped after"
    return [f"  {moved} of {len(result.tensors)} tensors permuted, {how} {sweeps}"]


def warnings_for(result) -> list[str]:
    out: list[str] = []
    if result.unsolved_groups:
        out.append(f"warning: {len(result.unsolved_groups)} permutation group(s) "
                   f"could not be solved: "
                   f"{', '.join(result.unsolved_groups[:5])}")
        out.append("         alignment did nothing for the tensors on those axes")
    if not result.converged:
        out.append(f"warning: alignment stopped at the {result.sweeps}-sweep cap "
                   "without converging")
    for name in result.not_alignable:
        verdict = result.assessments.get(name)
        if verdict is None:
            out.append(f"warning: '{name}' not alignable (no comparable base tensor)")
            out.append("         stored in full (raw-zstd), no delta applied")
        else:
            out.extend(warning_lines(name, verdict))
    return out


def detail_lines(result, topo=None) -> list[str]:
    """-v: group geometry first, then one line per tensor."""
    out: list[str] = []
    if topo is not None:
        out.extend("  " + line for line in describe(topo).splitlines())
    if result.assessments:
        out.append("  " + result.residual_summary().line())
    width = max((len(n) for n in result.tensors), default=0)
    for name in sorted(result.tensors):
        a = result.tensors[name]
        axes = ",".join(x for x in ("row" if a.pi_row is not None else "",
                                    "col" if a.pi_col is not None else "") if x)
        bits = [f"  {name:<{width}}  {axes or 'identity':<7}"]
        if a.pi_col is not None and a.col_block_size > 1:
            bits.append(f"block={a.col_block_size:<4}")
        else:
            bits.append(" " * 10)
        if name in result.assessments:
            v = result.assessments[name]
            bits.append(f"residual {v.pre:.3f} -> {v.post:.3f}")
            if not v.alignable:
                bits.append("NOT ALIGNABLE")
        out.append("".join(bits).rstrip())
    if result.unassigned:
        out.append(f"  unassigned ({len(result.unassigned)}): "
                   + ", ".join(result.unassigned[:8])
                   + (" ..." if len(result.unassigned) > 8 else ""))
    return out


def render(result, topo=None, base_ref: str | None = None,
           opts: ReportOptions | None = None, colour: bool = False) -> list[str]:
    opts = opts or ReportOptions()
    lines: list[str] = []
    if not opts.quiet:
        lines.append(header_line(result, base_ref))
        lines.extend(status_lines(result))
        if opts.verbose:
            lines.extend(paint(x, DIM, colour) for x in detail_lines(result, topo))
    lines.extend(paint(x, YELLOW, colour) for x in warnings_for(result))
    return lines


def emit(result, topo=None, base_ref: str | None = None,
         opts: ReportOptions | None = None, stream=None) -> None:
    """Writes to stderr. There is deliberately no stdout path here."""
    opts = opts or ReportOptions()
    stream = sys.stderr if stream is None else stream
    for line in render(result, topo, base_ref, opts, opts.coloured(stream)):
        print(line, file=stream)


def alignment_json(result, verbose: bool = False) -> dict:
    """The 'alignment' block of CLI.md 3.1. -v adds a per-tensor map."""
    out = dict(result.as_json())
    if result.unassigned:
        out["unassigned"] = list(result.unassigned)
    if verbose:
        out["tensors"] = {
            name: {
                "pi_row": a.pi_row is not None,
                "pi_col": a.pi_col is not None,
                "col_block_size": a.col_block_size,
                "alignable": a.alignable,
                "residual_pre": None if a.residual_pre != a.residual_pre
                else round(a.residual_pre, 6),
                "residual_post": None if a.residual_post != a.residual_post
                else round(a.residual_post, 6),
            }
            for name, a in sorted(result.tensors.items())
        }
    return out