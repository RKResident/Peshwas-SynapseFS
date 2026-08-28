"""How much did alignment actually buy, and is a delta worth storing at all?

Residuals are relative to the TARGET's own magnitude:

    ||B - A|| / ||B||

not to the unaligned residual. Comparing post to pre would score every
fine-tune at exactly 1.00 -- identity was already the right answer, so nothing
improved -- and flag the easiest case in the system as not alignable. Against
||B|| a fine-tune scores ~0.01 and unrelated tensors score ~1.41, which is what
the decision actually needs.

THE THRESHOLD. Break-even is 1.0: at that point the delta is as large as the
tensor itself and storing B raw wins outright. Two unrelated tensors of similar
scale sit at sqrt(2) ~ 1.41 because the difference of independent samples has
twice the variance. The default leaves margin below break-even for framing and
varint overhead; it is a number bench/ should confirm against real fixtures,
not a constant to defend forever.

Norms accumulate in float64. A float32 dot over ten million elements drifts far
enough to move a borderline threshold decision, and this is the one number the
CLI reports and the codec branches on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOT_ALIGNABLE_THRESHOLD = 0.9


def norm(a: np.ndarray) -> float:
    """L2 norm, accumulated in float64 regardless of the input dtype."""
    flat = np.asarray(a).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.dot(flat, flat, out=None) if flat.dtype == np.float64
                         else np.einsum("i,i->", flat, flat, dtype=np.float64)))


def relative_residual(target: np.ndarray, base: np.ndarray) -> float:
    """||target - base|| / ||target||. Non-finite or unusable gives inf."""
    t, b = np.asarray(target), np.asarray(base)
    if t.shape != b.shape:
        return float("inf")
    if not (np.isfinite(t).all() and np.isfinite(b).all()):
        return float("inf")
    scale = norm(t)
    if scale == 0.0:
        return 0.0 if np.array_equal(t, b) else float("inf")
    return norm(t.astype(np.float64) - b.astype(np.float64)) / scale


def improvement(pre: float, post: float) -> float:
    """Fraction of the residual that alignment removed. 0.0 when it did not."""
    if not np.isfinite(pre) or pre <= 0.0:
        return 0.0
    if not np.isfinite(post):
        return 0.0
    return max(0.0, (pre - post) / pre)


def is_alignable(post: float, threshold: float = NOT_ALIGNABLE_THRESHOLD) -> bool:
    """A delta is worth storing only if it is meaningfully smaller than B."""
    return bool(np.isfinite(post) and post < threshold)


@dataclass
class Assessment:
    pre: float = float("nan")
    post: float = float("nan")
    alignable: bool = True

    @property
    def improvement(self) -> float:
        return improvement(self.pre, self.post)

    @property
    def helped(self) -> bool:
        """Did the permutation do anything, as opposed to identity being right?"""
        return self.improvement > 1e-6


def assess(target: np.ndarray, base: np.ndarray,
           aligned: np.ndarray | None = None,
           threshold: float = NOT_ALIGNABLE_THRESHOLD) -> Assessment:
    """pre is against the unpermuted base, post against the aligned one."""
    pre = relative_residual(target, base)
    post = pre if aligned is None else relative_residual(target, aligned)
    return Assessment(pre, post, is_alignable(post, threshold))


@dataclass
class Summary:
    tensors: int = 0
    alignable: int = 0
    not_alignable: int = 0
    helped: int = 0
    mean_pre: float = 0.0
    mean_post: float = 0.0
    worst: str | None = None

    @property
    def mean_improvement(self) -> float:
        return improvement(self.mean_pre, self.mean_post)

    def line(self) -> str:
        return (f"residual {self.mean_pre:.3f} -> {self.mean_post:.3f} "
                f"({self.mean_improvement * 100:.1f}% removed by alignment), "
                f"{self.helped}/{self.tensors} tensors permuted, "
                f"{self.not_alignable} not alignable")


def summarize(assessments: dict[str, Assessment]) -> Summary:
    """Aggregate for the CLI and the benchmark table.

    These are norm ratios, not the graded residual_ratio -- that one is bytes
    after encoding and belongs to the codec team. This measures the alignment.
    """
    usable = {n: a for n, a in assessments.items() if np.isfinite(a.pre)}
    s = Summary(tensors=len(assessments))
    s.alignable = sum(1 for a in assessments.values() if a.alignable)
    s.not_alignable = s.tensors - s.alignable
    s.helped = sum(1 for a in assessments.values() if a.helped)
    if usable:
        s.mean_pre = float(np.mean([a.pre for a in usable.values()]))
        s.mean_post = float(np.mean([a.post for a in usable.values()]))
        s.worst = max(usable, key=lambda n: usable[n].post)
    return s


def warning_lines(name: str, a: Assessment) -> list[str]:
    """CLI.md 3.2, verbatim. Explicit reporting is graded; silence is not."""
    shown = "inf" if not np.isfinite(a.post) else f"{a.post:.2f}"
    return [f"warning: '{name}' not meaningfully alignable "
            f"(residual norm {shown} of baseline)",
            "         stored in full (raw-zstd), no delta applied"]