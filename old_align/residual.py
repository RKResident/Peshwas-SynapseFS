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
    # Subtract at the input width and let `norm` accumulate in float64.
    #
    # Widening both operands first made the subtraction exact, and bought
    # nothing: what the float64 accumulator is protecting against is drift over
    # ten million ADDITIONS, which `norm`'s einsum already handles whatever the
    # input dtype is. An elementwise float32 subtraction is correctly rounded,
    # so the error is at most one float32 ULP per element -- invisible against
    # a ratio compared to a 0.9 threshold.
    #
    # It was not free. Two float64 temporaries per call, 144 calls per
    # alignment, 231 MiB apiece for the largest tensor: 40% of the runtime
    # after the matrix cache landed, and the single largest allocator in the
    # solver.
    return norm(t - b) / scale


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
    #: Element count. `pre` and `post` are relative and so not comparable
    #: between tensors; `group_helped` weights by this to turn them into an
    #: estimate of BYTES, which is what the decision is actually about.
    numel: int = 0

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
    return Assessment(pre, post, is_alignable(post, threshold),
                      int(np.asarray(target).size))


#: Clamp for the log-ratio when a residual is exactly zero. e**8 is a factor of
#: ~3000 either way, far past any real permutation's effect, so it saturates
#: the vote without letting a single perfectly-matched tensor become infinite.
_LOG_CLAMP = 8.0


def group_bit_delta(assessments) -> float:
    """Estimated change in stored bits if this permutation is applied.

    Negative means the permutation is expected to shrink the commit.

    The codec stores a residual, and the bits a residual costs scale as
    `numel * log2(typical |delta|)` -- doubling every delta costs one more bit
    for every element. So a tensor's contribution is its element count times
    the log of how much its residual changed, and the group's verdict is the
    sum. Nothing else in this file weights tensors against each other, and
    getting that weight wrong is the whole difficulty:

      by COUNT, every tensor votes equally, so ten BatchNorm buffers outvote
      the kernel they belong to;

      by NORM (||target||), magnitude decides, and `running_var` holds
      variances -- 1792 of them summed to 199 while a 21.7-million-element
      convolution kernel summed to 11.8, so six statistics buffers outvoted
      50 MiB of weights whose residual had TRIPLED;

      by ELEMENT COUNT, the tensors that actually occupy the commit decide,
      which is the question being asked.

    Measured on epoch 1 -> 2 of the 92M benchmark, the norm-weighted version
    accepted a permutation that grew the commit from 145.8 MB to 153.2 MB.
    """
    total = 0.0
    for a in assessments:
        if not (np.isfinite(a.pre) and np.isfinite(a.post)) or a.numel <= 0:
            continue
        if a.pre <= 0.0 and a.post <= 0.0:
            continue
        if a.post <= 0.0:
            ratio = -_LOG_CLAMP
        elif a.pre <= 0.0:
            ratio = _LOG_CLAMP
        else:
            ratio = float(np.clip(np.log(a.post / a.pre), -_LOG_CLAMP, _LOG_CLAMP))
        total += a.numel * ratio
    return total


def group_helped(assessments, min_gain: float = 1e-6) -> bool:
    """Is this permutation worth applying to every tensor in its group?

    The decision a permutation group needs, and the reason it cannot be made
    one tensor at a time. A group is ONE ordering shared by every tensor that
    touches the axis -- a layer's weight and bias, its norm's parameters and
    BatchNorm buffers, and the next layer's input columns. Accepting it for
    some members and rejecting it for others does not produce a partially
    aligned checkpoint; it produces an incoherent one, because the stored
    permutation no longer describes a correspondence between the two models'
    units. Reconstruction still works -- each tensor gathers with whatever it
    stored -- so nothing catches this except looking.

    A group with nothing finite to judge is rejected: an unmeasured
    permutation is not an improvement anyone can defend.
    """
    usable = [a for a in assessments
              if np.isfinite(a.pre) and np.isfinite(a.post) and a.numel > 0]
    if not usable:
        return False
    return group_bit_delta(usable) < -min_gain


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