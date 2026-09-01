"""The per-tensor anchor policy: whether one tensor's residual is worth keeping
as a residual, or should be promoted to its own anchor (stored full, base=None).

This is the "dynamic star" replacing the flat `graph.REBASE_INTERVAL` rule.
Today every tensor in a checkpoint gets the same answer -- full every Nth
commit, delta every other -- decided once for the whole file. That is wrong
in both directions: a frozen layer gets a fresh full copy it never needed,
and a fast-drifting layer keeps diffing against an anchor that went stale
commits ago, storing near-raw bytes every time without anyone noticing,
because `codec/chunk.py::encode_chunk`'s `allow_raw_fallback` already quietly
stores those chunks raw and nothing promotes the tensor so the *next* commit
can skip repeating that work.

Deliberately a pure function of two numbers, not a class, not something that
touches a repository. The commit-time cost of encoding is what produces
`ratio`; the manifest chain on disk is what produces `depth`; this module
only has to combine them, which is why it is worth keeping separate from
`graph.py` (reads objects) and `codec/checkpoint.py` (does the encoding).
"""

from __future__ import annotations

import enum

__all__ = ["Decision", "ANCHOR", "DELTA", "decide"]


class Decision(enum.Enum):
    ANCHOR = "anchor"
    DELTA = "delta"


ANCHOR = Decision.ANCHOR
DELTA = Decision.DELTA

#: Same break-even reasoning as `align.residual.NOT_ALIGNABLE_THRESHOLD`: 1.0
#: is where storing the delta costs exactly as much as storing the tensor
#: raw, so anything at or above it has already lost. 0.9 leaves margin rather
#: than waiting for the delta to be strictly worse than raw before admitting
#: the anchor is stale.
DEFAULT_TAU = 0.9

#: Upper bound on how many hops a reconstruction may make for one tensor.
#: This is the FUSE-read-latency guardrail: depth is allowed to grow only on
#: low-drift tensors (small deltas), and a tensor that drifts enough to need
#: re-anchoring resets to depth 0 on its own. `max_depth` makes the bound on
#: the pathological case explicit instead of hoped-for.
DEFAULT_MAX_DEPTH = 3


def decide(
    ratio: float,
    depth: int,
    *,
    tau: float = DEFAULT_TAU,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Decision:
    """Should this tensor's residual, at this depth, become a new anchor?

    `ratio` is `stored_bytes / original_bytes` for this tensor's chunks as
    actually encoded against its current anchor -- 1.0 means the delta cost
    as much as the raw tensor, matching `NOT_ALIGNABLE_THRESHOLD`'s convention
    that ratio is a fraction of the original, not a compression multiplier.

    `depth` is the chain length this tensor's manifest would have *after*
    this commit if kept as DELTA -- i.e. the caller's existing chain depth at
    the anchor, plus one. Passing the depth the decision is actually about
    (rather than the depth before it) is what lets `max_depth` bound what it
    says it bounds.

    Two independent triggers, either one enough:

    - `ratio >= tau`: the delta is not paying for itself, so keep it a delta
      no longer than it takes to prove that.
    - `depth >= max_depth`: irrespective of how cheap the delta is, do not let
      the reconstruction walk grow past the bound. A tensor that drifts
      slowly can still ride this out for a long time -- `max_depth` resets it
      only every `max_depth` commits, not every commit.
    """
    if ratio >= tau or depth >= max_depth:
        return ANCHOR
    return DELTA
