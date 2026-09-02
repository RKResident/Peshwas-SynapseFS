"""The assignment step, and the permutation algebra around it.

group_cost() scores target unit i against base unit j; solve() picks one match
per row and column maximising the total. scipy does the hard part. Everything
else here exists to keep one convention straight:

    p[i] is the BASE index that TARGET index i was diffed against.

That is FileFormat.md 4.2, and it is the same array the codec gathers with and
the FUSE reader gathers with. Because the cost matrix is target-major, scipy's
col_ind IS p -- no inversion anywhere in the pipeline. invert() exists for
tests and for reading a permutation the other way round, not for the main path.

compose() is defined by what a gather does, so the direction cannot be
remembered backwards:

    A[compose(f, s)] == A[f][s]

from which compose(p, invert(p)) == identity follows rather than being asserted.

Identity is None throughout. solve() returns None when the assignment comes
back as the identity, which is what makes convergence-at-sweep-0 the fast path
and what gets written as null in the tensor-manifest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .Error import NotAlignable

Perm = "np.ndarray | None"


def solve(cost: np.ndarray, maximize: bool = True) -> np.ndarray | None:
    """Best one-to-one matching for a target-major cost matrix. None = identity."""
    c = np.asarray(cost)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cost matrix must be square, got shape {c.shape}")
    n = c.shape[0]
    if n == 0:
        raise ValueError("cost matrix is empty")
    if n == 1:
        return None
    if not np.isfinite(c).all():
        raise NotAlignable(
            f"cost matrix holds {int((~np.isfinite(c)).sum())} non-finite entries; "
            "cannot solve this group"
        )
    p = linear_sum_assignment(c, maximize=maximize)[1]
    return None if is_identity(p) else p.astype(np.int32, copy=False)


def is_identity(p: np.ndarray | None) -> bool:
    if p is None:
        return True
    a = np.asarray(p)
    return bool(a.size == 0 or (a == np.arange(a.size)).all())


def as_array(p: np.ndarray | None, n: int) -> np.ndarray:
    """Materialise a permutation. Only for code that cannot take None."""
    if p is None:
        return np.arange(n, dtype=np.int32)
    a = np.asarray(p)
    if a.size != n:
        raise ValueError(f"permutation of length {a.size}, expected {n}")
    return a


def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Equality that treats None and arange(n) as the same permutation."""
    if a is None or b is None:
        return is_identity(a) and is_identity(b)
    x, y = np.asarray(a), np.asarray(b)
    return x.shape == y.shape and bool((x == y).all())


def invert(p: np.ndarray | None) -> np.ndarray | None:
    """The permutation that undoes p. None is its own inverse."""
    if p is None:
        return None
    a = np.asarray(p)
    out = np.empty_like(a)
    out[a] = np.arange(a.size, dtype=a.dtype)
    return None if is_identity(out) else out


def compose(first: np.ndarray | None, second: np.ndarray | None
            ) -> np.ndarray | None:
    """The single gather equal to gathering by first, then by second."""
    if first is None:
        return None if second is None else np.asarray(second)
    if second is None:
        return np.asarray(first)
    f, s = np.asarray(first), np.asarray(second)
    if f.size != s.size:
        raise ValueError(f"cannot compose lengths {f.size} and {s.size}")
    out = f[s]
    return None if is_identity(out) else out


def is_permutation(p: np.ndarray | None, n: int | None = None) -> bool:
    """True if p is a bijection of range(len(p)). For validating what we read."""
    if p is None:
        return True
    a = np.asarray(p)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        return False
    if n is not None and a.size != n:
        return False
    return bool((np.sort(a) == np.arange(a.size)).all())


def agreement(p: np.ndarray | None, truth: np.ndarray | None, n: int) -> float:
    """Fraction of units placed correctly. The bench recovery-accuracy metric."""
    return float((as_array(p, n) == as_array(truth, n)).mean())


def pack(p: np.ndarray | None) -> bytes:
    """FileFormat.md 4.2: packed int32 LE, no header. None has no object."""
    if p is None:
        raise ValueError("identity is stored as null, not as a permutation object")
    a = np.asarray(p, dtype="<i4")
    if not is_permutation(a):
        raise ValueError("refusing to store a non-bijective permutation")
    return a.tobytes()


def unpack(blob: bytes) -> np.ndarray:
    if len(blob) % 4:
        raise ValueError(f"{len(blob)} bytes is not a whole number of int32")
    a = np.frombuffer(blob, dtype="<i4")
    if not is_permutation(a):
        raise ValueError("stored permutation is not a bijection")
    return a