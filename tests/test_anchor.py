"""Tests for synapsefs/anchor.py -- the per-tensor anchor policy.

A truth table over (ratio, depth), swept as data rather than against a
repository: `decide` is a pure function of two numbers, and every claim it
makes has to hold without ever touching disk.
"""

from __future__ import annotations

import pytest

from synapsefs.anchor import ANCHOR, DELTA, decide


def test_cheap_shallow_delta_stays_delta():
    assert decide(0.1, 0) is DELTA
    assert decide(0.5, 1) is DELTA


def test_ratio_at_or_above_tau_anchors():
    assert decide(0.9, 0) is ANCHOR   # exactly at tau: not strictly better than raw
    assert decide(0.95, 0) is ANCHOR
    assert decide(1.0, 0) is ANCHOR   # delta cost == raw cost
    assert decide(1.4, 0) is ANCHOR   # delta cost > raw cost


def test_ratio_just_under_tau_stays_delta():
    assert decide(0.899999, 0) is DELTA


def test_depth_at_or_above_max_depth_anchors_regardless_of_ratio():
    """The FUSE-read-latency guardrail: even a free delta must not extend the
    chain past `max_depth`."""
    assert decide(0.01, 3) is ANCHOR
    assert decide(0.0, 5) is ANCHOR


def test_depth_just_under_max_depth_stays_delta_if_cheap():
    assert decide(0.5, 2) is DELTA


@pytest.mark.parametrize("tau,depth_ok,depth_bad", [
    (0.5, 2, 4),
    (0.99, 2, 4),
])
def test_custom_tau_and_max_depth_are_honoured(tau, depth_ok, depth_bad):
    assert decide(tau - 0.01, depth_ok, tau=tau, max_depth=4) is DELTA
    assert decide(tau - 0.01, depth_bad, tau=tau, max_depth=4) is ANCHOR
    assert decide(tau, depth_ok, tau=tau, max_depth=4) is ANCHOR


def test_defaults_match_the_documented_break_even_reasoning():
    """Pinned to the same values `align.residual.NOT_ALIGNABLE_THRESHOLD` and
    `graph.REBASE_INTERVAL` use, per anchor.py's module docstring -- if either
    default drifts, this is the test that should catch it, not a bench run."""
    from synapsefs.anchor import DEFAULT_MAX_DEPTH, DEFAULT_TAU
    assert DEFAULT_TAU == 0.9
    assert DEFAULT_MAX_DEPTH == 3


def test_zero_ratio_infinite_headroom_stays_delta_until_depth_bites():
    for depth in range(DEFAULT_MAX_DEPTH_FOR_TEST := 3):
        assert decide(0.0, depth) is DELTA
    assert decide(0.0, DEFAULT_MAX_DEPTH_FOR_TEST) is ANCHOR
