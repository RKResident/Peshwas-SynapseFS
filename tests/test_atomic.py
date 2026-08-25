"""Tests for the atomic write primitive -- the thing the PS's crash-safety
requirement (module 2h) ultimately reduces to.

Two tiers, per the earlier design discussion:

  1. Logical/simulated tests -- fast, run on every commit.
  2. A real SIGKILL test -- spawns a subprocess that's actively writing and
     kills it mid-flight, then checks the filesystem for damage. Slower,
     but it's the actual evidence for "recovery behavior after a simulated
     crash mid-write", not just a description of what should happen.

Run everything:
    pytest tests/test_atomic.py -v

Run only the fast tier, skipping the real subprocess-kill test:
    pytest tests/test_atomic.py -v -m "not slow"
"""

from __future__ import annotations

import multiprocessing
import os
import signal
from pathlib import Path

import pytest

from synapsefs.store.atomic import atomic_write, gc_tmp_dir


# --------------------------------------------------------------------------
# Tier 1: logical correctness, no real crash involved.
# --------------------------------------------------------------------------


def test_write_then_read_back(tmp_path: Path):
    """Basic correctness: what you write is what you get back."""
    target = tmp_path / "objects" / "ab" / "abc123"
    tmp_dir = tmp_path / "objects" / "tmp"
    atomic_write(target, b"hello world", tmp_dir=tmp_dir)
    assert target.read_bytes() == b"hello world"


def test_creates_missing_shard_directory(tmp_path: Path):
    """The sharded object layout (objects/<hh>/<hash>) means the two-char
    shard directory frequently doesn't exist yet -- atomic_write must
    create it, not assume it's there."""
    target = tmp_path / "objects" / "zz" / "somehash"
    tmp_dir = tmp_path / "objects" / "tmp"
    assert not target.parent.exists()
    atomic_write(target, b"data", tmp_dir=tmp_dir)
    assert target.exists()


def test_no_tmp_file_left_behind_on_success(tmp_path: Path):
    """After a successful write, objects/tmp/ should be empty -- the file
    was renamed out of it, not copied."""
    tmp_dir = tmp_path / "objects" / "tmp"
    atomic_write(tmp_path / "objects" / "ab" / "h", b"x", tmp_dir=tmp_dir)
    assert list(tmp_dir.iterdir()) == []


def test_overwrite_replaces_wholesale(tmp_path: Path):
    """Writing to an existing target replaces it entirely via rename, never
    by seeking/truncating the existing file in place."""
    target = tmp_path / "objects" / "ab" / "h"
    tmp_dir = tmp_path / "objects" / "tmp"
    atomic_write(target, b"first", tmp_dir=tmp_dir)
    atomic_write(target, b"second-and-longer", tmp_dir=tmp_dir)
    assert target.read_bytes() == b"second-and-longer"


def test_stale_tmp_file_is_gc_able(tmp_path: Path):
    """Simulates the aftermath of a crash: a file left in objects/tmp/ that
    never got renamed out. gc_tmp_dir must remove it unconditionally --
    nothing in tmp/ is ever a real, addressable object."""
    tmp_dir = tmp_path / "objects" / "tmp"
    tmp_dir.mkdir(parents=True)
    stray = tmp_dir / "leftover.tmp"
    stray.write_bytes(b"\x00" * 10)  # stands in for a torn/incomplete write

    removed = gc_tmp_dir(tmp_dir)

    assert removed == 1
    assert not stray.exists()


def test_gc_tmp_dir_on_missing_directory_is_a_noop(tmp_path: Path):
    """A brand-new repo (objects/tmp/ not created yet) shouldn't crash the
    startup GC."""
    assert gc_tmp_dir(tmp_path / "does-not-exist") == 0


# --------------------------------------------------------------------------
# Tier 2: a real crash.
# --------------------------------------------------------------------------


def _write_loop(objects_dir: str, n: int, ready: "multiprocessing.synchronize.Event") -> None:
    """Child-process target: write `n` objects as fast as possible.

    Re-imports atomic_write inside the child rather than relying on a
    closure, since multiprocessing on some platforms (spawn start method)
    re-executes the module from scratch anyway -- being explicit here keeps
    the test's behavior independent of the platform's default start method.

    Signals `ready` after the *first* object has actually landed on disk
    (post-rename, post-directory-fsync), so the parent never kills us
    before there's real surface area on disk for the invariant to be
    checked against.
    """
    from synapsefs.store.atomic import atomic_write

    objects_dir_p = Path(objects_dir)
    tmp_dir = objects_dir_p / "tmp"
    for i in range(n):
        # Large enough that a single write() syscall is unlikely to always
        # complete the whole payload in one go, so there's real surface
        # area for a kill to land mid-write.
        payload = f"payload-{i}-".encode() * 500
        target = objects_dir_p / f"obj-{i}"
        atomic_write(target, payload, tmp_dir=tmp_dir)
        if i == 0:
            ready.set()


@pytest.mark.slow
def test_kill_mid_write_leaves_no_partial_objects(tmp_path: Path):
    """The real test: SIGKILL a process that is actively calling
    atomic_write() in a tight loop, then assert every object actually
    present on disk is complete and correct -- never truncated.

    This is the direct evidence for CLI.md's "recovery behavior after a
    simulated crash mid-write" and for the PS's "a crash ... must never
    leave the on-disk history corrupted" requirement.
    """
    objects_dir = tmp_path / "objects"
    (objects_dir / "tmp").mkdir(parents=True)

    ready = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_write_loop, args=(str(objects_dir), 4000, ready)
    )
    proc.start()
    # Block until the child has actually landed at least one complete,
    # fsync'd, renamed object -- a fixed sleep can't guarantee that (the
    # child might not even be scheduled yet), which is exactly why this
    # test used to fire "kill happened before any object was written"
    # spuriously. Waiting on the child's own readiness signal instead of
    # guessing a sleep duration is the fix, not a longer sleep.
    if not ready.wait(timeout=10):
        proc.kill()
        proc.join()
        pytest.fail("child never landed a single object within 10s")
    os.kill(proc.pid, signal.SIGKILL)
    proc.join()

    assert proc.exitcode != 0  # sanity: it really was killed, not finished

    checked = 0
    for entry in objects_dir.iterdir():
        if entry.name == "tmp":
            # A torn write CAN legitimately leave a file behind in tmp/ --
            # that's expected and fine; gc_tmp_dir cleans it on next open.
            # It is NOT part of the invariant under test here.
            continue
        content = entry.read_bytes()
        i = int(entry.name.split("-")[1])
        expected = f"payload-{i}-".encode() * 500
        assert content == expected, f"{entry.name} was left partially written"
        checked += 1

    # If literally nothing landed before the kill, the test isn't exercising
    # anything -- widen the sleep above if this ever fires on slower hardware.
    assert checked > 0, "kill happened before any object was written; test is a no-op"
