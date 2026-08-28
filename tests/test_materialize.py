"""Tests for reconstruction to a real file (synapsefs/materialize.py).

The claim under test is CLI.md ~4's: a commit written back out must be
*byte*-identical to the file that was committed, headers included -- not
numerically close. So the assertions here are `read_bytes() == read_bytes()`
and `max_ulp_diff == 0`, never `np.allclose`.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from synapsefs import graph
from synapsefs.cli.main import main
from synapsefs.errors import IntegrityError
from synapsefs.materialize import (
    compare_sources,
    header_layout,
    materialize,
)
from synapsefs.pack.packset import PackSet
from synapsefs.safetensors_io import SafetensorsFile
from synapsefs.store.repo import Repo

from tests.test_graph import commit_series, make_repo


def open_commit(repo: Repo, commit_hash: str):
    packs = PackSet(repo.objects_dir / "pack", tmp_dir=repo.objects_dir / "tmp")
    return packs, graph.CommitCheckpoint(repo.store, packs, commit_hash)


# -- header_layout ---------------------------------------------------------


def test_header_layout_follows_data_offsets_not_key_order(tmp_path):
    """A header whose JSON key order disagrees with its data_offsets order must
    still round-trip. This is the case a naive `for name in names()` writer
    gets wrong, and it is not hypothetical -- nothing in the safetensors spec
    ties the two orders together."""
    doc = {
        "z": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        "a": {"dtype": "F16", "shape": [2], "data_offsets": [4, 8]},
    }
    blob = json.dumps(doc).encode()
    header = struct.pack("<Q", len(blob)) + blob
    assert header_layout(header) == [("z", 0, 4), ("a", 4, 8)]


def test_header_layout_ignores_metadata_key(tmp_path):
    doc = {
        "__metadata__": {"format": "pt"},
        "a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
    }
    blob = json.dumps(doc).encode()
    header = struct.pack("<Q", len(blob)) + blob
    assert header_layout(header) == [("a", 0, 4)]


def test_non_contiguous_data_section_is_refused(tmp_path):
    """A gap between tensors is data the codec never ingested. Emitting zeros
    there and calling the result identical would be the worst outcome."""
    doc = {
        "a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        "b": {"dtype": "F16", "shape": [2], "data_offsets": [8, 12]},
    }
    blob = json.dumps(doc).encode()
    header = struct.pack("<Q", len(blob)) + blob
    # `source` is never touched: the tiling check runs before any row is read,
    # so None here also asserts that the refusal happens up front rather than
    # halfway through a written file.
    with pytest.raises(IntegrityError, match="not contiguous"):
        materialize(None, header, tmp_path / "x.safetensors")


# -- round trip ------------------------------------------------------------


def test_every_commit_materializes_byte_identically(tmp_path):
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 6)
    repo = Repo.find(repo_dir)

    commits = []
    head = repo.resolve_ref("HEAD")
    while head:
        commit = graph.get_json(repo.store, head)
        commits.append(head)
        head = commit["parents"][0] if commit["parents"] else None
    commits.reverse()

    assert len(commits) == len(paths)
    for commit_hash, source_path in zip(commits, paths):
        packs, view = open_commit(repo, commit_hash)
        try:
            out = tmp_path / f"out_{commit_hash[:8]}.safetensors"
            stats = materialize(view, view.header_bytes, out)
        finally:
            packs.close()
        assert out.read_bytes() == source_path.read_bytes()
        assert stats["total_bytes"] == source_path.stat().st_size


def test_tiny_batch_size_still_reconstructs_exactly(tmp_path):
    """Row batching must not change the output. A one-row batch exercises the
    boundary arithmetic on every single row rather than once per tensor."""
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 2)
    repo = Repo.find(repo_dir)

    packs, view = open_commit(repo, repo.resolve_ref("HEAD"))
    try:
        out = tmp_path / "tiny.safetensors"
        materialize(view, view.header_bytes, out, batch_bytes=1)
    finally:
        packs.close()
    assert out.read_bytes() == paths[-1].read_bytes()


def test_failed_materialize_leaves_no_partial_file(tmp_path):
    """atomic_writer's guarantee, exercised through this module: a decode that
    blows up must not leave a truncated checkpoint behind for torch to load."""
    repo_dir = make_repo(tmp_path)
    commit_series(tmp_path, repo_dir, 1)
    repo = Repo.find(repo_dir)

    packs, view = open_commit(repo, repo.resolve_ref("HEAD"))
    out = tmp_path / "broken.safetensors"
    try:
        class Exploding:
            def names(self): return view.names()
            def spec(self, name): return view.spec(name)
            def rows(self, name, start, stop): raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            materialize(Exploding(), view.header_bytes, out)
    finally:
        packs.close()
    assert not out.exists()


# -- comparison ------------------------------------------------------------


def test_compare_reports_zero_difference_for_a_correct_reconstruction(tmp_path):
    repo_dir = make_repo(tmp_path)
    paths = commit_series(tmp_path, repo_dir, 3)
    repo = Repo.find(repo_dir)

    packs, view = open_commit(repo, repo.resolve_ref("HEAD"))
    try:
        with SafetensorsFile(paths[-1]) as reference:
            results = compare_sources(view, reference)
    finally:
        packs.close()

    assert results
    for item in results:
        assert item.status == "identical", item
        assert item.mismatched == 0
        assert item.max_ulp_diff == 0


def test_compare_measures_a_one_ulp_difference(tmp_path):
    """The point of the ULP column: a single mantissa bit must be visible, and
    an epsilon comparison would round it away at fp16 magnitudes."""
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"

    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    save_file({"w": values}, str(left))

    nudged = values.view(np.uint16).copy()
    nudged[2] += 1                      # one representable step upward
    save_file({"w": nudged.view(np.float16)}, str(right))

    with SafetensorsFile(left) as a, SafetensorsFile(right) as b:
        [result] = compare_sources(a, b)
    assert result.status == "differs"
    assert result.mismatched == 1
    assert result.max_ulp_diff == 1


def test_compare_reports_missing_tensors_rather_than_skipping_them(tmp_path):
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"
    values = np.zeros(4, dtype=np.float16)
    save_file({"shared": values, "only_left": values}, str(left))
    save_file({"shared": values, "only_right": values}, str(right))

    with SafetensorsFile(left) as a, SafetensorsFile(right) as b:
        by_name = {r.name: r.status for r in compare_sources(a, b)}
    assert by_name == {
        "shared": "identical",
        "only_left": "missing_right",
        "only_right": "missing_left",
    }
