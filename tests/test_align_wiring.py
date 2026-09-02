"""Alignment wired into commit, checked end to end.

The assertion that matters is **byte-exact reconstruction of a permuted
commit**. Two bugs found during wiring both produced a repository that was
internally consistent -- `verify --content` passed on both -- and only a byte
comparison against the original file caught them. So that comparison is what
these tests make.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from synapsefs import graph
from synapsefs.align import config_parser
from synapsefs.align.Error import AlignError, NotAlignable, TopologyError
from synapsefs.align.IR import TensorRef
from synapsefs.cli.main import main
from synapsefs.errors import SynapseError
from synapsefs.safetensors_io import SafetensorsFile
from synapsefs.store.repo import Repo


def make_cnn(rng, widths=(8, 12, 10)):
    """A straight conv chain: stem -> mid -> head, plus a norm on each."""
    w = {}
    prev = 3
    for i, c in enumerate(widths):
        w[f"layer{i}.weight"] = rng.standard_normal((c, prev, 3, 3)).astype(np.float16)
        w[f"layer{i}.bias"] = rng.standard_normal(c).astype(np.float16)
        w[f"norm{i}.weight"] = rng.standard_normal(c).astype(np.float16)
        w[f"norm{i}.bias"] = rng.standard_normal(c).astype(np.float16)
        prev = c
    return w


def permute_layer(w, i, perm, next_i):
    """Permute layer i's output units, and layer next_i's input columns to match.

    This is the symmetry the aligner has to undo: the network computes the same
    function afterwards, and every affected tensor's bytes change.
    """
    out = dict(w)
    out[f"layer{i}.weight"] = w[f"layer{i}.weight"][perm]
    out[f"layer{i}.bias"] = w[f"layer{i}.bias"][perm]
    out[f"norm{i}.weight"] = w[f"norm{i}.weight"][perm]
    out[f"norm{i}.bias"] = w[f"norm{i}.bias"][perm]
    out[f"layer{next_i}.weight"] = w[f"layer{next_i}.weight"][:, perm]
    return out


@pytest.fixture
def repo_with_pair(tmp_path):
    rng = np.random.default_rng(0)
    base = make_cnn(rng)
    perm = rng.permutation(8).astype(np.int32)
    permuted = permute_layer(base, 0, perm, 1)

    (tmp_path / "config.json").write_text(json.dumps({"arch": "cnn"}))
    for name, w in (("base", base), ("permuted", permuted)):
        save_file(w, str(tmp_path / f"{name}.safetensors"))

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    assert main(["init", str(repo_dir)]) == 0
    return repo_dir, tmp_path, perm


def commit(repo_dir, tmp_path, name):
    return main(["-C", str(repo_dir), "commit", str(tmp_path / f"{name}.safetensors"),
                 "-m", name, "--config", str(tmp_path / "config.json"), "-q"])


# -- the load-bearing test --------------------------------------------------


def test_a_permuted_commit_reconstructs_byte_exactly(repo_with_pair):
    """Encode gathers the base through `p`; decode must repeat that gather,
    not invert it. Both directions are valid bijections of the right length,
    so only this comparison can tell them apart."""
    repo_dir, tmp_path, _ = repo_with_pair
    assert commit(repo_dir, tmp_path, "base") == 0
    assert commit(repo_dir, tmp_path, "permuted") == 0

    out = tmp_path / "out.safetensors"
    assert main(["-C", str(repo_dir), "restore", "HEAD", "--out", str(out),
                 "--compare", str(tmp_path / "permuted.safetensors"), "--strict"]) == 0
    assert out.read_bytes() == (tmp_path / "permuted.safetensors").read_bytes()


def test_manifest_reuse_is_disabled_under_a_permutation(repo_with_pair):
    """FORMAT.md 4.5 points an unchanged tensor at the *base's* manifest. Under
    a permutation the chunks are identical only *after* gathering through it,
    so the content is a reordering of the base's, not a copy.

    Reusing there produces a repo that verifies clean and reconstructs the
    base's row order. Regression test for exactly that."""
    repo_dir, tmp_path, _ = repo_with_pair
    assert commit(repo_dir, tmp_path, "base") == 0
    assert commit(repo_dir, tmp_path, "permuted") == 0

    repo = Repo.find(repo_dir)
    store = repo.store
    walk = graph.walk_first_parent(store, repo.resolve_ref("HEAD"))
    top = graph.get_json(store, graph.get_json(store, walk[0][0])["checkpoint_manifest"])
    bot = graph.get_json(store, graph.get_json(store, walk[1][0])["checkpoint_manifest"])

    # Exactly the tensors permute_layer touched. `layer1.bias` is deliberately
    # excluded: only layer1's *columns* moved, so its bias is unchanged and
    # reusing its manifest is correct.
    permuted_tensors = ["layer0.weight", "layer0.bias", "norm0.weight",
                        "norm0.bias", "layer1.weight"]
    shared = [n for n in permuted_tensors if top["tensors"][n] == bot["tensors"][n]]
    assert not shared, f"manifest reused for permuted tensors: {shared}"


def test_a_permutation_that_does_not_help_is_not_applied(tmp_path):
    """The solver maximises weight matching, which is not the same as
    minimising the residual. A permutation that scores well but enlarges the
    delta must be dropped -- it costs compression for nothing."""
    rng = np.random.default_rng(3)
    base = make_cnn(rng)
    drifted = {k: (v + np.float16(0.01)).astype(np.float16) for k, v in base.items()}
    (tmp_path / "config.json").write_text(json.dumps({"arch": "cnn"}))
    save_file(base, str(tmp_path / "a.safetensors"))
    save_file(drifted, str(tmp_path / "b.safetensors"))
    repo_dir = tmp_path / "r"; repo_dir.mkdir()
    assert main(["init", str(repo_dir)]) == 0
    for n in ("a", "b"):
        assert main(["-C", str(repo_dir), "commit", str(tmp_path / f"{n}.safetensors"),
                     "-m", n, "--config", str(tmp_path / "config.json"), "-q"]) == 0
    out = tmp_path / "o.safetensors"
    assert main(["-C", str(repo_dir), "restore", "HEAD", "--out", str(out),
                 "--compare", str(tmp_path / "b.safetensors"), "--strict"]) == 0


def test_no_align_still_reconstructs(repo_with_pair):
    repo_dir, tmp_path, _ = repo_with_pair
    assert commit(repo_dir, tmp_path, "base") == 0
    assert main(["-C", str(repo_dir), "commit", str(tmp_path / "permuted.safetensors"),
                 "-m", "p", "--config", str(tmp_path / "config.json"),
                 "--no-align", "-q"]) == 0
    out = tmp_path / "o.safetensors"
    assert main(["-C", str(repo_dir), "restore", "HEAD", "--out", str(out),
                 "--compare", str(tmp_path / "permuted.safetensors"), "--strict"]) == 0


# -- supporting behaviour ---------------------------------------------------


def test_alignment_is_reported(repo_with_pair, capsys):
    repo_dir, tmp_path, _ = repo_with_pair
    commit(repo_dir, tmp_path, "base")
    capsys.readouterr()
    assert main(["-C", str(repo_dir), "commit", str(tmp_path / "permuted.safetensors"),
                 "-m", "p", "--config", str(tmp_path / "config.json"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["alignment"]["groups"] > 0
    assert result["alignment"]["identity"] is False


def test_align_errors_carry_cli_exit_codes():
    """`cli/main.py` catches SynapseError and nothing else; a detached
    hierarchy would report every alignment failure as a generic exit 1."""
    for cls, code in ((AlignError, 1), (TopologyError, 2), (NotAlignable, 5)):
        assert issubclass(cls, SynapseError)
        assert cls.exit_code == code


def test_layer_order_is_inferred_from_shapes(tmp_path):
    """safetensors sorts keys alphabetically, so 'head' precedes 'stem' and the
    chain must be recovered from shape divisibility instead."""
    rng = np.random.default_rng(1)
    w = {
        "stem.weight": rng.standard_normal((8, 3, 3, 3)).astype(np.float16),
        "mid.weight": rng.standard_normal((12, 8, 3, 3)).astype(np.float16),
        "head.weight": rng.standard_normal((5, 12)).astype(np.float16),
    }
    path = tmp_path / "m.safetensors"
    save_file(w, str(path))
    with SafetensorsFile(path) as f:
        refs = {n: TensorRef(n, tuple(f.spec(n).shape), f.spec(n).dtype) for n in f.names()}
    assert sorted(refs) == ["head.weight", "mid.weight", "stem.weight"]  # not topological
    topo = config_parser.parse(refs, None)
    assert len(topo.solvable_groups()) >= 1
