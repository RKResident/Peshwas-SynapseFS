"""Tests for the fixture generator.

These matter more than ordinary tests. Every accuracy number SynapseFS reports
is measured *against* these fixtures, so a bug here does not fail loudly -- it
silently invalidates the metrics while all the downstream tests keep passing.

Two things are checked:

1. **The hand-written safetensors writer is correct**, verified by reading its
   output back with the real ``safetensors`` library and byte-comparing against
   what ``safetensors.numpy.save_file`` produces for the same input. This is
   what licenses us to bypass the library in order to emit bf16.
2. **Permutation is a symmetry**, verified functionally rather than structurally.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gen_fixtures import (  # noqa: E402
    ALL_VARIANTS,
    CNNSpec,
    MLPSpec,
    build_cnn,
    build_mlp,
    finetune,
    forward_cnn,
    forward_mlp,
    generate,
    permute_cnn,
    permute_mlp,
    save_safetensors,
    to_storage_dtype,
    verify_equivalence,
)

MLP = MLPSpec(in_dim=16, hidden=32, depth=4, out_dim=5)
CNN = CNNSpec(in_ch=3, channels=8, blocks=2, spatial=4, num_classes=5)


# --------------------------------------------------------------------------- #
# safetensors writer
# --------------------------------------------------------------------------- #


def test_writer_matches_reference_implementation(tmp_path: Path):
    """Our writer must produce byte-identical output to safetensors itself.

    If this ever fails, the writer has drifted from the reference and every
    byte-exactness claim built on top of it is suspect.
    """
    from safetensors.numpy import save_file

    t = {
        "layer1.weight": np.arange(8, dtype=np.float16).reshape(4, 2),
        "layer1.bias": np.array([0.5, -0.5, 1.0, -0.0], dtype=np.float16),
        "aaa.first_alphabetically": np.ones(3, dtype=np.float16),
    }
    meta = {"format": "pt"}

    ours, theirs = tmp_path / "ours.st", tmp_path / "theirs.st"
    save_safetensors(ours, t, metadata=meta)
    save_file(t, theirs, metadata=meta)

    assert ours.read_bytes() == theirs.read_bytes()


def test_writer_layout_conventions(tmp_path: Path):
    """The three conventions from docs/FileFormat.md section 1.2, asserted directly."""
    p = tmp_path / "m.safetensors"
    save_safetensors(
        p,
        {
            "z_last": np.ones(2, dtype=np.float16),
            "a_first": np.ones(2, dtype=np.float16),
        },
        metadata={"format": "pt"},
    )
    raw = p.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + n]

    # 3. data region is 8-byte aligned via trailing-space padding
    assert (8 + n) % 8 == 0
    assert header.rstrip(b" ") != header, "header should be space-padded"

    keys = list(json.loads(header.rstrip(b" ")).keys())
    # 1. __metadata__ first
    assert keys[0] == "__metadata__"
    # 2. tensor keys sorted alphabetically, not by insertion order
    assert keys[1:] == sorted(keys[1:]) == ["a_first", "z_last"]


def test_roundtrip_through_real_reader(tmp_path: Path):
    """Everything we write must be readable by the library the graders will use."""
    from safetensors.numpy import load_file

    t = {"w": np.random.default_rng(0).normal(size=(6, 4)).astype(np.float16)}
    p = tmp_path / "m.safetensors"
    save_safetensors(p, t, metadata={"format": "pt"})

    back = load_file(p)
    assert set(back) == set(t)
    assert np.array_equal(back["w"], t["w"])


def test_bf16_is_emitted_as_uint16_payload(tmp_path: Path):
    """bf16 has no numpy dtype, so it rides as uint16 with an explicit declaration."""
    arr = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    conv, declared = to_storage_dtype(arr, "bf16")

    assert declared == "BF16"
    assert conv.dtype == np.uint16

    p = tmp_path / "m.safetensors"
    save_safetensors(p, {"w": conv}, dtype_overrides={"w": "BF16"})

    n = struct.unpack("<Q", p.read_bytes()[:8])[0]
    header = json.loads(p.read_bytes()[8 : 8 + n].rstrip(b" "))
    assert header["w"]["dtype"] == "BF16"
    assert header["w"]["data_offsets"] == [0, 6]      # 3 elements x 2 bytes


def test_signed_zero_survives_storage(tmp_path: Path):
    """-0.0 (0x8000) must not collapse to +0.0 (0x0000).

    Guards the codec invariant from the opposite direction: if the *fixtures*
    ever normalise signed zero, the codec's handling of it is never exercised.
    """
    from safetensors.numpy import load_file

    t = {"b": np.array([0.0, -0.0], dtype=np.float16)}
    p = tmp_path / "m.safetensors"
    save_safetensors(p, t)

    bits = load_file(p)["b"].view(np.uint16)
    assert bits[0] == 0x0000
    assert bits[1] == 0x8000


# --------------------------------------------------------------------------- #
# Permutation is a symmetry
# --------------------------------------------------------------------------- #


def test_mlp_permutation_preserves_function():
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    perm, groups = permute_mlp(MLP, base, rng)

    x = rng.normal(size=(4, MLP.in_dim)).astype(np.float32)
    dev = verify_equivalence(lambda t, xx: forward_mlp(MLP, t, xx), base, perm, x)

    assert dev < 1e-3
    assert set(groups) == {f"g_hidden_{i}" for i in range(MLP.depth - 1)}


def test_cnn_permutation_preserves_function():
    """Covers the two hard cases at once: tied residual groups and block columns."""
    rng = np.random.default_rng(0)
    base = build_cnn(CNN, rng)
    perm, groups = permute_cnn(CNN, base, rng)

    x = rng.normal(size=(2, CNN.in_ch, CNN.spatial, CNN.spatial)).astype(np.float32)
    dev = verify_equivalence(lambda t, xx: forward_cnn(CNN, t, xx), base, perm, x)

    assert dev < 1e-3
    assert "g_skip" in groups
    assert all(f"g_block_{b}" in groups for b in range(CNN.blocks))


def test_permutation_is_actually_non_identity():
    """A generator that silently emitted identity would pass every other test."""
    rng = np.random.default_rng(0)
    _, groups = permute_mlp(MLP, build_mlp(MLP, rng), rng)
    ident = np.arange(MLP.hidden)
    assert any(not np.array_equal(p, ident) for p in groups.values())


def test_cnn_head_columns_move_in_blocks():
    """The conv-to-linear flatten case, asserted structurally.

    This bug is invisible under an identity permutation and invisible in the MLP
    fixture, so it needs its own check: head column block `i` of the permuted
    model must equal block `p[i]` of the base.
    """
    rng = np.random.default_rng(0)
    base = build_cnn(CNN, rng)
    perm, groups = permute_cnn(CNN, base, rng)

    blk = CNN.block_size
    assert blk == CNN.spatial**2 > 1, "flatten head should give a block size > 1"

    p = groups["g_skip"]
    for i, src in enumerate(p):
        got = perm["head.weight"][:, i * blk : (i + 1) * blk]
        want = base["head.weight"][:, src * blk : (src + 1) * blk]
        assert np.array_equal(got, want), f"head column block {i} mismatched"


def test_broken_permutation_is_caught():
    """verify_equivalence must actually fail on a wrong permutation.

    Without this, a vacuous check that always passes would look identical to a
    working one.
    """
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    perm, _ = permute_mlp(MLP, base, rng)

    # Permute one layer's rows without the matching column permutation on the
    # next layer -- the single most likely real mistake.
    broken = dict(perm)
    broken["layers.1.weight"] = broken["layers.1.weight"][:, ::-1]

    x = rng.normal(size=(4, MLP.in_dim)).astype(np.float32)
    with pytest.raises(AssertionError, match="NOT functionally equivalent"):
        verify_equivalence(lambda t, xx: forward_mlp(MLP, t, xx), base, broken, x)


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #


def test_finetune_perturbs_every_tensor_densely():
    """The realistic fine-tune case: every element moves, nothing dedups."""
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    ft = finetune(base, rng, rel_scale=1e-3)

    for k, v in base.items():
        if v.std() == 0:
            continue                                  # zero-init biases
        changed = np.count_nonzero(ft[k] != v) / v.size
        assert changed > 0.99, f"{k}: only {changed:.1%} of elements moved"


def test_finetune_density_after_fp16_quantisation():
    """Record how many deltas survive narrowing to fp16 -- the codec's real input.

    ``test_finetune_perturbs_every_tensor_densely`` measures the float32 working
    array, where essentially every element moves. On disk the picture differs:
    fp16 carries ~11 mantissa bits, so a relative perturbation near 1e-3 falls
    below one ULP for a sizeable fraction of weights and rounds away entirely.

    Measured at ``rel_scale=1e-3``: roughly 80% of elements change, so ~20% of
    the codec's deltas are exactly zero before compression even starts. That is
    real fp16 fine-tuning behaviour and it is good for the residual ratio -- but
    the compression team should know the zeros are there rather than discovering
    them while debugging a suspiciously good benchmark.

    The bounds here are deliberately loose. This test documents a property; it
    is not a threshold anyone should tune against.
    """
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    ft = finetune(base, rng, rel_scale=1e-3)

    w = "layers.1.weight"
    a = to_storage_dtype(base[w], "fp16")[0]
    b = to_storage_dtype(ft[w], "fp16")[0]

    changed = np.count_nonzero(a != b) / a.size
    assert 0.5 < changed < 1.0, (
        f"{changed:.1%} of fp16 elements changed; expected a substantial but "
        f"partial fraction. If this hits 100%, the noise scale grew and the "
        f"fixture no longer models a light fine-tuning step."
    )


def test_finetune_stays_small():
    """A fine-tune step should be a small delta, or the residual metric is fiction."""
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    ft = finetune(base, rng, rel_scale=1e-3)

    w = "layers.1.weight"
    rel = np.abs(ft[w] - base[w]).max() / np.abs(base[w]).max()
    assert rel < 0.05


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_generate_writes_a_complete_fixture_set(tmp_path: Path, arch: str):
    spec = MLP if arch == "mlp" else CNN
    m = generate(arch, spec, tmp_path / arch, seed=7, dtype="fp16",
                 variants=ALL_VARIANTS, rel_scale=1e-3)

    for v in ALL_VARIANTS:
        d = tmp_path / arch / v
        assert (d / "model.safetensors").is_file()
        assert (d / "config.json").is_file()

    assert (tmp_path / arch / "MANIFEST.json").is_file()
    assert m["equivalence_max_rel_deviation"]["permuted"] < 1e-3


def test_ground_truth_is_separate_from_config(tmp_path: Path):
    """The answer key must never leak into the file the aligner is allowed to read."""
    generate("mlp", MLP, tmp_path, seed=0, dtype="fp16",
             variants=["base", "permuted"], rel_scale=1e-3)

    cfg = json.loads((tmp_path / "permuted" / "config.json").read_text())
    flat = json.dumps(cfg).lower()
    assert "permut" not in flat, "config.json must not hint at the permutation"

    gt = json.loads((tmp_path / "permuted" / "ground_truth.json").read_text())
    assert "permutations" in gt
    assert (tmp_path / "base" / "ground_truth.json").exists() is False


def test_regeneration_is_deterministic(tmp_path: Path):
    """Same seed, same bytes -- this is what lets fixtures stay out of git."""
    for run in ("a", "b"):
        generate("mlp", MLP, tmp_path / run, seed=42, dtype="fp16",
                 variants=["base", "permuted", "finetuned"], rel_scale=1e-3)

    for v in ("base", "permuted", "finetuned"):
        a = (tmp_path / "a" / v / "model.safetensors").read_bytes()
        b = (tmp_path / "b" / v / "model.safetensors").read_bytes()
        assert a == b, f"{v} is not reproducible from its seed"


def test_not_alignable_variant_is_genuinely_different():
    """It should share the architecture but none of the values."""
    rng = np.random.default_rng(0)
    base = build_mlp(MLP, rng)
    other = build_mlp(MLP, np.random.default_rng(9999))

    assert set(base) == set(other)
    w = "layers.1.weight"
    assert base[w].shape == other[w].shape
    assert not np.allclose(base[w], other[w])
