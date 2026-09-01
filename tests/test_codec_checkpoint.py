"""Tests for `synapsefs.codec.checkpoint.encode_checkpoint`.

Coverage is organized around the defects this module replaces
(`synapsefs/codec/checkpoint_iter.py`, deleted -- it never successfully ran)
and the contract laid out in its docstring:

- The real encode/decode pipeline round-trips bit-exactly, including through
  base-diffing, dedup, and the pack-writer-facing `emit` callback.
- `original_bytes` is true tensor bytes, never the encoded-stream length
  (the old code's `total_original_bytes += plain_len` bug).
- `row_start`/`row_end` are inclusive (FORMAT.md 7.2) even though every
  slice used internally is half-open -- the single highest-risk off-by-one
  in this module, pinned explicitly.
- Dedup counts and behaves correctly, both within a run and against a
  caller-supplied `already_have` predicate.
- `emit` bounds memory: at most one payload is alive at a time, never a
  list of them (the old code's `chunk_records.append(...)` bug).
- A tensor missing from the base, or present with a different shape/dtype,
  is handled -- not raised -- and noted.
- 1-D and 0-d tensors are not special-cased; degenerate (zero-sized) shapes
  do not crash.
- Manifests are JSON-serialisable and match FORMAT.md section 7's field set.
"""

from __future__ import annotations

import dataclasses
import json
import struct
import weakref
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from safetensors.numpy import save_file

import synapsefs.codec.checkpoint as checkpoint_mod
from synapsefs.codec.checkpoint import (
    DEFAULT_CHUNK_SIZE_BYTES,
    ChunkRecord,
    encode_checkpoint,
)
from synapsefs.codec.chunk import (
    DELTA_SHUFFLE, RAW, RAW_SHUFFLE_ZSTD, decode_chunk, is_delta,
)
from synapsefs.safetensors_io import SafetensorsFile

FIXTURE_ROOT = Path("fixtures")

MANIFEST_FIELDS = {
    "content_hash",
    "name",
    "dtype",
    "shape",
    "base_tensor_manifest",
    "base_row_permutation",
    "base_col_permutation",
    "col_block_size",
    "chunks",
}
CHUNK_ENTRY_FIELDS = {"row_start", "row_end", "encoding", "object",
                      "plain_len", "stored_checksum"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _write_bf16_file(path: Path, arrays: dict, metadata: Optional[dict] = None) -> None:
    """Write a `.safetensors` file with BF16 tensors by hand.

    Copied from `tests/test_safetensors_io.py`'s helper of the same name
    (that module documents it as intentionally self-contained rather than
    importable, and this file is out of scope to change it in) -- numpy has
    no bfloat16 dtype, so `safetensors.numpy.save_file` cannot emit one.
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    offset = 0
    blobs = []
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        assert arr.dtype == np.uint16, "bf16 fixtures are carried as raw uint16"
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blobs.append(arr.tobytes())
        offset += nbytes

    blob = json.dumps(header).encode("utf-8")
    pad = (-(8 + len(blob))) % 8
    blob += b" " * pad

    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)


def _float32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _reconstruct_all(
    target: SafetensorsFile,
    base: Optional[SafetensorsFile],
    manifests: dict,
    hash_to_payload: dict,
) -> dict:
    """Rebuild every tensor's raw bytes from its manifest + emitted chunks,
    using the base reader for any `delta-zigzag-zstd` chunk's base rows.

    Works whether or not dedup dropped a chunk from `emit`, because dedup
    only ever drops a chunk whose hash was *already* emitted earlier in the
    same run (or already known to the store) -- so any hash appearing in a
    manifest that was actually emitted this run is present in
    `hash_to_payload`. Tests that also exercise `already_have=lambda h:
    True` do not reconstruct, since nothing is emitted in that case.
    """
    out = {}
    for name, manifest in manifests.items():
        spec = target.spec(name)
        parts = []
        for entry in manifest["chunks"]:
            payload = hash_to_payload[entry["object"]]
            base_rows = None
            if is_delta(entry["encoding"]):
                assert base is not None
                base_rows = base.rows(name, entry["row_start"], entry["row_end"] + 1)
            parts.append(
                decode_chunk(entry["encoding"], payload, base_rows, dtype=spec.dtype)
            )
        if parts:
            out[name] = np.concatenate(parts).tobytes()
        else:
            out[name] = b""
    return out


def _collect(records: dict):
    def emit(record: ChunkRecord) -> None:
        records[record.content_hash.hex()] = record.payload

    return emit


# --------------------------------------------------------------------------
# 1. Round-trip through the real pipeline (the test that matters most)
# --------------------------------------------------------------------------


def test_round_trip_through_real_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    base_w = rng.standard_normal((17, 5)).astype(np.float32)
    target_w = base_w + rng.standard_normal((17, 5)).astype(np.float32) * 0.01
    unchanged = rng.standard_normal((6, 3)).astype(np.float32)

    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"w": base_w, "same": unchanged}, str(base_path))
    save_file({"w": target_w, "same": unchanged}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(
        target_path,
        base_path,
        emit=_collect(records),
        chunk_size_bytes=64,  # forces multiple chunks per tensor
    )

    with SafetensorsFile(target_path) as tgt, SafetensorsFile(base_path) as base:
        got = _reconstruct_all(tgt, base, result.manifests, records)
        for name in ("w", "same"):
            assert got[name] == tgt.whole(name).tobytes(), name

    assert result.tensors == 2
    assert set(result.manifests) == {"w", "same"}


# --------------------------------------------------------------------------
# 2. original_bytes is true tensor bytes, not sum(plain_len) -- defect 2
# --------------------------------------------------------------------------


def test_original_bytes_is_true_tensor_bytes_not_plain_len(tmp_path):
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    target_path = tmp_path / "target.safetensors"
    # Two tensors with identical content -> one gets deduped, so the sum of
    # *emitted* plain_len undercounts true tensor bytes if dedup happened
    # (the old code's bug: it only ever accumulated per-chunk plain_len, and
    # had no dedup to begin with, so this collapses defect 2 and defect 6
    # into one regression).
    save_file({"a": arr, "b": arr.copy()}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(
        target_path, None, emit=_collect(records), chunk_size_bytes=DEFAULT_CHUNK_SIZE_BYTES
    )

    with SafetensorsFile(target_path) as tgt:
        expected = tgt.spec("a").nbytes + tgt.spec("b").nbytes
    assert result.original_bytes == expected

    # `records` (via `_collect`) only keeps payload bytes; re-run with a
    # plain_len-collecting emit to get the quantity defect 2 got wrong.
    plain_lens = []
    encode_checkpoint(
        target_path,
        None,
        emit=lambda rec: plain_lens.append(rec.plain_len),
        chunk_size_bytes=DEFAULT_CHUNK_SIZE_BYTES,
    )
    assert sum(plain_lens) < result.original_bytes
    assert result.chunks_deduped == 1
    assert result.chunks_new == 1


# --------------------------------------------------------------------------
# 3. Chunk boundaries -- inclusive row_end, the FORMAT.md 7.2 seam
# --------------------------------------------------------------------------


def test_chunk_boundaries_inclusive_and_contiguous(tmp_path):
    n_rows = 10
    row_elems = 4
    arr = np.arange(n_rows * row_elems, dtype=np.float32).reshape(n_rows, row_elems)
    target_path = tmp_path / "target.safetensors"
    save_file({"w": arr}, str(target_path))

    row_nbytes = row_elems * 4  # float32
    k = 3  # rows per chunk, deliberately not a divisor of n_rows
    chunk_size_bytes = k * row_nbytes

    result = encode_checkpoint(
        target_path, None, emit=lambda rec: None, chunk_size_bytes=chunk_size_bytes
    )
    chunks = result.manifests["w"]["chunks"]

    assert len(chunks) == -(-n_rows // k)  # ceil(n/k) == 4
    assert [c["row_start"] for c in chunks] == [0, 3, 6, 9]
    assert [c["row_end"] for c in chunks] == [2, 5, 8, 9]

    # Inclusive, contiguous, no gaps or overlaps, covers [0, n-1] exactly.
    covered = 0
    prev_end = -1
    for c in chunks:
        assert c["row_start"] == prev_end + 1
        assert c["row_end"] >= c["row_start"]
        covered += c["row_end"] - c["row_start"] + 1
        prev_end = c["row_end"]
    assert covered == n_rows
    assert chunks[-1]["row_end"] == n_rows - 1


# --------------------------------------------------------------------------
# 4. Dedup
# --------------------------------------------------------------------------


def test_dedup_within_run_emits_once_but_both_manifests_reference_hash(tmp_path):
    arr = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    target_path = tmp_path / "target.safetensors"
    save_file({"a": arr, "b": arr.copy()}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(target_path, None, emit=_collect(records))

    assert result.chunks_new == 1
    assert result.chunks_deduped == 1
    assert len(records) == 1

    hash_a = result.manifests["a"]["chunks"][0]["object"]
    hash_b = result.manifests["b"]["chunks"][0]["object"]
    assert hash_a == hash_b
    assert hash_a in records


def test_already_have_suppresses_all_emission(tmp_path):
    arr = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    target_path = tmp_path / "target.safetensors"
    save_file({"a": arr, "b": arr.copy()}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(
        target_path, None, emit=_collect(records), already_have=lambda h: True
    )

    assert records == {}
    assert result.chunks_new == 0
    assert result.chunks_deduped == 2
    # Manifests are unaffected by dedup source.
    assert len(result.manifests["a"]["chunks"]) == 1
    assert len(result.manifests["b"]["chunks"]) == 1
    assert result.manifests["a"]["chunks"][0]["object"] == result.manifests["b"]["chunks"][0]["object"]


# --------------------------------------------------------------------------
# 5. Bounded memory -- the actual fix for defect 5
# --------------------------------------------------------------------------


def test_emit_never_retains_more_than_one_payload_at_a_time(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((200, 32)).astype(np.float32)
    target_path = tmp_path / "target.safetensors"
    save_file({"w": arr}, str(target_path))

    row_nbytes = 32 * 4
    chunk_size_bytes = row_nbytes * 4  # -> ~50 chunks for 200 rows

    live = 0
    peak = 0

    class _Tracked:
        """Wraps a payload so its lifetime can be weakly observed.

        Not a `bytes` subclass: CPython does not support adding a
        `__weakref__` slot to a subclass of a variable-length builtin type
        like `bytes` (`__slots__` on such a subclass raises `TypeError`, and
        a plain subclass gets no weakref slot either). A small wrapper with
        `__len__` is a transparent enough stand-in, since `checkpoint.py`
        only ever calls `len()` on a payload (`EncodedChunk.stored_len`) or
        passes it through untouched.
        """

        def __init__(self, data: bytes) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

    def _track(payload: bytes) -> _Tracked:
        nonlocal live, peak
        tb = _Tracked(payload)
        live += 1
        peak = max(peak, live)

        def _on_gc():
            nonlocal live
            live -= 1

        weakref.finalize(tb, _on_gc)
        return tb

    real_encode_chunk = checkpoint_mod.encode_chunk

    def tracked_encode_chunk(*args, **kwargs):
        encoded = real_encode_chunk(*args, **kwargs)
        return dataclasses.replace(encoded, payload=_track(encoded.payload))

    monkeypatch.setattr(checkpoint_mod, "encode_chunk", tracked_encode_chunk)

    sizes = []

    def emit(record: ChunkRecord) -> None:
        # Exactly what the brief prescribes: record the length and drop the
        # reference. `record` itself goes out of scope when this returns.
        sizes.append(len(record.payload))

    result = encode_checkpoint(
        target_path, None, emit=emit, chunk_size_bytes=chunk_size_bytes, compress_raw=False
    )

    assert len(sizes) > 10  # comfortably more than one chunk
    assert sum(sizes) > chunk_size_bytes * 10  # total payload >> one chunk

    # The real pin: at no point were more than a couple of tracked payloads
    # alive simultaneously (1 in steady state, transiently 2 across a
    # variable reassignment) -- never proportional to the chunk count. The
    # old implementation, which appended every payload to a list, would have
    # a peak equal to len(sizes).
    assert peak <= 2, f"peak retained payloads was {peak}, expected <= 2 (got {len(sizes)} chunks total)"
    assert result.chunks_new == len(sizes)


# --------------------------------------------------------------------------
# 6. Root commit (no base)
# --------------------------------------------------------------------------


def test_root_commit_all_chunks_raw_zstd_and_round_trips(tmp_path):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    target_path = tmp_path / "target.safetensors"
    save_file({"w": arr}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(
        target_path, None, emit=_collect(records), chunk_size_bytes=64
    )
    for c in result.manifests["w"]["chunks"]:
        assert c["encoding"] == RAW_SHUFFLE_ZSTD

    with SafetensorsFile(target_path) as tgt:
        got = _reconstruct_all(tgt, None, result.manifests, records)
        assert got["w"] == tgt.whole("w").tobytes()


def test_root_commit_compress_raw_false_uses_raw_encoding(tmp_path):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    target_path = tmp_path / "target.safetensors"
    save_file({"w": arr}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(
        target_path, None, emit=_collect(records), chunk_size_bytes=64, compress_raw=False
    )
    for c in result.manifests["w"]["chunks"]:
        assert c["encoding"] == RAW

    with SafetensorsFile(target_path) as tgt:
        got = _reconstruct_all(tgt, None, result.manifests, records)
        assert got["w"] == tgt.whole("w").tobytes()


# --------------------------------------------------------------------------
# 7 & 8. Base handling judgement calls
# --------------------------------------------------------------------------


def test_tensor_absent_from_base_does_not_raise_and_is_noted(tmp_path):
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"a": np.zeros((4, 4), dtype=np.float32)}, str(base_path))
    save_file(
        {
            "a": np.zeros((4, 4), dtype=np.float32),
            "new_head": np.ones((3, 3), dtype=np.float32),
        },
        str(target_path),
    )

    records: dict = {}
    result = encode_checkpoint(target_path, base_path, emit=_collect(records))

    assert any("new_head" in note for note in result.notes)
    with SafetensorsFile(target_path) as tgt:
        got = _reconstruct_all(tgt, None, {"new_head": result.manifests["new_head"]}, records)
        assert got["new_head"] == tgt.whole("new_head").tobytes()


def test_tensor_changed_shape_between_base_and_target_does_not_raise(tmp_path):
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"w": np.zeros((4, 4), dtype=np.float32)}, str(base_path))
    save_file({"w": np.ones((6, 4), dtype=np.float32)}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(target_path, base_path, emit=_collect(records))

    assert any("w" in note for note in result.notes)
    for c in result.manifests["w"]["chunks"]:
        assert c["encoding"] in (RAW, RAW_SHUFFLE_ZSTD)

    with SafetensorsFile(target_path) as tgt:
        got = _reconstruct_all(tgt, None, result.manifests, records)
        assert got["w"] == tgt.whole("w").tobytes()


def test_tensor_changed_dtype_between_base_and_target_does_not_raise(tmp_path):
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"w": np.zeros((4, 4), dtype=np.float32)}, str(base_path))
    save_file({"w": np.ones((4, 4), dtype=np.float16)}, str(target_path))

    records: dict = {}
    result = encode_checkpoint(target_path, base_path, emit=_collect(records))

    assert any("w" in note for note in result.notes)
    with SafetensorsFile(target_path) as tgt:
        got = _reconstruct_all(tgt, None, result.manifests, records)
        assert got["w"] == tgt.whole("w").tobytes()


# --------------------------------------------------------------------------
# 9. 1-D and 0-d tensors, not special-cased
# --------------------------------------------------------------------------


def test_1d_and_0d_tensors_round_trip(tmp_path):
    target_path = tmp_path / "target.safetensors"
    save_file(
        {
            "vec": np.arange(10, dtype=np.float32),
            "scalar": np.array(3.14, dtype=np.float32),
        },
        str(target_path),
    )

    records: dict = {}
    result = encode_checkpoint(target_path, None, emit=_collect(records), chunk_size_bytes=8)

    with SafetensorsFile(target_path) as tgt:
        assert tgt.spec("vec").row_elems == 1
        assert tgt.spec("scalar").num_rows == 1
        got = _reconstruct_all(tgt, None, result.manifests, records)
        assert got["vec"] == tgt.whole("vec").tobytes()
        assert got["scalar"] == tgt.whole("scalar").tobytes()

    # Not special-cased: a 10-row 1-D tensor chunked at 8 bytes/chunk (2
    # rows/chunk, since row_elems == 1 -> row_nbytes == 4) still partitions
    # normally into multiple chunks rather than being treated as one blob.
    assert len(result.manifests["vec"]["chunks"]) > 1


# --------------------------------------------------------------------------
# 10. Degenerate shapes containing a 0
# --------------------------------------------------------------------------


def test_degenerate_zero_shapes_do_not_crash(tmp_path):
    target_path = tmp_path / "target.safetensors"
    save_file(
        {
            "empty_rows": np.zeros((0, 4), dtype=np.float32),
            "empty_cols": np.zeros((4, 0), dtype=np.float32),
            "normal": np.ones((2, 2), dtype=np.float32),
        },
        str(target_path),
    )

    result = encode_checkpoint(target_path, None, emit=lambda rec: None)

    assert result.manifests["empty_rows"]["chunks"] == []
    assert result.manifests["empty_cols"]["chunks"] == []
    assert result.manifests["empty_rows"]["shape"] == [0, 4]
    assert result.manifests["empty_cols"]["shape"] == [4, 0]
    assert len(result.manifests["normal"]["chunks"]) >= 1


# --------------------------------------------------------------------------
# 11. bf16 end-to-end
# --------------------------------------------------------------------------


def test_bf16_checkpoint_round_trips(tmp_path):
    rng = np.random.default_rng(2)
    base_f32 = rng.standard_normal((9, 3)).astype(np.float32)
    target_f32 = base_f32 + rng.standard_normal((9, 3)).astype(np.float32) * 0.1
    base_bits = _float32_to_bf16_bits(base_f32)
    target_bits = _float32_to_bf16_bits(target_f32)

    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    _write_bf16_file(base_path, {"w": base_bits})
    _write_bf16_file(target_path, {"w": target_bits})

    records: dict = {}
    result = encode_checkpoint(target_path, base_path, emit=_collect(records), chunk_size_bytes=8)

    with SafetensorsFile(target_path) as tgt, SafetensorsFile(base_path) as base:
        got = _reconstruct_all(tgt, base, result.manifests, records)
        assert got["w"] == tgt.whole("w").tobytes()


# --------------------------------------------------------------------------
# 12. Manifests are JSON-serialisable and match FORMAT.md section 7
# --------------------------------------------------------------------------


def test_manifests_are_json_serialisable_and_match_format_md_fields(tmp_path):
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"w": np.zeros((4, 4), dtype=np.float32)}, str(base_path))
    save_file({"w": np.ones((4, 4), dtype=np.float32)}, str(target_path))

    result = encode_checkpoint(target_path, base_path, emit=lambda rec: None, chunk_size_bytes=16)

    json.dumps(result.manifests)  # must not raise

    manifest = result.manifests["w"]
    assert set(manifest.keys()) == MANIFEST_FIELDS
    assert manifest["dtype"] == "F32"  # verbatim safetensors name, not lowercase
    assert isinstance(manifest["shape"], list)
    for entry in manifest["chunks"]:
        assert set(entry.keys()) == CHUNK_ENTRY_FIELDS
        assert isinstance(entry["object"], str)
        bytes.fromhex(entry["object"])  # valid hex


def test_result_notes_and_header_bytes_types(tmp_path):
    target_path = tmp_path / "target.safetensors"
    save_file({"w": np.ones((2, 2), dtype=np.float32)}, str(target_path))

    with open(target_path, "rb") as fh:
        (header_len,) = struct.unpack_from("<Q", fh.read(8))
        fh.seek(0)
        on_disk_header = fh.read(8 + header_len)

    result = encode_checkpoint(target_path, None, emit=lambda rec: None)
    assert result.header_bytes == on_disk_header
    assert isinstance(result.notes, list)


# --------------------------------------------------------------------------
# 13. Real fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["mlp-tiny", "cnn-tiny"])
def test_real_fixture_round_trips(model):
    base_path = FIXTURE_ROOT / model / "base" / "model.safetensors"
    target_path = FIXTURE_ROOT / model / "finetuned" / "model.safetensors"
    if not base_path.exists() or not target_path.exists():
        pytest.skip(f"fixtures/{model} not present (fresh clone without fixtures)")

    records: dict = {}
    result = encode_checkpoint(target_path, base_path, emit=_collect(records))

    with SafetensorsFile(target_path) as tgt, SafetensorsFile(base_path) as base:
        got = _reconstruct_all(tgt, base, result.manifests, records)
        for name in tgt.names():
            assert got[name] == tgt.whole(name).tobytes(), name

    with SafetensorsFile(target_path) as tgt:
        assert result.tensors == len(tgt.names())
    json.dumps(result.manifests)


def test_stored_bytes_excludes_deduped_chunks(tmp_path):
    """`stored_bytes` must be what the commit actually costs on disk.

    Counting deduped chunks makes the numerator blind to the one thing dedup
    exists for: with four byte-identical tensors it overstated the real cost
    fourfold, so `residual_ratio` read 92% where the true figure was 23%.
    """
    path = tmp_path / "dup.safetensors"
    w = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float16)
    save_file({"a": w, "b": w.copy(), "c": w.copy(), "e": w.copy()}, str(path))

    emitted = []
    result = encode_checkpoint(path, None, emit=lambda rec: emitted.append(len(rec.payload)))

    assert result.chunks_new == 1 and result.chunks_deduped == 3
    assert result.stored_bytes == sum(emitted)
    # and the saving is reported rather than silently folded away
    assert result.deduped_bytes == 3 * result.stored_bytes
    assert result.stored_bytes + result.deduped_bytes == 4 * sum(emitted)


def test_stored_bytes_excludes_chunks_the_store_already_has(tmp_path):
    """Same rule for the `already_have` path: a chunk the object store already
    holds from an earlier commit adds nothing to this one."""
    path = tmp_path / "m.safetensors"
    save_file({"w": np.arange(256, dtype=np.float16).reshape(16, 16)}, str(path))

    emitted = []
    result = encode_checkpoint(
        path, None,
        emit=lambda rec: emitted.append(rec),
        already_have=lambda h: True,
    )

    assert emitted == []
    assert result.stored_bytes == 0
    assert result.chunks_new == 0 and result.chunks_deduped > 0
    assert result.deduped_bytes > 0
    # The manifest is unaffected -- dedup changes storage, never the manifest.
    assert result.manifests["w"]["chunks"]


# --------------------------------------------------------------------------
# 9. The reuse pointer is never null (regression)
# --------------------------------------------------------------------------


def test_an_all_zero_unchanged_tensor_reuses_a_real_manifest_hash(tmp_path):
    """`reused_manifests` must never carry a null hash.

    Regression for a silent repository corruption. An all-zero tensor that
    does not change between commits -- a bias-free conv layer's zero bias, or
    any buffer a frozen layer leaves behind -- is byte-identical to its base,
    so FORMAT.md 4.5's reuse rule applies. But its chunks encode *raw*, not
    delta: an all-zero raw stream and an all-zero residual land within a byte
    of each other once zstd's frame overhead dominates (measured on a real
    448-element frozen bias: raw 19, escape 20), and `encode_chunk`'s raw
    fallback takes the smaller one while deliberately keeping `is_identical`
    -- that flag describes the content, not the encoding.

    `encode_checkpoint` then nulled the base pointer, because no chunk was a
    delta, and wrote that null into `reused_manifests`, where it means
    something else entirely: the identity of the manifest to reuse. `commit`
    reported success and `restore`/`verify`/`log` crashed on every commit
    from that point on.

    Found on a real 24-epoch CNN run, where 6 of 24 commits -- including
    HEAD -- became unrestorable. Not caught by the MLP fixtures: nothing in
    them ever freezes, so no tensor is ever both unchanged *and* raw-encoded.
    """
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"

    frozen = np.zeros(448, dtype=np.float16)      # the bias-free conv layer
    moving = np.arange(64, dtype=np.float16).reshape(8, 8)

    save_file({"frozen": frozen, "moving": moving}, str(base_path))
    save_file({"frozen": frozen.copy(), "moving": moving + np.float16(1)},
              str(target_path))

    base_manifests = {"frozen": "a" * 64, "moving": "b" * 64}
    result = encode_checkpoint(
        target_path, base_path, emit=_collect({}), base_manifests=base_manifests,
    )

    # The precondition this test exists for: unchanged, and stored raw.
    assert "frozen" in result.reused_manifests, (
        "an unchanged tensor must take the reuse path"
    )
    # And the property that was broken: a reuse pointer is a real hash.
    for name, manifest_hash in result.reused_manifests.items():
        assert manifest_hash is not None, f"{name} reuses a null manifest hash"
    assert result.reused_manifests["frozen"] == base_manifests["frozen"]


def test_a_changed_all_raw_tensor_still_drops_its_unused_base_pointer(tmp_path):
    """The other half of the same branch, so the fix cannot overshoot.

    When a tensor *did* change but every chunk still fell back to raw, the
    manifest we write must not name a base no chunk references -- that would
    make reconstruction walk into a manifest contributing zero bytes and force
    GC to retain the subtree behind it. Only the reuse path is exempt.
    """
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"

    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 32)).astype(np.float16)
    # Unrelated content: the residual cannot beat storing the target raw.
    target = (rng.standard_normal((32, 32)) * 1000).astype(np.float16)

    save_file({"w": base}, str(base_path))
    save_file({"w": target}, str(target_path))

    result = encode_checkpoint(
        target_path, base_path, emit=_collect({}),
        base_manifests={"w": "c" * 64},
    )

    manifest = result.manifests["w"]
    if not any(is_delta(c["encoding"]) for c in manifest["chunks"]):
        assert manifest["base_tensor_manifest"] is None, (
            "a manifest whose chunks are all raw must not name a base"
        )
