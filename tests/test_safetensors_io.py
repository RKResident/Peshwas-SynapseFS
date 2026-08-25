"""Tests for `synapsefs.safetensors_io.SafetensorsFile`.

Coverage is organized to match the reader's actual contract (see the module
docstring in `synapsefs/safetensors_io.py`):

- Correctness against the reference `safetensors` implementation, for every
  dtype it *can* read.
- bf16, which the reference implementation cannot read at all -- the reason
  this module exists.
- Row semantics (`rows`, `whole`, `gather_rows`) and zero-copy.
- Header byte-exactness and `__metadata__` handling (FileFormat.md section 1).
- Validation of malformed input (`UsageError`, never `IntegrityError` --
  these are user-supplied files, not repo-internal data).
- End-to-end agreement with the chunk codec on a real fixture pair.
- Context-manager lifecycle.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from synapsefs.codec.chunk import decode_chunk, encode_chunk
from synapsefs.errors import UsageError
from synapsefs.safetensors_io import SafetensorsFile

FIXTURE_BASE = Path("fixtures/mlp-tiny/base/model.safetensors")
FIXTURE_FINETUNED = Path("fixtures/mlp-tiny/finetuned/model.safetensors")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _write_bf16_file(path: Path, arrays: dict, metadata: dict | None = None) -> None:
    """Write a `.safetensors` file with BF16 tensors by hand.

    Mirrors what `tools/gen_fixtures.py` does (and for the same reason: numpy
    has no bfloat16 dtype, so `safetensors.numpy.save_file` cannot emit one).
    Not imported from `gen_fixtures.py` -- that module is out of scope for
    this change and this helper is intentionally self-contained.
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
    """Truncate float32 to bf16 bit patterns, as `gen_fixtures.py` does."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


_UNSIGNED_OF = {
    "float16": np.uint16,
    "float32": np.uint32,
    "int64": np.uint64,
}


# --------------------------------------------------------------------------
# 1. Agreement with the reference implementation
# --------------------------------------------------------------------------


def test_agrees_with_reference_for_supported_dtypes(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {
        "f16": rng.standard_normal((4, 3)).astype(np.float16),
        "f32": rng.standard_normal((5, 2)).astype(np.float32),
        "i64": rng.integers(-1000, 1000, size=(2, 4), dtype=np.int64),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf, safe_open(str(path), framework="numpy") as ref:

        def _check(name, arr):
            # Scoped in a helper so the zero-copy view `got` (which pins an
            # exported buffer on sf's mmap) is released via CPython
            # refcounting when this function returns, rather than lingering
            # in the test's frame until the `with` block exits and tries to
            # close the mmap out from under it.
            got = sf.whole(name)
            unsigned = _UNSIGNED_OF[arr.dtype.name]
            want = ref.get_tensor(name).view(unsigned).reshape(got.shape)
            np.testing.assert_array_equal(got, want)

        for name, arr in tensors.items():
            _check(name, arr)


# --------------------------------------------------------------------------
# 2. bf16 round-trip -- the case that motivates this module
# --------------------------------------------------------------------------


def test_bf16_round_trips_while_reference_fails(tmp_path):
    rng = np.random.default_rng(1)
    src = rng.standard_normal((4, 5)).astype(np.float32)
    bits = _float32_to_bf16_bits(src)
    path = tmp_path / "model.safetensors"
    _write_bf16_file(path, {"w": bits})

    with SafetensorsFile(path) as sf:
        def _check():
            # Scoped so the mmap-backed view is released before `with`
            # exits and closes the mmap (see comment in the dtype-agreement
            # test above).
            got = sf.whole("w")
            np.testing.assert_array_equal(got, bits.reshape(got.shape))

        _check()

    # Documents *why* this module exists: if safetensors ever gains bf16
    # support in the numpy backend, this assertion starts failing and this
    # module becomes removable.
    with safe_open(str(path), framework="numpy") as ref:
        with pytest.raises(TypeError):
            ref.get_tensor("w")


# --------------------------------------------------------------------------
# 3. rows() vs whole()
# --------------------------------------------------------------------------


def test_rows_matches_whole_slices(tmp_path):
    tensors = {"w": np.arange(24, dtype=np.float32).reshape(6, 4)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        # Copied so this reference array does not itself pin the mmap's
        # exported buffer past the end of the `with` block.
        whole = sf.whole("w").copy()
        for a, b in [(0, 1), (5, 6), (0, 6), (2, 4), (3, 3)]:
            np.testing.assert_array_equal(sf.rows("w", a, b), whole[a:b])


# --------------------------------------------------------------------------
# 4. Zero-copy
# --------------------------------------------------------------------------


def test_rows_are_zero_copy_views_onto_the_mmap(tmp_path):
    tensors = {"w": np.arange(24, dtype=np.float32).reshape(6, 4)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        arr = sf.rows("w", 1, 3)
        # Walk .base until we hit something that isn't a numpy array: a
        # zero-copy chain bottoms out at a memoryview over the mmap, not at
        # a fresh buffer. A copy would instead have `base is None`.
        obj = arr
        seen_memoryview = False
        while isinstance(obj, np.ndarray):
            assert obj.base is not None, "array is a copy, not a view"
            obj = obj.base
        if isinstance(obj, memoryview):
            seen_memoryview = True
            obj = obj.obj
        assert seen_memoryview
        assert obj is sf._mm
        # Release the exported buffer before `with` exits and tries to
        # close the mmap out from under it.
        del arr


# --------------------------------------------------------------------------
# 5. 1-D and 0-d tensors
# --------------------------------------------------------------------------


def test_1d_and_0d_tensors_produce_documented_shapes(tmp_path):
    tensors = {
        "bias": np.arange(7, dtype=np.float32),
        "scalar": np.array(42.0, dtype=np.float32),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        bias_spec = sf.spec("bias")
        assert bias_spec.num_rows == 7
        assert bias_spec.row_elems == 1
        assert sf.whole("bias").shape == (7, 1)

        scalar_spec = sf.spec("scalar")
        assert scalar_spec.num_rows == 1
        assert scalar_spec.row_elems == 1
        assert sf.whole("scalar").shape == (1, 1)


# --------------------------------------------------------------------------
# 6. gather_rows
# --------------------------------------------------------------------------


def test_gather_rows_matches_fancy_indexing(tmp_path):
    tensors = {"w": np.arange(40, dtype=np.int64).reshape(10, 4)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        # Copied for the same reason as in test_rows_matches_whole_slices.
        whole = sf.whole("w").copy()
        for indices in ([0, 1, 2], [9, 8, 7], [0, 0, 3, 3], list(range(10))[::-1]):
            got = sf.gather_rows("w", indices)
            np.testing.assert_array_equal(got, whole[np.asarray(indices)])


# --------------------------------------------------------------------------
# 7. header_bytes verbatim
# --------------------------------------------------------------------------


def test_header_bytes_is_verbatim(tmp_path):
    tensors = {"w": np.arange(8, dtype=np.float32).reshape(2, 4)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt"})

    on_disk = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", on_disk, 0)

    with SafetensorsFile(path) as sf:
        assert sf.header_bytes == on_disk[: 8 + header_len]


# --------------------------------------------------------------------------
# 8. __metadata__
# --------------------------------------------------------------------------


def test_metadata_excluded_from_names_and_surfaced_separately(tmp_path):
    tensors = {"w": np.arange(4, dtype=np.float32)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt", "note": "x"})

    with SafetensorsFile(path) as sf:
        assert "__metadata__" not in sf.names()
        assert sf.names() == ["w"]
        assert sf.metadata == {"format": "pt", "note": "x"}


def test_metadata_defaults_to_empty_dict_when_absent(tmp_path):
    tensors = {"w": np.arange(4, dtype=np.float32)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        assert sf.metadata == {}


# --------------------------------------------------------------------------
# 9. Malformed inputs
# --------------------------------------------------------------------------


def test_truncated_file_raises_usage_error(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"\x03\x00\x00\x00\x00\x00\x00")  # 7 bytes, < 8
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_header_len_larger_than_file_raises_usage_error(tmp_path):
    path = tmp_path / "model.safetensors"
    # Claims a header of 1000 bytes but the file has none.
    path.write_bytes(struct.pack("<Q", 1000) + b"{}")
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_non_json_header_raises_usage_error(tmp_path):
    path = tmp_path / "model.safetensors"
    body = b"not json at all!"
    pad = (-(8 + len(body))) % 8
    body += b" " * pad
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_data_offsets_inconsistent_with_shape_raises_usage_error(tmp_path):
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}  # needs 16
    body = json.dumps(header).encode("utf-8")
    pad = (-(8 + len(body))) % 8
    body += b" " * pad
    data = b"\x00" * 16
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_unknown_dtype_raises_usage_error(tmp_path):
    header = {"w": {"dtype": "COMPLEX128", "shape": [2], "data_offsets": [0, 32]}}
    body = json.dumps(header).encode("utf-8")
    pad = (-(8 + len(body))) % 8
    body += b" " * pad
    data = b"\x00" * 32
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_missing_required_field_raises_usage_error(tmp_path):
    header = {"w": {"dtype": "F32", "shape": [4]}}  # no data_offsets
    body = json.dumps(header).encode("utf-8")
    pad = (-(8 + len(body))) % 8
    body += b" " * pad
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_data_offsets_outside_data_region_raises_usage_error(tmp_path):
    header = {"w": {"dtype": "F16", "shape": [4], "data_offsets": [0, 100]}}
    body = json.dumps(header).encode("utf-8")
    pad = (-(8 + len(body))) % 8
    body += b" " * pad
    data = b"\x00" * 4  # far short of 100
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)
    with pytest.raises(UsageError):
        SafetensorsFile(path)


def test_out_of_range_rows_raises_usage_error(tmp_path):
    tensors = {"w": np.arange(12, dtype=np.float32).reshape(3, 4)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    with SafetensorsFile(path) as sf:
        with pytest.raises(UsageError):
            sf.rows("w", 0, 4)
        with pytest.raises(UsageError):
            sf.rows("w", 2, 1)
        with pytest.raises(UsageError):
            sf.gather_rows("w", [0, 1, 3])
        with pytest.raises(UsageError):
            sf.gather_rows("w", [-1])


# --------------------------------------------------------------------------
# 10. Integration with the codec
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURE_BASE.exists() and FIXTURE_FINETUNED.exists()),
    reason="fixtures/mlp-tiny not generated; run tools/gen_fixtures.py",
)
def test_reader_feeds_codec_bit_identically_on_real_fixture():
    with SafetensorsFile(FIXTURE_BASE) as base_sf, SafetensorsFile(
        FIXTURE_FINETUNED
    ) as tgt_sf:
        common = sorted(set(base_sf.names()) & set(tgt_sf.names()))
        assert common, "expected at least one shared tensor between base and finetuned"

        def _check_pair(name):
            # Scoped so the mmap-backed `base_rows`/`tgt_rows` views are
            # released when this function returns, rather than lingering
            # (as the loop variable's last binding) until the `with` block
            # exits and tries to close both mmaps out from under them.
            base_spec = base_sf.spec(name)
            tgt_spec = tgt_sf.spec(name)
            if base_spec.shape != tgt_spec.shape or base_spec.dtype != tgt_spec.dtype:
                return  # not alignable without permutation; out of scope here

            base_rows = base_sf.whole(name)
            tgt_rows = tgt_sf.whole(name)

            encoded = encode_chunk(tgt_rows, base_rows, dtype=tgt_spec.dtype)
            decoded = decode_chunk(
                encoded.encoding, encoded.payload, base_rows, dtype=tgt_spec.dtype
            )
            np.testing.assert_array_equal(decoded.reshape(tgt_rows.shape), tgt_rows)

        for name in common:
            _check_pair(name)


# --------------------------------------------------------------------------
# 11. Context manager lifecycle
# --------------------------------------------------------------------------


def test_close_closes_mmap_and_file_handle_and_blocks_further_reads(tmp_path):
    tensors = {"w": np.arange(4, dtype=np.float32)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    sf = SafetensorsFile(path)
    sf.close()
    assert sf._mm.closed
    assert sf._fh.closed

    with pytest.raises(UsageError):
        sf.names()
    with pytest.raises(UsageError):
        sf.whole("w")

    # Double close is a no-op, not an error.
    sf.close()


def test_context_manager_closes_on_exit(tmp_path):
    tensors = {"w": np.arange(4, dtype=np.float32)}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))

    with SafetensorsFile(path) as sf:
        pass
    assert sf._mm.closed
    assert sf._fh.closed


def test_close_succeeds_while_zero_copy_views_are_still_alive(tmp_path):
    """The ordinary chunk-encoding loop holds a view at block exit, because
    `rows()` is zero-copy by design. `mmap.close()` refuses to run while a view
    exports a pointer into it, and letting that BufferError escape would throw
    away work that had already succeeded. Regression test: it must not escape,
    and any view held across the exit must stay readable.
    """
    path = tmp_path / "m.safetensors"
    save_file({"w": np.arange(64, dtype=np.float16).reshape(8, 8)}, str(path))

    held = []
    with SafetensorsFile(path) as f:  # must not raise on exit
        for start in range(0, 8, 4):
            held.append(f.rows("w", start, start + 4))

    assert np.array_equal(
        np.concatenate(held).ravel(), np.arange(64, dtype=np.float16).view(np.uint16)
    )


def test_reader_rejects_new_reads_after_close(tmp_path):
    """Swallowing the BufferError must not leave the reader usable."""
    path = tmp_path / "m.safetensors"
    save_file({"w": np.arange(8, dtype=np.float16)}, str(path))

    f = SafetensorsFile(path)
    view = f.rows("w", 0, 4)
    f.close()

    assert view[0][0] == np.float16(0).view(np.uint16)  # existing view still valid
    with pytest.raises(UsageError, match="closed"):
        f.rows("w", 0, 4)
