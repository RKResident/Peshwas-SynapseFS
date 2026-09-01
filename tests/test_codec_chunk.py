"""Tests for the chunk codec (synapsefs/codec/chunk.py, FORMAT.md section 8).

The load-bearing claim of this whole project is that reconstruction is
*byte-exact*, not approximately exact -- so these tests are heavier on
exhaustive enumeration than a unit-test suite normally would be. The 8- and
16-bit domains are small enough to check completely, and checking them
completely is strictly better than sampling when the failure mode is "one
bit pattern in 65,536 is wrong".

Run:
    pytest tests/test_codec_chunk.py -v
"""

from __future__ import annotations

import blake3
import numpy as np
import pytest
import zstandard as zstd

from synapsefs.codec.chunk import (
    DELTA,
    DEFAULT_LEVEL,
    plain_stream,
    unshuffle,
    shuffle,
    DELTA_SHUFFLE,
    RAW,
    RAW_ZSTD,
    RAW_SHUFFLE_ZSTD,
    EncodedChunk,
    decode_chunk,
    dtype_spec,
    encode_chunk,
    DELTA_ZIGZAG_ESCAPE,
    _escape_stream,
    _unescape_stream,
    is_delta,
    unzigzag,
    zigzag,
)

ALL_U16 = np.arange(65536, dtype=np.uint16)

# 16 bits is the only width small enough to enumerate, and it is also the
# width that matters most (the PS grades fp16/bf16 checkpoints). The wider
# dtypes get randomized coverage instead -- see
# `test_key_roundtrips_at_every_supported_width`.


# ---------------------------------------------------------------------------
# Steps 2-3: zigzag
#
# The monotone key that used to sit in front of this step (an xor mask making
# bit-pattern order match float order across the sign boundary) is gone --
# nothing in the encode/decode path calls it, and its tests went with it. What
# remains is plain modular subtraction plus zigzag, which is what the codec
# has actually used since `codec/chunk.py`'s "no monotone key" note.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 8])
def test_zigzag_roundtrips_at_every_supported_width(width):
    """The exhaustive 16-bit tests below only reach one width. The codec also
    has to be right at 4 and 8 bytes (F32, and the int64 BatchNorm counter),
    including the extremes where the wraparound bites."""
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    hi = np.iinfo(unsigned).max
    rng = np.random.default_rng(width)
    domain = np.concatenate([
        np.array([0, 1, hi, hi - 1, hi // 2, hi // 2 + 1], dtype=unsigned),
        rng.integers(0, hi, size=4096, dtype=np.uint64).astype(unsigned),
    ])
    assert np.array_equal(unzigzag(zigzag(domain)), domain)


@pytest.mark.parametrize("domain", [ALL_U16], ids=["16bit"])
def test_zigzag_is_a_bijection_at_native_width(domain):
    zz = zigzag(domain)
    assert zz.dtype == domain.dtype
    assert len(np.unique(zz)) == len(domain)
    assert np.array_equal(unzigzag(zz), domain)


def test_zigzag_keeps_small_deltas_small():
    """Why zigzag is here at all: a delta of -1 must not become 0xFFFF, or
    zstd sees high-entropy bytes exactly where the residual is smallest."""
    deltas = np.array([0, 1, -1, 2, -2, 100, -100], dtype=np.int16)
    zz = zigzag(deltas.view(np.uint16))
    assert list(zz) == [0, 2, 1, 4, 3, 200, 199]


def test_modular_delta_is_exact_even_when_it_wraps():
    """Native-width subtraction wraps instead of widening. Reconstruction is
    still exact -- (a - b) + b == a mod 2**n for every pair, including the
    pairs that wrap."""
    base = np.array([0x0000, 0xFFFF, 0x0001, 0x8000], dtype=np.uint16)
    target = np.array([0xFFFF, 0x0000, 0x8000, 0x0001], dtype=np.uint16)
    recovered = base + unzigzag(zigzag(target - base))
    assert np.array_equal(recovered, target)


# ---------------------------------------------------------------------------
# Chunk encode / decode round-trips
# ---------------------------------------------------------------------------


def _random(dtype_name: str, n: int, seed: int) -> np.ndarray:
    """A chunk of `n` elements whose bytes are valid for `dtype_name`."""
    rng = np.random.default_rng(seed)
    width, _ = dtype_spec(dtype_name)
    raw = rng.integers(0, 256, size=n * width, dtype=np.uint8)
    return raw.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[width])


# Every dtype the codec supports -- see `_DTYPES` in chunk.py for why the
# list is this short.
ROUNDTRIP_DTYPES = ["F16", "BF16", "F32", "I64"]


@pytest.mark.parametrize("dtype", ROUNDTRIP_DTYPES)
def test_roundtrip_with_a_base(dtype):
    base = _random(dtype, 512, seed=1)
    target = _random(dtype, 512, seed=2)
    chunk = encode_chunk(target, base, dtype=dtype)
    assert chunk.encoding in (DELTA_SHUFFLE, RAW_SHUFFLE_ZSTD)
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype=dtype)
    assert np.array_equal(out, target.reshape(-1))


@pytest.mark.parametrize("dtype", ROUNDTRIP_DTYPES)
@pytest.mark.parametrize("compress_raw", [True, False], ids=["raw-zstd", "raw"])
def test_roundtrip_without_a_base(dtype, compress_raw):
    target = _random(dtype, 512, seed=3)
    chunk = encode_chunk(target, None, dtype=dtype, compress_raw=compress_raw)
    assert chunk.encoding == (RAW_SHUFFLE_ZSTD if compress_raw else RAW)
    out = decode_chunk(chunk.encoding, chunk.payload, dtype=dtype)
    assert np.array_equal(out, target.reshape(-1))


def test_roundtrip_of_every_fp16_bit_pattern_in_one_chunk():
    """The whole fp16 domain as a single chunk, against a rotated version of
    itself so that every element takes a different delta -- denormals, both
    zeros, every NaN payload, both infinities."""
    target = ALL_U16.view(np.float16)
    base = np.roll(target, 1)
    chunk = encode_chunk(target, base, dtype="F16")
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="F16")
    assert np.array_equal(out.view(np.float16).view(np.uint16), ALL_U16)


def test_roundtrip_preserves_bf16_which_numpy_cannot_represent():
    """bf16 arrives as uint16 because numpy has no bfloat16. The codec must
    key it as a float anyway -- which is why dtype is a required argument and
    not inferred from the array."""
    base = _random("BF16", 256, seed=4)
    target = (base + np.uint16(3)).astype(np.uint16)
    chunk = encode_chunk(target, base, dtype="BF16")
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="BF16")
    assert np.array_equal(out, target)


def test_multidimensional_chunks_roundtrip_flat():
    """decode returns a flat bit array; shape lives in the manifest, not here."""
    base = np.random.default_rng(5).standard_normal((32, 16)).astype(np.float16)
    target = base + np.float16(0.01)
    chunk = encode_chunk(target, base, dtype="F16")
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="F16")
    assert out.shape == (512,)
    assert np.array_equal(out.view(np.float16).reshape(32, 16), target)


# ---------------------------------------------------------------------------
# Sizing, fallback, and the residual_ratio denominator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ROUNDTRIP_DTYPES)
def test_stream_never_inflates(dtype):
    """The reason FORMAT.md's varint/bitpack step is unnecessary: the residual
    stream is exactly the size of the chunk it encodes. If someone reintroduces
    widening, this fails and `EncodedChunk.original_len` stops being redundant
    -- which is precisely when the two fields need to diverge."""
    base = _random(dtype, 512, seed=6)
    target = _random(dtype, 512, seed=7)
    chunk = encode_chunk(target, base, dtype=dtype, allow_raw_fallback=False)
    assert chunk.plain_len == chunk.original_len == target.nbytes


def test_original_len_is_the_tensor_size_not_the_stream_size():
    """`residual_ratio` (CLI.md section 3.1) is a graded metric; its
    denominator must be real tensor bytes."""
    base = np.zeros(1024, dtype=np.float16)
    chunk = encode_chunk(base + np.float16(1.0), base, dtype="F16")
    assert chunk.original_len == 2048


def test_nearly_identical_chunks_compress_far_better_than_raw():
    """End-to-end sanity that the codec actually does its job: a slightly
    perturbed tensor must produce a residual dramatically smaller than storing
    the tensor outright. This is the residual_ratio premise in miniature."""
    rng = np.random.default_rng(8)
    base = rng.standard_normal(65536).astype(np.float16)
    target = base.copy()
    touched = rng.choice(65536, size=256, replace=False)
    target[touched] = target[touched] + np.float16(0.001)

    delta = encode_chunk(target, base, dtype="F16", allow_raw_fallback=False)
    raw = encode_chunk(target, None, dtype="F16")
    # Which residual encoding wins depends on how many deltas fit in a byte,
    # and this fixture barely moves 256 of 65536 elements, so it takes the
    # escape path. The claim under test is the ratio, not the encoding name.
    assert is_delta(delta.encoding)
    assert delta.stored_len * 10 < raw.stored_len


def test_identical_chunks_produce_an_all_zero_residual():
    base = np.random.default_rng(9).standard_normal(65536).astype(np.float16)
    chunk = encode_chunk(base, base, dtype="F16", allow_raw_fallback=False)
    assert chunk.stored_len < 200  # an all-zero 128 KiB stream
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="F16")
    assert np.array_equal(out.view(np.float16), base)


def test_raw_fallback_wins_when_the_delta_does_not_help():
    """FORMAT.md section 7: "per-chunk when delta doesn't help". A structured
    target against unrelated random noise deltas worse than it stores."""
    rng = np.random.default_rng(10)
    base = rng.integers(0, 65536, size=16384, dtype=np.uint16).view(np.float16)
    target = np.zeros(16384, dtype=np.float16)  # compresses to almost nothing raw

    fell_back = encode_chunk(target, base, dtype="F16", allow_raw_fallback=True)
    forced = encode_chunk(target, base, dtype="F16", allow_raw_fallback=False)

    assert forced.encoding == DELTA_SHUFFLE
    assert fell_back.encoding == RAW_SHUFFLE_ZSTD
    assert fell_back.stored_len < forced.stored_len
    # A fallen-back chunk must still decode -- and without needing the base.
    assert np.array_equal(
        decode_chunk(fell_back.encoding, fell_back.payload, dtype="F16"),
        target.view(np.uint16),
    )


def test_fallback_changes_the_content_hash_to_the_raw_content():
    """Deliberate: a raw chunk dedups against any identical raw chunk in the
    repo, while a residual only matches a residual taken against the same base."""
    rng = np.random.default_rng(11)
    base = rng.integers(0, 65536, size=16384, dtype=np.uint16).view(np.float16)
    target = np.zeros(16384, dtype=np.float16)

    fell_back = encode_chunk(target, base, dtype="F16")
    standalone = encode_chunk(target, None, dtype="F16")
    assert fell_back.content_hash == standalone.content_hash


def test_stored_len_always_matches_the_payload():
    chunk = encode_chunk(np.zeros(64, dtype=np.float16), None, dtype="F16")
    assert chunk.stored_len == len(chunk.payload)


# ---------------------------------------------------------------------------
# Hash identity
# ---------------------------------------------------------------------------


def test_content_hash_is_independent_of_compression_level():
    """FORMAT.md section 2: hashes cover uncompressed bytes, so that changing
    the zstd level or adding a pack dictionary never forks a chunk's identity
    and silently destroys dedup across a repack."""
    base = np.random.default_rng(12).standard_normal(4096).astype(np.float16)
    target = base + np.float16(0.5)
    hashes = {
        encode_chunk(
            target, base, dtype="F16", level=level, allow_raw_fallback=False
        ).content_hash
        for level in (1, 3, 9, 19)
    }
    assert len(hashes) == 1


def test_content_hash_is_32_bytes_of_blake3():
    chunk = encode_chunk(np.zeros(8, dtype=np.float16), None, dtype="F16")
    assert isinstance(chunk.content_hash, bytes) and len(chunk.content_hash) == 32


def test_different_content_gives_different_hashes():
    a = encode_chunk(np.zeros(64, dtype=np.float16), None, dtype="F16")
    b = encode_chunk(np.ones(64, dtype=np.float16), None, dtype="F16")
    assert a.content_hash != b.content_hash


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_element_width_disagreeing_with_dtype_is_an_error():
    """The guard against the silent-10x-regression bug: an fp32 array keyed as
    if it were 16-bit still round-trips, so only an explicit check catches it."""
    arr = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError, match="4-byte elements"):
        encode_chunk(arr, None, dtype="F16")


def test_batchnorm_num_batches_tracked_round_trips():
    """A `.half()`'d ResNet-style checkpoint still carries an int64 0-d
    `num_batches_tracked` buffer per BatchNorm -- `.half()` does not touch
    integer buffers. The PS demands byte-for-byte reconstruction, so that
    scalar must survive. This is the entire reason the SINT key kind and the
    8-byte width exist; without it the codec could be fp16-only."""
    base = np.array(41, dtype=np.int64)
    target = np.array(42, dtype=np.int64)
    chunk = encode_chunk(target, base, dtype="I64", allow_raw_fallback=False)
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="I64")
    assert out.view(np.int64)[0] == 42


def test_shape_mismatch_is_rejected_before_flattening():
    """(4, 8) and (8, 4) have equal element counts and would otherwise encode
    into a residual that reconstructs transposed garbage."""
    target = np.zeros((4, 8), dtype=np.float16)
    base = np.zeros((8, 4), dtype=np.float16)
    with pytest.raises(ValueError, match="shape mismatch"):
        encode_chunk(target, base, dtype="F16")


def test_unknown_dtype_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="unsupported dtype"):
        encode_chunk(np.zeros(4, dtype=np.float16), None, dtype="FLOAT16")
    # A real safetensors dtype we deliberately do not support fails the
    # same loud way, rather than being silently mis-keyed.
    with pytest.raises(ValueError, match="unsupported dtype"):
        encode_chunk(np.zeros(4, dtype=np.uint8), None, dtype="U8")


def test_unknown_encoding_is_rejected_on_decode():
    with pytest.raises(ValueError, match="unknown chunk encoding"):
        decode_chunk("delta-v2", b"", dtype="F16")


def test_delta_without_a_base_is_an_error():
    base = np.zeros(64, dtype=np.float16)
    chunk = encode_chunk(base + np.float16(1), base, dtype="F16", allow_raw_fallback=False)
    with pytest.raises(ValueError, match="without a base"):
        decode_chunk(chunk.encoding, chunk.payload, None, dtype="F16")


def test_base_of_the_wrong_length_is_an_error():
    base = np.zeros(64, dtype=np.float16)
    chunk = encode_chunk(base + np.float16(1), base, dtype="F16", allow_raw_fallback=False)
    with pytest.raises(ValueError, match="base chunk has"):
        decode_chunk(chunk.encoding, chunk.payload, np.zeros(32, np.float16), dtype="F16")


# ---------------------------------------------------------------------------
# Contracts the pack writer and FUSE read path rely on
# ---------------------------------------------------------------------------


def test_raw_decode_is_a_zero_copy_view_of_the_payload():
    """`raw` exists so the FUSE read path can memcpy out of the page cache
    instead of decompressing (FORMAT.md section 7). If this ever starts
    copying, that argument is gone."""
    target = np.arange(64, dtype=np.uint16)
    chunk = encode_chunk(target, None, dtype="F16", compress_raw=False)
    out = decode_chunk(RAW, chunk.payload, dtype="F16")
    assert out.base is chunk.payload
    assert not out.flags.writeable


def test_encoded_chunk_is_immutable():
    chunk = encode_chunk(np.zeros(8, dtype=np.float16), None, dtype="F16")
    with pytest.raises(AttributeError):
        chunk.encoding = DELTA_SHUFFLE  # type: ignore[misc]


def test_a_shared_compressor_is_accepted():
    compressor = zstd.ZstdCompressor(level=7)
    base = np.zeros(256, dtype=np.float16)
    chunk = encode_chunk(base + np.float16(1), base, dtype="F16", compressor=compressor)
    out = decode_chunk(chunk.encoding, chunk.payload, base, dtype="F16")
    assert np.array_equal(out.view(np.float16), base + np.float16(1))


# --- byte shuffle -----------------------------------------------------------


@pytest.mark.parametrize("width", [2, 4, 8])
def test_shuffle_round_trips_at_every_width(width):
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 256, size=64 * width, dtype=np.uint8).tobytes()
    assert unshuffle(shuffle(stream, width), width) == stream


def test_shuffle_groups_bytes_by_position():
    """The transform, stated concretely: byte 0 of every element first, then
    byte 1. Asserted on a literal rather than via round-trip, so a transpose in
    the wrong direction cannot pass."""
    stream = bytes([0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2])
    assert shuffle(stream, 2) == bytes([0xA0, 0xA1, 0xA2, 0xB0, 0xB1, 0xB2])


def test_shuffled_encodings_are_what_encode_chunk_produces():
    base = np.arange(256, dtype=np.uint16)
    assert encode_chunk(base, None, dtype="F16").encoding == RAW_SHUFFLE_ZSTD


def test_residual_encoding_is_chosen_by_how_many_deltas_fit_in_a_byte():
    """The selection rule, at both ends.

    An element costs 1 byte under the escape encoding when it fits and 3 when
    it does not, so it only pays while most of them fit. Choosing by dtype
    instead would be wrong in both directions: bf16 against a distant base
    escapes 82.9% of the time and belongs on the shuffle path, and any dtype
    whose residuals are small belongs on the escape path.
    """
    rng = np.random.default_rng(4)
    base = rng.integers(0, 65536, size=8192).astype(np.uint16)

    tiny = (base + rng.integers(-8, 8, size=8192)).astype(np.uint16)
    assert encode_chunk(tiny, base, dtype="BF16",
                        allow_raw_fallback=False).encoding == DELTA_ZIGZAG_ESCAPE

    wide = (base + rng.integers(-30000, 30000, size=8192)).astype(np.uint16)
    assert encode_chunk(wide, base, dtype="BF16",
                        allow_raw_fallback=False).encoding == DELTA_SHUFFLE


def test_escape_stream_round_trips_the_entire_uint16_domain():
    """Every value either fits the narrow plane or escapes to the wide one, and
    the boundary at 0xFF is exactly where an off-by-one would hide."""
    domain = np.arange(65536, dtype=np.uint16)
    assert np.array_equal(_unescape_stream(_escape_stream(domain)), domain)


@pytest.mark.parametrize("spread", [3, 200, 40000])
def test_escape_encoding_round_trips_through_encode_decode(spread):
    rng = np.random.default_rng(11)
    base = rng.integers(0, 65536, size=4096).astype(np.uint16)
    target = (base + rng.integers(-spread, spread + 1, size=4096)).astype(np.uint16)
    enc = encode_chunk(target, base, dtype="BF16", allow_raw_fallback=False)
    back = decode_chunk(enc.encoding, enc.payload, base, dtype="BF16")
    assert np.array_equal(back, target)


def test_shuffle_actually_shrinks_a_residual():
    """The reason the transform exists. Isolated against the *same* stream
    compressed without the shuffle, at the same level -- so this measures the
    shuffle and nothing else."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 60000, size=8192, dtype=np.uint16)
    target = (base + rng.integers(0, 8, size=8192, dtype=np.uint16)).astype(np.uint16)

    shuffled = encode_chunk(target, base, dtype="F16")
    delta = (target - base).astype(np.uint16)
    unshuffled = len(zstd.ZstdCompressor(level=DEFAULT_LEVEL).compress(delta.tobytes()))
    assert shuffled.stored_len < unshuffled


def test_the_encoder_no_longer_zigzags():
    """Zigzag and byte shuffle are substitutes, and shuffle is the better one.
    Pinned as a test because the losing combination still decodes, so nothing
    else would notice it creeping back in."""
    base = np.arange(4096, dtype=np.uint16)
    target = (base + 300).astype(np.uint16)
    chunk = encode_chunk(target, base, dtype="F16")

    delta = (target - base).astype(np.uint16)
    assert plain_stream(chunk.encoding, chunk.payload) == shuffle(delta.tobytes(), 2)


def test_the_hash_covers_the_shuffled_stream():
    """`plain_stream` must return exactly what was hashed, or `verify --deep`
    compares the wrong bytes and reports every chunk as substituted."""
    base = np.arange(512, dtype=np.uint16)
    chunk = encode_chunk(base + 5, base, dtype="F16")
    assert blake3.blake3(plain_stream(chunk.encoding, chunk.payload)).digest() \
        == chunk.content_hash


def test_unshuffled_encodings_still_decode():
    """Chunks written before the shuffle landed must keep decoding. `RAW_ZSTD`
    is not an alias of `RAW_SHUFFLE_ZSTD` -- it means 'this stream was never
    shuffled' -- so decode must not un-shuffle it.

    `DELTA` (`delta-zigzag-zstd`) was the other pre-shuffle encoding, but it is
    no longer decodable at all: its residual was a zigzag over monotone keys,
    and both the un-zigzag path and the key functions it depended on are gone.
    Nothing writes it any more either, so this only needs to cover the
    survivor."""
    base = np.arange(256, dtype=np.uint16)
    target = base + 7
    compressor = zstd.ZstdCompressor(level=DEFAULT_LEVEL)

    decoded = decode_chunk(
        RAW_ZSTD, compressor.compress(target.tobytes()), None, dtype="F16",
    )
    assert np.array_equal(decoded, target)


def test_delta_legacy_encoding_is_no_longer_decodable():
    """Pinned rather than silently true: `DELTA` used to decode via the
    monotone key, and now raises instead of silently reinterpreting the
    stream as plain-modular residual bytes it never was."""
    with pytest.raises(ValueError):
        decode_chunk(DELTA, b"\x00\x00", np.arange(1, dtype=np.uint16), dtype="F16")


def test_the_encoder_no_longer_uses_the_monotone_key():
    """The residual is a plain modular subtraction of raw bit patterns.

    Pinned because the monotone-key version still decodes (as
    `delta-zigzag-zstd`) and round-trips identically, so nothing else in the
    suite would notice it coming back."""
    base = np.array([0x3C00, 0xBC00, 0x0001], dtype=np.uint16)   # 1.0, -1.0, tiny
    target = np.array([0x3C05, 0xBC05, 0x0006], dtype=np.uint16)
    chunk = encode_chunk(target, base, dtype="F16")

    expected = shuffle((target - base).astype(np.uint16).tobytes(), 2)
    assert plain_stream(chunk.encoding, chunk.payload) == expected
    assert np.array_equal(decode_chunk(chunk.encoding, chunk.payload, base,
                                       dtype="F16"), target)


def test_negative_weights_round_trip_without_the_key():
    """The key existed to keep negative floats ordered. Removing it must not
    cost exactness -- modular subtraction is reversible regardless of how the
    bits are laid out."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 65536, size=4096, dtype=np.uint16)
    target = (base + rng.integers(0, 2000, size=4096, dtype=np.uint16)).astype(np.uint16)
    chunk = encode_chunk(target, base, dtype="F16")
    assert np.array_equal(
        decode_chunk(chunk.encoding, chunk.payload, base, dtype="F16"), target)
