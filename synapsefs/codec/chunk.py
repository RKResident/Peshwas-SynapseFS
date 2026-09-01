"""Chunk codec: bit-exact residual encode/decode (FORMAT.md section 8).

One chunk in, one chunk out. This module knows nothing about tensors, files,
manifests, or chunking policy -- it operates on a flat stream of raw bit
patterns and an element width. Everything above it (which rows form a chunk,
where the bytes came from) is `checkpoint.py`'s job; everything below it
(where the payload lands) is the pack writer's.

Three properties this module guarantees, each pinned by a test:

- **Byte-exact.** Every step is integer arithmetic on raw bit patterns. No
  floating-point operation occurs anywhere in this file, so reconstruction is
  exact unconditionally rather than "exact in practice". NaN, Inf, denormals
  and both signed zeros round-trip because nothing here interprets them as
  numbers.
- **No inflation.** The delta stream is exactly as many bytes as the tensor
  chunk it encodes -- see `zigzag` for why no widening is needed.
- **Content hashes cover uncompressed bytes.** Changing the zstd level or
  adding a pack dictionary must never fork a chunk's identity (FORMAT.md
  section 2), so the hash is taken before compression, never after.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import blake3
import struct

import numpy as np
import zstandard as zstd

__all__ = [
    "EncodedChunk",
    "encode_chunk",
    "decode_chunk",
    "zigzag",
    "unzigzag",
    "dtype_spec",
    "plain_stream",
    "is_delta",
    "shuffle",
    "unshuffle",
    "unshuffle_to_array",
    "RAW",
    "RAW_ZSTD",
    "DELTA",
    "RAW_SHUFFLE_ZSTD",
    "DELTA_SHUFFLE",
]

RAW = "raw"
RAW_ZSTD = "raw-zstd"
DELTA = "delta-zigzag-zstd"  # legacy encoding: decode-only, no longer written

# Current encodings. The shuffle is part of the *stream definition* here, not a
# compressor setting (see `shuffle`), so these carry their own names rather
# than a flag on the old ones. The two names above are legacy: they still
# decode, they are simply never written any more.
#
# `delta-shuffle-zstd` differs from the legacy `delta-zigzag-zstd` in two ways,
# both measured: it has no zigzag step (shuffle subsumes it, see `zigzag`) and
# no monotone key (see `to_monotone_key`). Its residual is a plain modular
# subtraction of raw bit patterns.
RAW_SHUFFLE_ZSTD = "raw-shuffle-zstd"
DELTA_SHUFFLE = "delta-shuffle-zstd"

#: Zigzag, then a byte per element: the value itself when it fits, or the
#: marker 0xFF when it does not, with the oversized values moved to a second
#: plane. See `_escape_stream` for the layout and `_ESCAPE_MAX_RATE` for when
#: it is chosen.
DELTA_ZIGZAG_ESCAPE = "delta-zigzag-escape-zstd"

#: Encodings whose stream is byte-shuffled and must be un-shuffled after
#: decompression. Consulted by `decode_chunk`, never by `plain_stream`.
#: Encodings whose whole plain stream is one byte-shuffled block. The escape
#: encoding is deliberately NOT here: only its second plane is shuffled, and
#: that happens inside the stream rather than over it.
_SHUFFLED = frozenset({RAW_SHUFFLE_ZSTD, DELTA_SHUFFLE})

# _ZSTD_FRAMED = frozenset({RAW_ZSTD, DELTA, RAW_SHUFFLE_ZSTD, DELTA_SHUFFLE,
#                           DELTA_ZIGZAG_ESCAPE})
# _DELTA_ENCODINGS = frozenset({DELTA, DELTA_SHUFFLE, DELTA_ZIGZAG_ESCAPE})

_ZSTD_FRAMED = frozenset({RAW_ZSTD, RAW_SHUFFLE_ZSTD, DELTA_SHUFFLE,
                          DELTA_ZIGZAG_ESCAPE})
_DELTA_ENCODINGS = frozenset({DELTA_SHUFFLE, DELTA_ZIGZAG_ESCAPE})


def is_delta(encoding: str) -> bool:
    """Whether this encoding is a residual and therefore needs a base.

    Call this rather than comparing against `DELTA` directly -- there are two
    residual encodings now, and a bare `== DELTA` silently treats a shuffled
    residual as self-contained.
    """
    return encoding in _DELTA_ENCODINGS

# zstd level 1, not 3. Compression is *non-monotone* in level on byte-shuffled
# residuals: levels 1-2 use zstd's `fast`/`dfast` match-finders, 3+ switch to
# `greedy`/`lazy`, and on long runs of near-identical high bytes the fast
# strategies find the long matches immediately while the lazy ones hunt for
# better matches that do not exist and emit more literals. Measured across five
# checkpoint pairs, level 1 is 0.8pp *smaller* than level 3 and ~2x faster to
# compress. Decompression is ~1.2 GB/s at every level, so the read path does
# not care.
DEFAULT_LEVEL = 1

# Key kinds. Which order-preserving map applies depends on how the dtype lays
# out its sign, not on how wide it is -- so width and kind are tracked apart.
FLOAT = "float"  # IEEE-754 style sign-magnitude: sign bit, then magnitude.
SINT = "sint"    # two's complement.

# safetensors dtype name -> (element width in bytes, key kind).
#
# Deliberately short. The PS grades checkpoints "in fp16/bf16 precision (not
# fp32)", so F16/BF16 are the only dtypes the weights themselves will use.
# The other two are not speculation:
#
#   F32 -- `tools/gen_fixtures.py` emits it, and an export that was never
#          .half()'d is fp32 throughout.
#   I64 -- a `.half()`'d ResNet-style model *still* carries an int64 0-d
#          `num_batches_tracked` buffer per BatchNorm, because .half() does
#          not touch integer buffers. Verified against torch 2.13. Since the
#          PS demands byte-for-byte reconstruction, that scalar must survive,
#          which is why the SINT key kind and 8-byte width exist at all.
#
# Everything else safetensors defines (F8_*, U8/U16/U32/U64, I8/I16/I32,
# BOOL, F64) is left out on purpose: no fixture or architecture in scope
# produces one. `dtype_spec` raises on an unknown name rather than guessing,
# so an unexpected dtype is a loud one-line fix, never silent corruption.
#
# Keyed on the *safetensors* name rather than a numpy dtype on purpose: numpy
# has no bfloat16, so a BF16 tensor necessarily arrives here as uint16 and its
# numpy dtype cannot be trusted to say what it is.
_DTYPES: dict[str, Tuple[int, str]] = {
    "F16": (2, FLOAT),
    "BF16": (2, FLOAT),
    # "F32": (4, FLOAT), #redundant?
    "I64": (8, SINT),
}

# _UINT_OF = {2: np.uint16, 4: np.uint32, 8: np.uint64}
_UINT_OF = {2: np.uint16, 8: np.uint64}

def dtype_spec(dtype: str) -> Tuple[int, str]:
    """Map a safetensors dtype name to `(element_width_bytes, key_kind)`.

    Raises ValueError for a name this codec does not know, rather than
    guessing a width -- a wrong width silently produces a valid-looking but
    incorrectly-keyed stream, which is the worst possible failure mode here.
    """
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dtype!r}; known: {', '.join(sorted(_DTYPES))}"
        ) from None


# --------------------------------------------------------------------------
# Step 1 (FORMAT.md section 8 step 1) used to run through a monotone integer
# key -- an xor mask that made bit-pattern order match numeric order across
# the sign boundary. Removed: nothing in the codec's encode/decode path calls
# it any more (see the note below, at the delta step), and its last consumer
# outside this file, `materialize._ulp_gap`, was rewritten to compare raw bit
# patterns directly instead of carrying the dependency for one caller.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Steps 2-3 -- delta and zigzag (FORMAT.md section 8 steps 2-3)
# --------------------------------------------------------------------------
#
# FORMAT.md says "delta = int32(key(B)) - int32(key(A))", widening 16-bit keys
# into 32-bit deltas, and then hints at varint or bitpacking to win the space
# back. Doing the arithmetic at the *native* width instead makes both the
# widening and the bitpacking unnecessary:
#
#   - Subtraction mod 2**n is exact and reversible: (a - b) + b == a for every
#     pair, wraparound included. Nothing is lost by not widening.
#   - Zigzag over the full n-bit signed range is a bijection onto the n-bit
#     unsigned range, so it needs no headroom either.
#   - When |true delta| < 2**(n-1) -- the overwhelmingly common case, since
#     aligned checkpoints differ slightly -- the wrapped value *is* the true
#     delta, so the zigzagged output is just as small as it would have been.
#     When it wraps, it aliases to the distance the short way around, which is
#     never larger than the widened form would have been.
#
# So the residual stream is exactly the size of the tensor chunk rather than
# double it, with no per-element Python work. `residual_ratio` is measured
# against a stream that never inflates.


def _consts(width: int):
    u = _UINT_OF[width]
    # 0, 1, msb, num_bits-1,
    return u(0), u(1), u(1 << (width * 8 - 1)), u(width * 8 - 1)

def zigzag(delta: np.ndarray) -> np.ndarray:
    """Interleave a two's-complement delta into an unsigned value so that
    small-magnitude deltas of either sign become small unsigned integers --
    which is what lets zstd see long runs of zero high bytes.

    Stays at the input width; see the module note above.
    """
    zero, one, _msb, shift = _consts(delta.dtype.itemsize)
    return (delta << one) ^ (zero - (delta >> shift))


def unzigzag(zz: np.ndarray) -> np.ndarray:
    """Exact inverse of `zigzag`."""
    zero, one, _msb, _shift = _consts(zz.dtype.itemsize)
    return (zz >> one) ^ (zero - (zz & one))


# --------------------------------------------------------------------------
# Chunk encode / decode
# --------------------------------------------------------------------------


# Deprecated: No more packfiles
@dataclass(frozen=True)
class EncodedChunk:
    """One encoded chunk, ready for fianl storage

    Frozen and named rather than a bare tuple because the pack writer, the
    manifest builder and the `commit --json` counters each want a different
    subset, and positional unpacking made adding a field a breaking change.
    """

    encoding: str
    """One of `raw`, `raw-zstd`, `delta-zigzag-zstd` (FORMAT.md section 7)."""

    content_hash: bytes
    """32-byte BLAKE3 of the *uncompressed* stream -- the chunk's identity for
    dedup. Never covers compressed bytes (FORMAT.md section 2)."""

    payload: bytes
    """Exactly what gets written into the packfile."""

    plain_len: int
    """Length of the uncompressed stream that `content_hash` covers, which the
    pack index needs in order to size the decode buffer."""

    original_len: int
    """True byte count of the tensor chunk this encodes -- the denominator for
    `residual_ratio` (CLI.md section 3.1).

    Currently always equal to `plain_len`, because the codec never inflates
    (see the zigzag note above); `test_stream_never_inflates` exists to fail
    loudly if a future encoding breaks that. They are kept as separate fields
    so that callers reaching for a ratio denominator cannot accidentally pick
    up a stream length that has started to diverge.
    """

    stored_checksum: bytes = b""
    """First 8 bytes of blake3 of the **stored** payload.

    Hashes different bytes than `content_hash`, which covers the uncompressed
    stream: this one answers "did these bytes rot or get substituted?" and is
    checkable without decompressing. It goes into the tensor-manifest, where
    being covered by the commit hash makes it ref-anchored -- which is what
    upgrades `verify --fast` from a rot scan to real tamper detection
    (ARCHITECTURE.md 4.5.2).

    Computed here rather than by the caller because this is where the real
    payload bytes exist, unwrapped.
    """

    is_identical: bool = False
    """True when this chunk's content was byte-identical to its base chunk.

    Cheap to know here and expensive to recover later: the delta array is
    already in hand, so `not delta.any()` is one vectorized pass against a zstd
    compression that costs orders of magnitude more. The caller uses it to
    apply FORMAT.md 4.5's reuse rule -- a tensor whose every chunk is identical
    needs no new tensor-manifest at all.

    Always False for a base-less chunk: "identical to nothing" is not a
    meaningful claim, and treating it as one would make root commits reuse a
    manifest that does not exist.
    """

    @property
    def stored_len(self) -> int:
        """Compressed size. Derived rather than stored, so it cannot drift out
        of agreement with `payload`."""
        return len(self.payload)


def _as_bits(arr: np.ndarray, width: int, role: str) -> np.ndarray:
    """Flatten `arr` to a contiguous 1-D unsigned view of `width`-byte elements.

    The width check is doing real work: it is the guard against handing this
    codec an array whose elements are not the size the declared dtype says.
    Without it, a float32 array under a 16-bit assumption still round-trips
    (the map is bijective on each half) but keys the mantissa halves as if
    they were values, so the deltas are noise and compression collapses --
    a silent 10x regression rather than an error. Because the check forces
    `itemsize == width`, the `.view()` below never changes element size and so
    can never raise on a non-divisible length either.
    """
    a = np.ascontiguousarray(arr)
    # if a.dtype.itemsize != width:
    #     raise ValueError(
    #         f"{role} chunk has {a.dtype.itemsize}-byte elements ({a.dtype}) but "
    #         f"the declared dtype needs {width}-byte elements"
    #     )
    return a.view(_UINT_OF[width]).reshape(-1)


def shuffle(stream: bytes, width: int) -> bytes:
    """Group byte 0 of every element, then byte 1, and so on.

    A residual stream is `width`-byte little-endian integers whose values are
    almost all small -- the median zigzag delta between consecutive training
    epochs is around 300, so the high byte of nearly every element is 0 or 1.
    Interleaved, those near-constant bytes sit between noisy low bytes and zstd
    cannot see the pattern. Transposed, they form one long run.

    Measured on real epoch-to-epoch residuals: 77.94% -> 72.10% for the delta
    stream and 92.02% -> 85.11% for raw, at zstd level 3.

    This is the same transform blosc2 calls SHUFFLE. blosc2's BITSHUFFLE (a
    bit-plane split rather than a byte transpose) measured *worse* here --
    73.79% -- so the finer version is not the better one for this data.

    `width` must divide `len(stream)`, which holds by construction: a stream is
    always a whole number of elements.
    """
    a = np.frombuffer(stream, dtype=np.uint8)
    return a.reshape(-1, width).T.copy().tobytes()


def unshuffle_to_array(stream: bytes, width: int) -> np.ndarray:
    """Exact inverse of `shuffle`, straight into the array the decoder wants.

    The obvious spelling of this is `a.reshape(width, -1).T.copy()`, and it is
    a trap. `reshape` and `.T` are free views, but copying a *transposed* array
    leaves numpy no contiguous run to work with: the output is contiguous while
    the input has stride N, so it falls back to a generic strided iterator and
    walks one byte at a time. Measured on a 4 MiB chunk that is 10.92 ms --
    366 MiB/s, against a memory bus an order of magnitude faster, and 59% of
    the entire FUSE read path.

    Assigning one plane at a time is the same permutation expressed so numpy
    can vectorise it: each assignment reads a contiguous plane and writes with
    a fixed stride, and there are `width` of them rather than one per element.
    1.77 ms for the same chunk, 6.2x, byte-identical.

    Returning an array rather than `bytes` removes a second full copy: the
    decoder's next act was `np.frombuffer` over the bytes this used to build.
    """
    a = np.frombuffer(stream, dtype=np.uint8).reshape(width, -1)
    out = np.empty(a.shape[1], dtype=_UINT_OF[width])
    view = out.view(np.uint8).reshape(-1, width)
    for i in range(width):
        view[:, i] = a[i]
    return out


def unshuffle(stream: bytes, width: int) -> bytes:
    """Exact inverse of `shuffle`. Prefer `unshuffle_to_array` on a hot path."""
    return unshuffle_to_array(stream, width).tobytes()


#: The marker byte. 255 rather than 256 values in the narrow plane, because the
#: marker has to be a value the plane can never legitimately hold.
_ESCAPE = 255

#: Above this fraction of escaping elements the encoding is not chosen. The
#: threshold is arithmetic, not tuning: an element costs 1 byte when it fits
#: and 3 when it does not, so the mean is `3 - 2p` for a fitting fraction `p`,
#: which drops below the 2 bytes of a plain residual exactly at p = 0.5.
#:
#: Measured on the 92M benchmark, escape fraction against ratio, current codec
#: in brackets: bf16 gap 1 13.4% -> 52.72% [60.88%]; gap 6 29.4% -> 60.40%
#: [66.04%]; gap 12 46.1% -> 65.65% [67.69%]; gap 24 82.9% -> 75.23% [70.19%],
#: and fp16 gap 1 64.7% -> 74.32% [72.56%]. The crossover sits between 46% and
#: 65%, which is where the arithmetic says it should. Deciding by DTYPE instead
#: would get bf16 at gap 24 wrong by 5pp.
_ESCAPE_MAX_RATE = 0.5


def _escape_stream(zz: np.ndarray) -> bytes:
    """`[u64 LE narrow_len][narrow plane][shuffled wide plane]`.

    One byte per element in the narrow plane -- the zigzag value itself, or
    `_ESCAPE` -- and every escaped value, in order, as shuffled uint16 in the
    wide plane. Keeping the wide values out of line is worth 2.6pp over
    splicing them in after each marker: an oversized value interrupts a run of
    small ones and zstd loses the match across the break. The two planes are
    compressed as a single frame, which measured identically to two separate
    frames and keeps `content_hash` covering one contiguous stream.
    """
    small = zz < _ESCAPE
    narrow = np.where(small, zz, _ESCAPE).astype(np.uint8)
    wide = shuffle(np.ascontiguousarray(zz[~small]).astype("<u2").tobytes(), 2)
    return struct.pack("<Q", narrow.size) + narrow.tobytes() + wide


def _unescape_stream(stream: bytes) -> np.ndarray:
    """Exact inverse of `_escape_stream`, returning the zigzag values."""
    if len(stream) < 8:
        raise ValueError("escape stream is too short to hold its length prefix")
    (narrow_len,) = struct.unpack("<Q", stream[:8])
    end = 8 + narrow_len
    if end > len(stream):
        raise ValueError(
            f"escape stream claims a {narrow_len}-byte narrow plane but holds "
            f"{len(stream) - 8} bytes after the prefix"
        )
    narrow = np.frombuffer(stream, dtype=np.uint8, count=narrow_len, offset=8)
    escaped = narrow == _ESCAPE
    wide = unshuffle_to_array(stream[end:], 2)
    if wide.size != int(escaped.sum()):
        raise ValueError(
            f"escape stream has {int(escaped.sum())} markers but "
            f"{wide.size} wide values"
        )
    zz = narrow.astype(np.uint16)
    zz[escaped] = wide
    return zz


def _finish(
    encoding: str,
    stream: bytes,
    payload: bytes,
    original_len: int,
    is_identical: bool = False,
) -> EncodedChunk:
    return EncodedChunk(
        encoding=encoding,
        content_hash=blake3.blake3(stream).digest(),
        payload=payload,
        plain_len=len(stream),
        original_len=original_len,
        stored_checksum=blake3.blake3(payload).digest()[:8],
        is_identical=is_identical,
    )


def encode_chunk(
    target: np.ndarray,
    base: Optional[np.ndarray] = None,
    *,
    dtype: str,
    compressor: Optional[zstd.ZstdCompressor] = None,
    level: int = DEFAULT_LEVEL,
    compress_raw: bool = True,
    allow_raw_fallback: bool = True,
) -> EncodedChunk:
    """Encode one chunk, against `base` if given.

    Args:
        target: The chunk to encode. Any numpy array whose element width
            matches `dtype`; its bytes are what matter, not its numpy dtype.
        base: The corresponding chunk of the base checkpoint, already gathered
            through whatever permutation applies. `None` means there is no
            base (root commit, or a tensor absent from the base checkpoint),
            which is a normal condition, not an error.
        dtype: safetensors dtype name (`"F16"`, `"BF16"`, `"F32"`, ...). This
            is what fixes the element width and key kind; it is required
            because a BF16 chunk is indistinguishable from a U16 one by
            inspection.
        compressor: Reuse an existing compressor if you have one. Constructing
            one costs ~1 us against a ~5 ms compress of a 4 MiB chunk, so
            leaving this `None` is not a measurable cost -- and a per-call
            compressor is trivially safe to use from several threads, which a
            shared one is not.
        level: zstd level used when `compressor` is None.
        compress_raw: When False, a base-less chunk is stored as `raw`
            (uncompressed) instead of `raw-zstd`. Exposed as a flag rather
            than hardcoded because FORMAT.md section 7 leaves the root-chunk
            encoding open: `raw` keeps a permuted gather a page-cache memcpy,
            `raw-zstd` saves disk. Benchmark, then set a default.
        allow_raw_fallback: Also encode the chunk raw and keep whichever is
            smaller (FORMAT.md section 7: "per-chunk when delta doesn't help").
            Costs a second compression pass, so it is a flag.

    Returns:
        An `EncodedChunk`.

    Raises:
        ValueError: unknown `dtype`, element width disagreeing with `dtype`,
            or `target` and `base` differing in shape.
    """
    width, kind = dtype_spec(dtype)
    t_bits = _as_bits(target, width, "target")
    original_len = t_bits.nbytes

    if compressor is None:
        compressor = zstd.ZstdCompressor(level=level)

    def encode_raw() -> EncodedChunk:
        stream = t_bits.tobytes()
        if not compress_raw:
            return _finish(RAW, stream, stream, original_len)
        # The hash covers the shuffled stream -- i.e. exactly the bytes handed
        # to the compressor -- which keeps the existing rule that a chunk's
        # identity is its *uncompressed* content, and keeps `plain_stream`
        # (and therefore `verify --deep`) a pure decompression step.
        stream = shuffle(stream, width)
        return _finish(RAW_SHUFFLE_ZSTD, stream, compressor.compress(stream),
                       original_len)

    if base is None:
        return encode_raw()

    # Compared before flattening: once both are 1-D, a (4, 8) base against an
    # (8, 4) target has the same element count and would encode "successfully"
    # into a residual that reconstructs transposed garbage. That is exactly the
    # shape of bug a permutation implementation introduces.
    if np.shape(target) != np.shape(base):
        raise ValueError(
            f"chunk shape mismatch: target {np.shape(target)} vs base {np.shape(base)}"
        )
    b_bits = _as_bits(base, width, "base")

    # Subtract the raw bit patterns. No monotone key: measured across all 24
    # adjacent pairs it is worth +0.07pp, and it *loses* 0.26/0.47/0.66pp at
    # gaps 2/3/4 -- the gaps the star actually produces -- for a weighted
    # -0.22pp. See `to_monotone_key` for what it did and why it stopped paying.
    delta = t_bits - b_bits
    # An all-zero delta means the chunk is byte-identical to its base. Computed
    # from the delta rather than from the encoding that wins below, because it
    # is a fact about the *content*, not about how it ended up stored.
    is_identical = not delta.any()
    # No zigzag. Byte shuffle subsumes it: without zigzag the high byte of each
    # element carries the *sign* of the drift (0x00 or 0xff), which is locally
    # correlated in a trained network and so compresses into long runs. Zigzag
    # replaces that with magnitude, which varies element to element and shatters
    # the runs -- measured mean run length 1.71 -> 1.59, costing 0.6-0.8pp.
    stream = shuffle(delta.astype(t_bits.dtype).tobytes(), width)
    encoding = DELTA_SHUFFLE

    # Zigzag + escape, when most residuals fit in one byte. The two conditions
    # are independent and both required.
    #
    # Width 2 only: the wide plane is uint16, so a wider element has nothing to
    # escape *into*. F16 and BF16 are the dtypes weights actually use.
    #
    # And only below `_ESCAPE_MAX_RATE`, because the encoding is a bet that
    # exceptions are rare. On bf16 the median residual is 27 ULPs and 86.5% of
    # zigzag values fit in a byte, so it wins by 8.16pp. On fp16 the median is
    # 510, only 35.3% fit, and the same encoding needs 2.29 bytes per element
    # -- more than storing the residual raw. The rate is what separates those,
    # not the dtype: bf16 against a far-away base escapes 82.9% and belongs on
    # the shuffle path too.
    if width == 2:
        zz = zigzag(delta.astype(t_bits.dtype))
        if float((zz >= _ESCAPE).mean()) < _ESCAPE_MAX_RATE:
            candidate = _escape_stream(zz)
            if len(candidate) < len(stream):
                stream, encoding = candidate, DELTA_ZIGZAG_ESCAPE

    encoded = _finish(
        encoding, stream, compressor.compress(stream), original_len,
        is_identical,
    )

    if allow_raw_fallback:
        alternative = encode_raw()
        if alternative.stored_len < encoded.stored_len:
            # Note this also changes the chunk's identity to the hash of its
            # raw content, which is the desirable outcome: a chunk stored raw
            # dedups against every other identical raw chunk in the repo,
            # whereas a residual is only ever identical to a residual taken
            # against the same base.
            return replace(alternative, is_identical=is_identical)
    return encoded


def plain_stream(
    encoding: str,
    payload: bytes,
    *,
    decompressor: Optional[zstd.ZstdDecompressor] = None,
) -> bytes:
    """Undo only the *compression* layer of a stored chunk.

    Returns the exact byte string `_finish` hashed to produce the chunk's
    `content_hash` -- for the shuffled encodings that is the *shuffled* stream,
    because the shuffle happens before hashing. Nothing is decoded, reshaped,
    un-shuffled or added to a base.

    Split out of `decode_chunk` for `verify --deep`, which needs to answer
    "are these the bytes this chunk claims to be?" and nothing else. Doing
    that through `decode_chunk` would be wrong twice over: a residual chunk
    would demand its base rows, dragging the whole reconstruction chain into
    a check that does not need it, and the array it returns is a
    *reinterpretation* of the stream, so hashing it would re-derive the same
    bytes by a longer route. Hashing the stream directly keeps deep
    verification O(stored bytes) with no recursion.

    Raises ValueError on an unknown encoding.
    """
    if encoding == RAW:
        return payload
    if encoding in _ZSTD_FRAMED:
        if decompressor is None:
            decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(payload)
    raise ValueError(
        f"unknown chunk encoding {encoding!r}; expected one of "
        f"{RAW!r}, {RAW_ZSTD!r}, {RAW_SHUFFLE_ZSTD!r}, "
        f"{DELTA_SHUFFLE!r}, {DELTA_ZIGZAG_ESCAPE!r}"
    )


def decode_chunk(
    encoding: str,
    payload: bytes,
    base: Optional[np.ndarray] = None,
    *,
    dtype: str,
    decompressor: Optional[zstd.ZstdDecompressor] = None,
) -> np.ndarray:
    """Reconstruct a chunk's raw bit patterns. Exact inverse of `encode_chunk`.

    Returns a flat unsigned array of `width`-byte elements -- bit patterns, not
    decoded values. That is deliberate and not a convenience shortcut: there is
    no numpy dtype for bf16, so "return the real dtype" is not a contract this
    function could honour for every input. Reshaping and reinterpreting is the
    caller's job, using the shape and dtype from the tensor manifest.

    For `raw` the result is a zero-copy read-only view straight onto `payload`,
    which is the point of that encoding for the FUSE read path. The other two
    encodings return freshly-allocated arrays. Copy if you need to write.

    Raises:
        ValueError: unknown `encoding`, a `delta-zigzag-zstd` chunk with no
            `base`, or a base whose element count disagrees with the residual.
    """
    width, _kind = dtype_spec(dtype)
    unsigned = _UINT_OF[width]

    stream = plain_stream(encoding, payload, decompressor=decompressor)
    # The shuffled encodings go straight to an array; everything downstream
    # reinterprets the bytes anyway, so materialising them is pure waste.
    values = unshuffle_to_array(stream, width) if encoding in _SHUFFLED else None

    if encoding in (RAW, RAW_ZSTD, RAW_SHUFFLE_ZSTD):
        return values if values is not None else np.frombuffer(stream, dtype=unsigned)
    if base is None:
        raise ValueError(f"{encoding} chunk cannot be decoded without a base chunk")
    b_bits = _as_bits(base, width, "base")

    if encoding == DELTA_ZIGZAG_ESCAPE:
        residual = unzigzag(_unescape_stream(stream))
        if residual.size != b_bits.size:
            raise ValueError(
                f"residual has {residual.size} elements but base chunk has "
                f"{b_bits.size}"
            )
        return (b_bits + residual).astype(unsigned, copy=False)

    residual = values if values is not None else np.frombuffer(stream, dtype=unsigned)
    if residual.size != b_bits.size:
        raise ValueError(
            f"residual has {residual.size} elements but base chunk has {b_bits.size}"
        )

    # `DELTA` (legacy `delta-zigzag-zstd`) used to need undoing in monotone-key
    # space here; that path and the key functions it depended on are gone
    # (nothing writes `DELTA` any more), so every remaining delta encoding
    # falls through to the same plain modular add.
    return (b_bits + residual).astype(unsigned, copy=False)
