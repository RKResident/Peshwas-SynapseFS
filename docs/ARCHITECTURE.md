# SynapseFS — Implementation Manual

Revision 2 · 2026-08-28

---

## 0. How to use this document

This is written to be **reimplemented from**, not merely read. Every on-disk
format is specified to the byte; every algorithm is given with a worked
example; every module says what its *core* is and what is hardening you can
skip on a first pass.

The existing Python is a reference implementation, not a target to reproduce.
It carries defensive code that the PS does not ask for. Where a section says
**skippable**, the code has it and you probably should not.

Reading order for a rewrite: §1 → §2 → §3 (formats) → §4 (algorithms) → §5
(modules) → §7 (traps) → §8 (build order).

**New to the project?** Read `PRIMER.md` first — it explains the problem, the
core ideas, and every term used below, assuming no familiarity with the
codebase. This document assumes all of it.

| companion doc | covers |
|---|---|
| `PRIMER.md` | the ideas and vocabulary, for someone new |
| `SYSTEMS_PRIMER.md` | FUSE, inodes, caching, RSS, threads — assumes no systems background |
| `FUSE.md` | `synapsefs/fuse/`, file by file and function by function |
| `ALIGNMENT.md` | `synapsefs/align/`, file by file and function by function |
| `FORMAT.md` | the same formats, with more design rationale and history |
| `CLI.md` | command surface, flags, exit codes |
| `INDEX_V2.md`, `CHUNK_STORE.md` | proposed alternatives, not implemented |
| `kris_docs.md` | a parallel spec by another team member; **not** what this code does |

---

## 1. What you are building

A version control system for neural-network checkpoints, plus a read-only
filesystem that serves them.

**The guarantee.** For any commit, the checkpoint you get back is
**byte-for-byte identical** to the `.safetensors` file that went in — header,
metadata, padding and all. Not numerically close. Byte-identical.

Everything else follows from defending that guarantee cheaply:

- storing residuals instead of whole files, because consecutive checkpoints are
  ~99% the same weights (§4.1)
- aligning permuted networks first, because two functionally identical models
  can have their hidden units in different orders and a naive diff sees 100%
  change (§4.6)
- content addressing, because the same chunk stored twice should cost once, and
  because a hash chain is what makes tampering detectable (§4.5)

### 1.1 State of the build

| PS module | grade | state |
|---|---|---|
| 1 Alignment & compression | 25% | codec done; aligner **wired in** (§8); scaling measured to n=10,000 at 100% recovery (§4.6.3); per-tensor adaptive anchoring behind `--anchor-policy` (§4.3.2) |
| 2 Filesystem (FUSE) | 25% | done; runs live — mounts, lists, and reads back byte-identical for full and residual commits at 206 MiB/s (§4.4.1). `forget()` and the `(parent, name)` inode index are in |
| 3 Cryptographic integrity | 20% | done (tiers to be re-based on §4.5.2) |
| 4 Networking & CLI | 15% | CLI complete incl. `merge`; `push`/`pull`/`serve` merged in |
| 5 Documentation | 15% | this + `PRIMER.md` + `FORMAT.md` + `CLI.md` + `README.md` |

**Architecture scope.** Only BatchNorm, LayerNorm, convolution and dense layers
need supporting. Attention, embeddings and LoRA are out of scope, which is why
`config_parser.UNSUPPORTED_HINTS` rejects them outright rather than wiring a
topology it cannot verify.

---

## 2. Conventions

Six rules. Every one of them is load-bearing somewhere far from where it is
defined.

### 2.1 Hashing

BLAKE3-256 everywhere. Hex (64 chars) when naming an object, raw 32 bytes when
indexing one.

**Three different hashes exist and conflating them breaks things silently:**

| name | hashes | answers |
|---|---|---|
| chunk `content_hash` | the chunk's **uncompressed** stream | *which chunk is this?* — survives recompression |
| `stored_checksum` | the chunk's **stored** (compressed) bytes, first 8 | *did the disk rot, or was this substituted?* — no decompression needed |
| tensor `content_hash` | the tensor's **fully reconstructed** bytes | *is this the same weights?* — independent of base, chunking, encoding |

The third is not a manifest hash. Two branches can hold identical weights whose
manifests differ (aligned against different bases), so comparing manifest hashes
would report a conflict on a tensor nobody touched.

### 2.2 Canonical JSON

Anything hashed is serialised exactly one way:

```python
json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
```

`{"a":1,"b":2}` and `{"b":2,"a":1}` are the same object and must not be two
hashes, or dedup stops working between two code paths that build dicts in
different orders.

### 2.3 Identity is `null`, never `[0,1,2,...]`

A permutation that does nothing is `None` in memory and `null` on disk. This
makes the fine-tuning fast path free — you skip the gather instead of
performing an identity gather — rather than a special case.

### 2.4 Permutation direction

> **`p[i]` is the BASE index that TARGET index `i` was diffed against.**

Everything else is derived from that sentence:

- to use it: `aligned_base = base[p]` — pull the base into target order
- composing: `compose(f, s)` is *defined* by `A[compose(f,s)] == A[f][s]`
- the LAP cost matrix is target-major, so SciPy's `col_ind` **is** `p` — there
  is no inversion anywhere in the pipeline

If you find yourself needing `invert()` on the main path, the convention has
been broken upstream. **No structural check catches a reversal** — both
directions are valid bijections of the right length. Only the tensor
`content_hash` sees it. See §7.1.

### 2.5 Logical 2-D shape

Every tensor is `[rows, cols]` with `rows = shape[0]`, `cols = prod(shape[1:])`.
A 1-D tensor has `cols == 1`. A 0-d scalar has `rows == cols == 1`. Chunk
boundaries are always row boundaries. Nothing in the storage layer ever thinks
about a tensor's real rank.

### 2.6 Four dtypes

| safetensors | width | key kind |
|---|---|---|
| `F16`, `BF16`, `F32` | 2, 2, 4 | `FLOAT` (sign-magnitude) |
| `I64` | 8 | `SINT` (two's complement) |

That is not an arbitrary subset. The PS evaluates fp16/bf16 checkpoints, and
`I64` is forced: `model.half()` leaves BatchNorm's `num_batches_tracked` as
int64, so **every real fp16 checkpoint is a mixed-dtype file.** Supporting more
is wasted work.

---

## 3. On-disk formats

Complete. If two things disagree, the byte layout wins.

### 3.1 Directory layout

```
.synapse/
    HEAD
    refs/heads/<branch>
    objects/
        <ab>/<62-hex>            loose objects: commit, checkpoint-manifest,
                                 tensor-manifest, header, config   (§3.4)
        <ab>/<cd>/<60-hex>       chunk payloads                    (§3.5)
        tmp/                     staging for atomic writes
        incoming/                transfer packs, transient         (§3.6)
```

Mutable: `HEAD` and `refs/heads/*`. Everything else is immutable and
content-addressed.

**There is no persistent packfile and no pack index.** Chunks are loose
objects, two shard levels deep. Packs survive only as a *wire* format for
`push`/`pull` (§3.6) — built on demand, unpacked on arrival, deleted.

### 3.2 `HEAD` and refs — plain text

```
HEAD, attached:   "ref: refs/heads/main\n"
HEAD, detached:   "4d8e2f...c5674a79a2b\n"          (64 hex + newline)
refs/heads/main:  "4d8e2f...c5674a79a2b\n"          (64 hex + newline)
```

Three states, and you must distinguish them:

| `HEAD` | branch ref | meaning |
|---|---|---|
| `ref: refs/heads/main` | absent | **unborn** — after `init`, before the first commit |
| `ref: refs/heads/main` | present | attached |
| raw hash | — | detached |

The absence of a `ref: ` prefix is the *only* on-disk difference between
attached and detached.

### 3.3 Object paths

**One scheme for every object kind**, chunks included:

```
objects/<h[0:2]>/<h[2:4]>/<h[4:64]>
```

For `1234567890…`: `objects/12/34/567890…`. The filename is the hash **minus
the shard prefix** (60 hex), matching the loose-object convention in
`kris_docs.md`.

An earlier draft gave chunks two shard levels and everything else one, on the
assumption that chunks vastly outnumber objects. **Measured, they do not** — 25
commits of the STL-10 model produce 1,002 loose objects against 810 chunks,
because every commit writes 38 tensor-manifests plus a checkpoint-manifest plus
a commit. Split depths bought nothing and cost an "is this a chunk or an
object?" ambiguity at every call site. One scheme, no special cases.

**Cost of the second level, measured:** ~17,500 directories ≈ 68 MiB of
directory blocks at 20,000 objects, against ~1 MiB at one level (random hashes
give roughly one object per leaf directory long before the buckets fill). If
you expect fewer than ~100k objects total, **one level is the better default**
and the scheme is otherwise identical. A *third* level costs 146 MiB — more
than the data — so never add one.

### 3.4 The five loose object kinds

All are canonical JSON (§2.2) except `header`, which is raw bytes.

#### 3.4.1 `commit`

```json
{
  "checkpoint_manifest": "2009317d…",
  "checkpoint_name": "epoch25.safetensors",
  "full": true,
  "message": "epoch25",
  "parents": ["71c84462…"],
  "timestamp": "2026-08-27T20:26:47Z"
}
```

| field | type | notes |
|---|---|---|
| `checkpoint_manifest` | hex | the tree this commit points at |
| `parents` | list of hex | `[]` for root, 1 normal, ≥2 merge |
| `timestamp` | string | `%Y-%m-%dT%H:%M:%SZ`, UTC |
| `message` | string | |
| `full` | bool | this commit stored its checkpoint outright (a star hub, §4.3) |
| `checkpoint_name` | string | **basename only**; where `checkout` writes it |

`full` and `checkpoint_name` are derived/incidental data inside a hashed
immutable object, which is normally wrong. Justification: computing `full` on
demand means loading every tensor-manifest of every ancestor to answer a
one-bit question, on the hot path of every commit. `checkpoint_name` has
nowhere else to live — the safetensors header names tensors, not files.

**Re-strip `checkpoint_name` with `basename()` when reading.** A commit from a
peer carrying `../../.ssh/authorized_keys` is otherwise a write-anywhere
primitive.

#### 3.4.2 `checkpoint-manifest`

```json
{
  "header_object": "81819084…",
  "tensors": { "blocks.0.bn1.bias": "d7aa8573…", "…": "…" },
  "topology_config_hash": "61b029dd…"
}
```

`tensors` maps tensor name → tensor-manifest hash. `topology_config_hash` is
the stored `config.json` (or `null`).

#### 3.4.3 `tensor-manifest`

```json
{
  "name": "blocks.0.bn1.bias",
  "dtype": "F16",
  "shape": [96],
  "content_hash": "dec935e9…",
  "base_tensor_manifest": null,
  "base_row_permutation": null,
  "base_col_permutation": null,
  "col_block_size": 1,
  "chunks": [
    {
      "row_start": 0,
      "row_end": 95,
      "encoding": "raw-zstd",
      "object": "dec935e9…",
      "plain_len": 192,
      "stored_checksum": "bbaa17e69c87186d"
    }
  ]
}
```

| field | notes |
|---|---|
| `dtype` | uppercase safetensors name (`FORMAT.md` §7's lowercase example is wrong) |
| `content_hash` | `blake3` of the tensor's reconstructed **data bytes only** (no header), in its own row order, computed incrementally chunk by chunk so no tensor is materialised |
| `base_tensor_manifest` | the manifest this was diffed against, or `null` |
| `base_row_permutation` | hash of a permutation object, or `null` for identity |
| `base_col_permutation` | same, for the column axis |
| `col_block_size` | columns per permuted unit; 1 unless a conv→linear flatten (§4.6) |
| `chunks[].row_end` | **inclusive** |
| `chunks[].object` | chunk content hash — also its path (§3.3) |
| `chunks[].plain_len` | decompressed length, so a reader can size its buffer |
| `chunks[].stored_checksum` | first **8 bytes** of `blake3(chunk file bytes)`, hex-encoded — so **16 hex characters** |

The last two used to live in the pack index. With no index they have to move
here, and **moving them is an upgrade, not a compromise** — see §4.5.2. Note
`stored_len` did *not* move: it is `stat().st_size`.

**Invariant:** `base_tensor_manifest` is non-null **if and only if** at least
one chunk has `encoding == "delta-zigzag-zstd"`. A delta with no base is
unresolvable; a base with no delta makes GC retain a subtree nothing reads.

**Why not a Merkle root over the chunk hashes?** Because a chunk hash is a hash
of the *representation*, not the content: a delta chunk hashes the residual
stream **against one specific base**. Two commits holding byte-identical
weights diffed against different bases produce different chunk hashes, so their
Merkle roots would differ — and merge, which asks "are these tensors the same?",
would report a conflict on a tensor nobody touched. `content_hash` has to be
independent of base, chunk boundaries and encoding, and only hashing the
reconstructed bytes achieves that.

*A per-chunk **plaintext** hash (of decoded rows, not the residual) would be a
reasonable **addition** — it buys incremental verification of one tensor region
without reconstructing the whole tensor, for 32 bytes per chunk. It cannot
replace `content_hash`.*

> In the example, `content_hash == chunks[0].object`. Not a coincidence: with a
> single `raw-zstd` chunk, the chunk's uncompressed stream *is* the whole
> tensor. With two chunks, or any delta, they diverge.

#### 3.4.4 `header`

The source file's safetensors header, **verbatim, including the 8-byte length
prefix**. Never re-serialised — that is what makes byte-exactness achievable
(§4.4).

#### 3.4.5 `permutation` *(not yet written by any code path)*

Packed `int32` little-endian, no header, no length prefix. Length is
`file_size / 4`. Identity is stored as `null` in the manifest and has no
object.

### 3.5 Chunk files

A chunk is **a contiguous range of rows from exactly one tensor**, encoded and
stored as its own file at `objects/<ab>/<cd>/<60-hex>` (§3.3).

#### The file itself

**There is no framing whatsoever.** The file is the payload and nothing else:

| offset | size | content |
|---|---|---|
| `0` | `S` | the encoded payload, where `S = stat().st_size` |

No magic number, no version, no length prefix, no checksum, no trailer. That is
deliberate — a chunk file carries no self-description, because everything about
it is recorded either in its **path** or in the **tensor-manifest** that
references it.

#### Worked example

Take `blocks.0.bn1.bias`: `dtype F16`, `shape [96]`. Under §2.5 that is 96
rows × 1 column. It fits in one chunk, so the manifest says:

```json
{ "row_start": 0, "row_end": 95, "encoding": "raw-zstd",
  "object": "dec935e9…", "plain_len": 192, "stored_checksum": "bbaa17e6…" }
```

To read those rows:

1. **Find the file.** `object` is `dec935e9329067a0…`, so the path is
   `objects/de/c9/35e9329067a0…`.
2. **Read it whole.** Say it is 158 bytes — that is `stored_len`; nothing
   records it because `stat()` does.
3. **Undo the compression**, chosen by `encoding`:
   - `raw` → the bytes are already the stream
   - `raw-zstd` / `delta-zigzag-zstd` → one zstd frame, starting `28 b5 2f fd`
   
   You now hold 192 bytes = `plain_len` = 96 elements × 2 bytes. ✓
4. **Interpret the stream**, again by `encoding`:
   - `raw-shuffle-zstd` → un-shuffle and you have the tensor's bit patterns,
     little-endian, row-major. Done.
   - `delta-zigzag-escape-zstd` → parse the two planes, un-zigzag, add to base
   - `delta-shuffle-zstd` → un-shuffle, then treat as a **residual**, not
     values. Fetch the same row range from
     `base_tensor_manifest`, convert both to monotone keys, add, convert back
     (§4.1).

#### The encodings, and what the payload actually holds

| `encoding` | bytes `0..4` | after decompression you hold |
|---|---|---|
| `raw-shuffle-zstd` | `28 b5 2f fd` | byte-shuffled tensor bit patterns |
| `delta-shuffle-zstd` | `28 b5 2f fd` | byte-shuffled residual stream |
| `delta-zigzag-escape-zstd` | `28 b5 2f fd` | the escape layout below |
| `raw` | *(no frame)* | tensor bit patterns, unshuffled |
| `raw-zstd` *(legacy)* | `28 b5 2f fd` | tensor bit patterns, unshuffled |
| `delta-zigzag-zstd` *(legacy)* | `28 b5 2f fd` | zigzag residual, unshuffled |

#### `delta-zigzag-escape-zstd`, byte for byte

The decompressed stream is **not** one uniform block. It has internal
structure, and it is the only encoding of which that is true:

| offset | size | content |
|---|---|---|
| `0` | `8` | `narrow_len`, `uint64` little-endian — the element count |
| `8` | `narrow_len` | the **narrow plane**: one byte per element |
| `8 + narrow_len` | rest | the **wide plane**: byte-shuffled `uint16` |

A narrow byte is the zigzag residual itself when it is `0..254`, or `0xFF` to
say "this one did not fit". The escaped values appear in the wide plane in
element order, one `uint16` each, and the number of them must equal the number
of `0xFF` markers — a mismatch is corruption and raises.

`content_hash` covers this whole stream, prefix included, exactly as for the
other encodings. Both planes go in **one** zstd frame; two separate frames
measured identically (−0.00pp) and would have split the hash across two blobs.

The wide plane is kept out of line rather than spliced in after each marker.
That is worth **2.6pp**: an oversized value dropped between small ones breaks
the run the compressor was matching on.

Decoding is a single pass — read a byte, take it or pull the next `uint16` from
the wide plane, then un-zigzag and add to the base.

In all three cases the decompressed length is `plain_len`, and it always equals
`(row_end - row_start + 1) × cols × width` — the residual stream is exactly the
size of the tensor region it encodes (§4.1).

#### Where every field lives

Nothing here is in the file:

| quantity | comes from |
|---|---|
| `content_hash` | **the path** — concatenate `<ab> + <cd> + filename` |
| `stored_len` | `stat().st_size` |
| `plain_len` | manifest `chunks[].plain_len` |
| `stored_checksum` | manifest `chunks[].stored_checksum` |
| `encoding` | manifest `chunks[].encoding` |
| which rows | manifest `chunks[].row_start` / `row_end` (inclusive) |
| `dtype`, `shape` | the tensor-manifest |
| the base to add | the tensor-manifest's `base_tensor_manifest` |

#### The two hashes, again

They cover **different bytes on purpose**:

```
path            == blake3(decompressed stream)     identity; survives recompression
stored_checksum == blake3(file bytes)[:8]          rot/tamper check; no decompression
```

`fsck` checks both:

```
blake3(plain_stream(encoding, file_bytes))  ==  <ab> + <cd> + filename
blake3(file_bytes)[:8]                      ==  manifest.stored_checksum
```

### 3.6 Transfer — one chunk at a time

**No pack format is needed, for storage or for the wire.** `pack.py`,
`index.py` and `packset.py` all go.

PS module 4b asks that "only missing blocks are transferred" and that an
interrupted sync "can resume correctly without re-transferring already-received
blocks". Loose chunks satisfy both directly and more simply than a container
would:

```
client -> server:  HAVE <commit hash>
server -> client:  list of object + chunk hashes reachable from it
client:            filter to those where not path_for(h).exists()
client -> server:  GET <hash>          (pipelined, do not wait per response)
server -> client:  <raw file bytes>
client:            atomic_write(path_for(h), bytes)
```

Resume is *"which files exist?"* — every chunk is independently complete, so a
transfer killed at any point leaves a valid partial repository and the next run
re-requests exactly the difference. There is no partially-written container to
detect, discard or scan.

**Why a pack is not worth it at this scale.** A 7B model is ~3,326 chunks
(§4.2). Sent strictly serially at 1 ms RTT that is 3.3 s of pure latency, and
pipelining requests removes almost all of it; on a LAN the whole question
disappears. A pack starts to pay only around ~100k chunks or over a high-RTT
link, and it can be added later without touching the storage layer — which is
the point of having dropped it from storage first.

*The one thing to get right:* the client must verify each received chunk
against the hash it asked for **before** writing it (`blake3(plain_stream(...))
== requested hash`), or a hostile peer can seed arbitrary content into the
store. That check is the same one `verify --deep` performs.

### 3.7 The input format: safetensors

```
offset  size        field
0       8           header_len   u64 LE
8       header_len  header JSON
8+hl    …           tensor data, contiguous
```

Header JSON maps name → `{"dtype", "shape", "data_offsets": [begin, end]}`,
offsets relative to the start of the data region. `__metadata__` is a reserved
key holding no tensor.

**Two traps.** JSON key order need not match `data_offsets` order — writing
tensors in `names()` order produces a valid but byte-different file (§7.2).
And the reference `safetensors` library **cannot read bf16 at all**; both
`get_slice` and `get_tensor` raise. You must mmap and slice yourself.

---

## 4. The algorithms

### 4.1 The residual codec

The whole compression idea in one line: **subtract, then group the bytes that
barely change.**

**Step 1 — reinterpret the bits as unsigned integers** of the tensor's element
width (2 bytes for F16/BF16, 4 for F32, 8 for I64). No conversion happens; this
is a view, not a cast.

**Step 2 — subtract, mod 2ⁿ.** `delta = target_bits - base_bits`.

Floating-point subtraction would be wrong here: `(a - b) + b` does not
reliably return `a`, and the project's whole promise is byte-exactness.
Modular integer subtraction is a bijection, so it is exactly reversible with
no rounding anywhere.

Two facts about what this produces, both of which the next step depends on:

- For a weight that drifted slightly **upward**, the delta is a small positive
  number: high bytes `0x00`.
- For one that drifted **downward**, the borrow out of the low bits propagates
  all the way up, so the delta is a small negative number in two's complement:
  high bytes `0xff`.

So the high byte of each element is **sign extension** — `0x00` or `0xff` —
for the ~66% of weights whose drift fits in 8 bits. And drift direction is
locally correlated in a trained network, so those bytes come in runs.

**Step 3 — byte shuffle.** Transpose so all low bytes are contiguous and all
high bytes are contiguous, turning that sign signal into long runs an LZ
matcher can see:

```python
def shuffle(buf, w):                        # w = element width in bytes
    a = np.frombuffer(buf, np.uint8); n = len(a) // w
    return a[:n*w].reshape(n, w).T.copy().tobytes()
```

**Step 4 — zstd, at level 1.**

Worked example, fp16, three elements:

```
  value(t)     bits  |    value(b)     bits  |  delta (mod 2^16)   high byte
       1.0     3c00  | 1.0009765625    3c01  |             65535        0xff
 1.00195312     3c02 |          1.0    3c00  |                 2        0x00
       0.5     3800  | 0.4995117188    37fe  |                 2        0x00
```

Row 1 wraps to `65535`, i.e. `-1`. The wraparound is not a bug being
tolerated, it is the mechanism — one ULP of drift costs one unit, in either
direction, and the sign lands in the high byte.

Because arithmetic stays in the native width, **the residual stream is exactly
the size of the tensor region.** No widening, no varints.

**Raw fallback.** After encoding a delta, also encode raw and keep whichever is
smaller. Two unrelated tensors produce a delta larger than the tensor itself.

#### What is deliberately *not* in this pipeline

Three transforms were tried, measured, and rejected. All numbers are from real
epoch-to-epoch residuals; see §4.1.1 for why they fail.

| transform | result | verdict |
|---|---|---|
| **zigzag** (map ±small onto small unsigned) | costs 0.6–0.8pp | removed — shuffle subsumes it |
| **monotone key** (order-preserving float-bit fix-up) | +0.07pp at gap 1, −0.26/−0.47/−0.66pp at gaps 2/3/4 | removed — weighted −0.22pp |
| **bit shuffle** (bit-plane transpose, blosc BITSHUFFLE) | 83.6% vs 72.9% | rejected |
| **zero-padded field planes** (sign / exp / mantissa, byte-aligned) | 89.0% vs 73.9% | rejected |
| **diagonal / transposed scan order** | +2.2pp | rejected |

The monotone key is still in the source: `materialize.compare_sources` uses it
to compute ULP distances for `restore --compare`, and the legacy
`delta-zigzag-zstd` encoding needs it to decode. It is simply no longer on the
write path — which also removes the `FLOAT`/`SINT` key-kind distinction from
storage entirely.

#### 4.1.1 One principle explains all four rejections

> **The transform must be a byte-level *permutation*.**

Break byte alignment and you destroy the LZ matches that are doing the work:
after bit-packing, each output byte is assembled from several *different*
elements, so two identical elements no longer produce identical bytes. That
kills bit shuffle, sub-byte field splits, and diagonal scans.

Preserve alignment by *padding* each field to a whole byte and you pay zstd's
per-byte floor on every byte you added. Measured: splitting the high byte into
sign / exponent / mantissa-high collapses the alphabet from 152 distinct values
to 2, 20 and 4 — and still costs 957 KB more, because zstd charges ~1.47
bits/element for a 2-value plane whose true entropy is ~0.97.

There is a second, sharper reason bit-level transforms fail here, and it is a
property of the *data* rather than of zstd:

```
mean run length of the sign bit:  3.06 elements
fraction of runs >= 8 elements:   8.8%

representation                        raw       zstd   entropy/byte
high BYTE plane (1 byte/elem)   3,174,058  1,296,380      2.97 b
sign BIT plane  (8 elem/byte)     396,758    358,874      7.21 b
```

**Bit-packing only pays when runs are at least as long as the packing width.**
A byte plane stores one element per byte, so a run of three same-sign weights
is three identical bytes and zstd's matcher removes them. Bit-packing puts
eight elements in a byte, so a run of three never fills one; consecutive packed
bytes are near-random combinations of eight unrelated signs, and entropy rises
from 2.97 to 7.21 bits per byte. The packed plane compresses to 90% of its raw
size -- effectively not at all. All sixteen planes together come to 6,089,793
bytes against 4,470,523 for two byte planes, **36% worse**.

*The exception worth knowing:* bit-packing a **single** flag costs 1/8 byte per
element regardless of how incompressible it is, which is cheap enough to win
outright. That is why the PFor scheme below uses a bit-packed bitmap and
byte-aligned payload planes. One bit: good. Sixteen planes: bad.

Byte shuffle is essentially the only transform that is both aligned and
non-expanding. A split-point sweep shows the cliff:

```
low k bits in one frame, high 16-k in another:
  k=6  84.46%   k=7  83.67%   k=8  73.92%   k=9  80.58%   k=10  80.18%
                              ^^^^^^^^^^^ the byte boundary
```

#### 4.1.2 Why zstd level 1

Compression is **non-monotone in level** on this data:

| level | ratio | compress | decompress |
|---|---|---|---|
| **1** | **74.79%** | **398 MB/s** | 1163 MB/s |
| 3 | 75.84% | 212 MB/s | 1211 MB/s |
| 9 | 75.17% | 55 MB/s | 1223 MB/s |

Levels 1–2 use zstd's `fast`/`dfast` match-finders; 3+ switch to
`greedy`/`lazy`, which hunt for better matches that do not exist in run-heavy
data and emit more literals. Decompression is ~1.2 GB/s at every level, so the
read path is indifferent.

Other compressors were measured. `lzma preset6` is 0.81pp smaller — and
2 MB/s to compress, 44 MB/s to decompress against zstd's 126 / 389. The FUSE
read path is graded on cold-cache throughput; 9× slower decompression is the
wrong trade for 0.8pp. `zlib`, `bz2` and `brotli` are all worse on both axes.

**Available but not taken:** compressing the two byte planes as *separate*
zstd frames measures 73.92% against 74.54% for one combined frame, because
each plane gets its own entropy model. The low plane can then be stored
uncompressed for the same size and better throughput — it is 8.0-bit noise.
Worth doing; not yet done.

#### 4.1.3 Everything above is conditional on the mantissa width

Read that section as *"measured on fp16"*, because it was, and several of its
conclusions **reverse on bf16**. Both formats are two bytes; they split them
differently — fp16 is 1 sign / 5 exponent / 10 mantissa, bf16 is 1 / 8 / 7 —
and the byte shuffle therefore cuts in a different place.

That changes two things at once. bf16's high byte is sign plus exponent and no
mantissa at all, so it is pure structure; and with three fewer mantissa bits
the same weight movement is ~8× fewer ULPs, so the residuals themselves are
far smaller.

| | fp16 | bf16 |
|---|---|---|
| high plane (zstd / H₀) | 45.20% / 4.44 bits | **27.36% / 1.55 bits** |
| low plane (zstd / H₀) | 100.00% / 8.00 bits | 94.66% / 7.53 bits |
| combined | 72.60% | 61.01% |
| median residual, gap 1 | 510 ULP | **27 ULP** |
| residuals fitting in a byte (zigzag) | 35.3% | **86.5%** |

The 100.00% low plane that bounded §4.1.2's headroom argument is an **fp16
artifact**, not a law: bf16's low byte carries an exponent bit, which is
structured, and it compresses.

Consequences, all measured at gap 1 on the 92M benchmark:

| scheme | fp16 | bf16 |
|---|---|---|
| byte shuffle (the fp16 answer) | **72.56%** | 60.88% |
| zigzag + byte shuffle | 72.48% | 56.79% |
| zigzag + PFor bitmap | 71.65% | 52.77% |
| zigzag + escape byte, wide plane split | 74.32% | **52.72%** |

**Zigzag was removed from the codec for losing on fp16** (−0.08pp at gap 1,
+0.07pp at gap 3 — noise) and is now essential: it is what lets a *negative*
residual fit in the narrow plane at all, taking the fitting fraction from
51.2% to 86.5%. **PFor** was measured at −1.2pp on fp16 and never implemented;
on bf16 it is worth −8pp. The **escape byte** was rejected outright on fp16
and now ties PFor.

The flag width turns out not to matter — an 8-bit `0xFF` marker and a 1-bit
bitmap entry land within 0.05pp — while the **layout** is worth 2.6pp. Same
lesson as bit-shuffling, sub-byte planes, Elf and per-field deltas:
homogeneous byte streams compress, mixed ones do not.

XOR was re-checked against the new encoding and loses: median 63 against
subtraction's 26 at gap 1, because an XOR's magnitude is the position of the
highest differing bit rather than a distance, so it jumps to the next power of
two at every carry boundary. It *beat* subtraction under plain byte shuffle at
gap 1 (60.63% vs 60.88%), which is a reminder that the old ablation's XOR
column is no longer a guide.

**Selection is per chunk, by counting, not by dtype.** An element costs 1 byte
when it fits and 3 when it escapes, so the mean is `3 − 2p` and only beats a
2-byte residual when `p > 0.5`. Measured crossover sits between 46.1% and
64.7% escape, where the arithmetic says it should. A dtype rule would store
5pp *worse* than the old codec on a bf16 commit whose base is a distant anchor
(82.9% escape at gap 24) — a case the star topology produces whenever the
rebase interval widens.

### 4.2 Chunking

```
rows_per_chunk = max(1, chunk_size_bytes // (row_elems * width))     # 4 MiB default
```

Chunks are always whole rows.

**Chunk size IS a compression decision** — it was not before byte shuffle, and
that change inverted the answer. Measured on the 92M-parameter benchmark model,
epoch23 → epoch24, with the current codec:

| chunk size | chunks | stored ratio |
|---|---|---|
| 64 KiB | 3,312 | 80.70% |
| 256 KiB | 791 | 74.61% |
| 1 MiB | 234 | 72.81% |
| **4 MiB** *(default)* | **102** | **72.63%** |
| 16 MiB | 70 | 72.61% |
| 64 MiB | 63 | 72.60% |

**8.1 points** between 64 KiB and 4 MiB. The sweep that used to sit here showed
0.3pp of noise across the same range — it predated byte shuffle and is why this
section previously claimed size did not matter.

The reason is that shuffle builds its byte planes **per chunk**. A 64 KiB chunk
yields a 32 KiB high-byte plane; a 4 MiB chunk yields a 2 MiB one. The
sign-extension runs zstd feeds on average only ~1.2 elements, so it needs a long
plane to accumulate enough matches to outweigh per-frame overhead — and at 3,312
chunks that framing is itself non-trivial. Before shuffle there were no planes,
so the size did not matter.

**The curve is flat above 4 MiB** (72.63% → 72.60% out to 64 MiB), so the
default sits at the knee. Going larger buys ~0.03% and costs:

- **read latency** — a 128 KiB FUSE read decodes its whole containing chunk:
  ~3.4 ms at 4 MiB against ~54 ms at 64 MiB, at ~1.2 GB/s
- **dedup granularity** — one changed weight re-stores the whole chunk
- **peak RSS**, which scales with the in-flight chunk

Encode time is flat across the entire range (1.8–2.4 s), so speed does not enter
the decision.

Keep 4 MiB. If mount latency ever forces a smaller chunk, **1 MiB costs 0.2pp**
and is the affordable step; 256 KiB costs 2.0pp and is not.

**How many chunks is that in practice?** A 6 MB CNN puts every tensor in one
chunk, which is misleading. A 7B transformer at fp16 does not:

```
q/k/v/o_proj      128 tensors × 32.0 MiB  ->  8 chunks each
mlp gate/up/down   96 tensors × 86.0 MiB  -> 22 chunks each
embedding           2 tensors × 250.0 MiB -> 63 chunks each
layernorms         64 tensors × ~0 MiB    ->  1 chunk each
                                              ~3,326 chunks total
```

A handful of large tensors dominate both bytes and chunk count.

### 4.3 Star topology, not a chain

Commits form a **star**, not a chain. Every residual diffs directly against
its group's nearest full ancestor; every `REBASE_INTERVAL`-th commit (currently
4) is stored full and becomes a new hub.

```
chain:  A(full) <- B-A <- C-B <- D-C      reconstructing D = 3 decodes
star:   A(full) <- B-A                    reconstructing D = 1 decode
        A       <----- C-A
        A       <---------- D-A
```

`N` bounds **drift from the hub**, not walk depth.

**Measured on all 25 checkpoints of the 92M benchmark model**
(`tools/experiments/topology_star_vs_chain.py`):

| | raw | star | chain | chain saves |
|---|---|---|---|---|
| features.0.weight | 90.70% | 70.50% | 68.30% | 2.20pp |
| features.31.weight | 84.45% | 79.18% | 77.44% | 1.73pp |
| **all tensors** | **84.39%** | **79.27%** | **77.56%** | **1.71pp** |

Reconstruction, worst commit in a group (`features.31.weight`, 55.1 MiB):

| | decodes | time |
|---|---|---|
| star | 1 | **57.7 ms** |
| chain | 3 | 165.3 ms (**2.87x**) |

**The chain is 1.71pp smaller; the star reconstructs 2.87x faster.** Residual
ratio is graded at 7% and mmap read throughput at 8%, so the trade favours the
star -- and more so for the FUSE path specifically, where a 128 KiB partial
read under a chain decodes three 4 MiB chunks instead of one.

> **CORRECTION.** This section previously claimed the star costs "~9% more
> storage" and reconstructs "2.19x faster". Both came from a 512x512 synthetic
> tensor measured before byte shuffle, the zigzag removal and the zstd level
> change. The real figures on real checkpoints are **1.71pp** and **2.87x** --
> the star's cost was overstated 5x and its benefit understated. The conclusion
> was right for the wrong numbers.

Deciding at commit time:

```
anchor      = nearest_full_ancestor(base_hash)      # follow first parents
since_full  = commits_since_full(head)
store_full  = anchor is None or since_full >= REBASE_INTERVAL - 1
```

plus the dynamic trigger: if the alignment pass reports that most tensors are
not alignable against the anchor, the anchor is a *different model* (a diverged
branch), so this commit is stored full and becomes the branch's own hub. See
`UNUSABLE_ANCHOR_FRACTION` in `cli/commands/commit.py`.

Everything above describes `--anchor-policy flat`, still the default. §4.3.2
covers the per-tensor alternative, which changes *when* hubs are created but
leaves the star-vs-chain reasoning here intact.

#### 4.3.1 Considered and rejected: bidirectional prediction (B-frames)

In video terms the star is I-frames and P-frames, and the missing third kind is
a **B-frame** -- predicted from a frame on each side rather than one behind.
Over a rebase interval with anchors at both ends, bisect: predict the middle
from the two anchors, then each quarter from its neighbours.

It is not the second-order delta rejected in §4.1: that extrapolated *forward*
(`x[k] - 2x[k-1] + x[k-2]`), compounding two steps of noise in one direction,
and cost +3.83pp. A **centred** estimate averages two independent errors
instead of stacking them, halving the variance rather than doubling it. Same
ingredients, opposite sign -- which is why it was worth measuring separately.

Measured over one interval (anchors at 21 and 25, epochs 22-24 stored):

| | star | chain | dyadic B |
|---|---|---|---|
| bf16 | 55.40% | 52.97% | **51.05%** |
| fp16 | 74.48% | 72.75% | **71.57%** |

It beats **both** existing layouts on both dtypes -- −4.35pp against the star
on bf16 -- at decode depth 2 (star is 1, chain is 3). Confirmed independently
against the shipping codec: 52.72% → 49.96% on bf16, 72.57% → 70.94% on fp16.

**It is not implemented, and the reason is workflow, not compression.**

A B-frame for epoch 22 is predicted from 21 and *23*. The natural workflow is
to commit as you train, and when epoch 22 finishes, 23 does not exist. Three
ways out, all with a real cost:

1. **One-epoch lag** -- hold the newest checkpoint, commit it once its
   successor exists. Two checkpoints on disk, which you have anyway, but the
   newest is uncommitted for one epoch and a crash loses it.
2. **Import a finished run** -- commit all of it at once in dyadic order.
   Clean, but it needs every checkpoint on disk simultaneously (4.4 GB for the
   92M benchmark), which is what the system exists to avoid.
3. **Repack afterwards** -- and the format forbids it. See below.

Against a −2.8pp gain: a user-visible flag whose correct use is non-obvious,
and a decode path that needs two parents per chunk, which slows the FUSE read
the PS weights at 25% with mmap throughput at 8%. A storage win that costs read
speed is the wrong trade here.

**The deeper lesson is the repack one.** `base_tensor_manifest` lives *inside*
the tensor manifest, which is itself content-addressed -- so changing a base
changes the manifest hash, the checkpoint-manifest hash, and the commit hash.
Repacking would rewrite history.

That is a design mistake worth naming. Git keeps content identity and storage
layout **separate**: an object's hash is its content, packing is orthogonal,
and `git gc` repacks without touching a single commit hash. Had commits
referenced tensors by `content_hash` with a separate index recording *how* each
is materialised, B-frames would be a background optimisation nobody has to
think about, and the star/chain/dyadic choice would be revisable after the fact
rather than baked in at commit time.

Reproduce with `tools/experiments/bframes.py`.

#### 4.3.2 Per-tensor adaptive anchoring (`--anchor-policy adaptive`)

The flat rule makes **one** decision for the whole checkpoint: full every
`REBASE_INTERVAL`-th commit, residual otherwise. That is wrong in both
directions at once. A frozen layer gets a fresh full copy it did not need; a
fast-drifting layer keeps diffing against an anchor that went stale two commits
ago and stores near-raw bytes every time.

`--anchor-policy adaptive` moves the decision to the tensor. **No format change
was required**: `base_tensor_manifest` is a field of the *tensor*-manifest, not
the commit, and `CommitCheckpoint._rows_from_manifest` already recursed through
it to arbitrary depth. The depth-1 star was a policy in `commit.py`, not a
constraint on disk, so existing repositories stay readable and no version bump
was needed.

Three pieces:

- `graph.tensor_anchor(store, commit, name)` -- follows one tensor's
  `base_tensor_manifest` chain to its root, returning `(manifest, depth)`.
  Derived entirely from objects already on disk.
- `anchor.decide(ratio, depth)` -- a pure function. **ANCHOR** if
  `ratio >= tau` (0.9, the same break-even reasoning as
  `residual.NOT_ALIGNABLE_THRESHOLD`: at 1.0 raw wins outright) or if
  `depth >= max_depth` (3, bounding FUSE read latency). **DELTA** otherwise.
- `commit.CompositeBase` -- a third implementation of the `names()`/`spec()`/
  `rows()` interface §5.5 describes, routing each tensor's reads to *its own*
  anchor rather than one shared hub.

Encoding runs twice: once against each tensor's current anchor (which is what
produces `ratio`), then a second pass re-encoding only the promoted tensors with
`base=None`. That second pass is close to free in the case it exists for --
`encode_chunk`'s `allow_raw_fallback` was *already* storing those chunks raw
every commit and then discarding the fact, so nothing promoted them and the next
commit repeated the work. `UNUSABLE_TENSOR_FRACTION` (0.5) folds a commit where
most tensors want re-anchoring back into a plain full commit, the same reasoning
as `UNUSABLE_ANCHOR_FRACTION` one level down.

**Measured on 24 sequential epochs of the 90M PlainCNN**
(`~/Downloads/raw_checkpoints_90m`, real training, layers freezing at different
epochs -- head and `features.7` at 15, `features.3` at 20, `features.28` at 23,
while `features.21`/`features.24` move to the end):

| policy | stored | ratio | full commits |
|---|---:|---:|---:|
| flat | 2,862,191,960 B | 64.50% | 6/24 |
| adaptive | 2,827,058,583 B | **63.71%** | 8/24 |

0.79pp smaller -- 33.5 MiB over the run -- while doing *more* full commits.
Reconstruction depth and the latency it costs:

| policy | max depth | depth 0 | depth 1 | depth 2 | `restore HEAD` |
|---|---:|---:|---:|---:|---:|
| flat | 1 | 676 | 1052 | 0 | 0.99 s |
| adaptive | 2 | 869 | 442 | 417 | 1.01 s |

`max_depth = 3` never bound. Adaptive leaves *more* tensors at depth 0 than flat
does (869 vs 676), because a promoted tensor resets its own chain -- the deeper
chains it creates are paid for by the tensors it takes off chains entirely. All
48 commits across both policies restore byte-identical.

**What the measurement actually says, which is not what the design predicted.**
The per-tensor mixed-anchor set does engage here -- depth 2 appears, 417
tensor-commits deep, which never happens on the small MLP fixtures. But it is
not where the saving comes from. On `mnist_pair` (a 2-hidden-layer MLP whose two
weight matrices carry 98.5% of the bytes and drift within 2-7% of each other at
every epoch) adaptive wins **2.3pp** on both seeds -- a *larger* margin than the
CNN with genuinely heterogeneous drift gets. What `decide` is really buying is
better *timing* of whole-checkpoint re-anchoring, ratio-triggered rather than
every 4th commit. The per-tensor granularity largely collapses into that through
`UNUSABLE_TENSOR_FRACTION`, because layers within a CNN stage drift together
even when stages differ.

0.79pp for a second encode pass is a thin margin. The flag stays opt-in; making
it the default wants a workload whose layers drift on genuinely independent
schedules -- a frozen-backbone fine-tune, which this series is not.

### 4.4 Reconstruction

To rebuild a tensor's rows `[start, stop)`:

```
for chunk in manifest.chunks:
    if chunk.row_end < start or chunk.row_start >= stop: continue     # skip
    payload = packs.read(chunk.object)
    if chunk.encoding == DELTA:
        base = rows_from(manifest.base_tensor_manifest,
                         chunk.row_start, chunk.row_end + 1)           # recurse
    piece = decode_chunk(chunk.encoding, payload, base, dtype)
slice the concatenation back down to [start, stop)
```

Only overlapping chunks are fetched — the property the FUSE read path depends
on. Under the flat star the recursion is depth 1; under `--anchor-policy
adaptive` it is bounded by `anchor.DEFAULT_MAX_DEPTH` (3, and measured to peak
at 2 — §4.3.2). The recursion itself was always general, which is why that
policy needed no reader change.

#### 4.4.1 Two mistakes that cost 13× between them

Both were found by mounting the filesystem and copying a file off it. Neither
appears in the test suite, because the fixtures are small enough that a tensor
has one chunk and neither condition can arise. A live mount is a test the unit
tests cannot replace.

**Key the chunk cache on the chunk.** It was keyed on `(commit, tensor, start_row,
end_row)` — the *request*. The kernel issues 128 KiB reads against 4 MiB chunks,
so every read produced a fresh key, missed, and decoded the whole chunk to
return 3% of it:

| | object fetches | bytes read | time |
|---|---|---|---|
| whole-file read | 369 | 277.7 MiB | 2.4 s |
| 128 KiB reads, request-keyed | 3,189 | **8,761.8 MiB** | 297.8 s |
| 128 KiB reads, chunk-keyed | 369 | 277.7 MiB | 1.5 s |

`CommitCheckpoint.chunk_spans()` exposes the boundaries and `_get_rows` snaps
requests outward to them. The general rule: **a cache key must describe the
unit of work performed, not the unit requested.**

**Do not `.copy()` a transposed array.** `unshuffle` was
`a.reshape(width, -1).T.copy().tobytes()`. `reshape` and `.T` are free views;
copying a *transposed* array is not — the output is contiguous while the input
has stride N, so numpy falls back to a generic strided iterator and moves one
byte at a time. 10.92 ms per 4 MiB chunk, **59% of the whole read path**, four
times the cost of the decompression it supports.

Assigning one plane at a time is the same permutation written so numpy can
vectorise it — `width` contiguous reads with strided writes instead of one
element-wise transpose — and returning the array rather than `bytes` removes a
second full copy the caller was about to undo with `np.frombuffer`:

| | per 4 MiB chunk | throughput |
|---|---|---|
| `.T.copy().tobytes()` | 10.92 ms | 366 MiB/s |
| plane assign → `bytes` | 2.07 ms | 1929 MiB/s |
| plane assign → `uint16` array | **1.77 ms** | **2263 MiB/s** |

A third, smaller one in the same path: `(b_bits + residual).astype(unsigned)`
allocated a whole extra buffer, because both operands are already that dtype
and `.astype` copies unconditionally. `copy=False` removes it — 14% of the
remaining time.

Together, on a 177 MiB checkpoint through a live mount:

| | wall | throughput |
|---|---|---|
| before | 11.14 s | 16 MiB/s |
| chunk-keyed cache | 2.41 s | 73 MiB/s |
| + fixed `unshuffle` | 1.23 s | 144 MiB/s |
| + `copy=False` | **0.86 s** | **206 MiB/s** |

For reference `restore`, which does not go through FUSE at all, takes 4.05 s
for the same commit — the mount is now the faster path, because reads overlap
with the copy.

#### 4.4.2 Is a C++ port worth it?

Not for speed. After the above, the read path splits roughly 65% real work
(zstd, numpy, file I/O — all already compiled) against 35% attributed to
Python frames, and that 35% is an *upper* bound on what a port could remove
because numpy operator dispatch inside those frames is counted as Python.
zstd decompression alone is 31%, and C++ cannot beat C at it.

The honest argument for C++ is **memory**. Each chunk currently allocates
compressed bytes, decompressed bytes, an unshuffled array, base rows and a
result. A preallocated per-thread arena gets that to two buffers and zero
allocation in the read path.

If it is attempted, do it bottom-up and stop when the numbers stop justifying
the next step: fix the numpy shapes (done, 2.8×), parallelise chunk decode
(threads suffice — `zstandard` releases the GIL), then a pybind11 module for
`decode_chunk` alone, then `CommitCheckpoint.rows` with mmap. A full libfuse3
daemon comes last and probably never: it would have to reimplement the
manifest walk, the object store, five encodings and the safetensors parser —
~1,000 lines of Python becoming ~2,500 of C++, with every format change then
made twice and kept bit-compatible.

To rebuild a whole **file** byte-exactly:

1. write the stored `header` object verbatim (it includes its length prefix)
2. parse it, sort tensors by `data_offsets[0]`
3. write each tensor's bytes in **that** order, streaming in row batches

Step 2 is not optional — see §7.2.

### 4.5 The verification trust chain

PS §2.c: *"trust is rooted at a locally accepted commit/ref ID."* So the ref is
the axiom and everything else is derived from it:

```
ref  (trusted by assumption)
 └─ commit hash             → re-hash the commit's bytes
     └─ checkpoint_manifest → re-hash
         ├─ header_object   → re-hash
         └─ tensor_manifest → re-hash
             └─ chunks[].object → decompress the payload, re-hash
```

**Every comparison must be against a hash that came from the object's parent.**
That is the entire design.

Under the packfile layout this was constantly under threat, because the pack
index and the pack trailer were *files an attacker rewrites alongside the
payload* — verifying against either proved only that both had been updated
together. Loose chunks remove that hazard at the root: **there is no
self-certifying structure left below the manifest.**

#### 4.5.1 The four checks

| # | check | reference value comes from |
|---|---|---|
| 1 | object bytes hash to the name they are stored under | the parent object that named it |
| 2 | every `chunks[].object` exists | the tensor-manifest |
| 3 | `blake3(chunk file)[:8] == chunks[].stored_checksum` | the tensor-manifest |
| 4 | `blake3(plain_stream(file)) == chunks[].object` | the tensor-manifest |

#### 4.5.2 Why the fast tier is now tamper-proof

`stored_checksum` used to live in the pack index. It now lives in the
tensor-manifest (§3.4.3), which is covered by the manifest hash → the
checkpoint-manifest → the commit → the ref. **An attacker who substitutes a
chunk file cannot adjust the checksum without changing the commit hash.**

That upgrades check 3 from a rot scan into a genuine tamper check, and it costs
no decompression:

| tier | checks | anchored to | detects | measured |
|---|---|---|---|---|
| `--shallow` | 1, 2 | ref | broken links, corrupt metadata | 0.05 s |
| `--fast` | 1, 2, 3 | **ref** | **rot *and* substitution** | 0.19 s, 694 MiB/s |
| default | 1, 2, 3, 4 | ref | the above, plus a forged 8-byte prefix (2⁶⁴ work) | 0.54 s, 239 MiB/s |
| `--content` | + tensor `content_hash` | ref | wrong-order permutation | 2.03 s, 64 MiB/s |

So full tamper detection is available at **~2.9× the throughput** it used to
require. Whether the default should stay at check 4 is now a genuine choice
rather than a forced one: check 3 alone is sound against an attacker who cannot
do 2⁶⁴ work, which is every attacker.

Check 4 is still worth running by default. It hashes the **decompressed
stream**, not reconstructed rows, so a residual chunk never needs its base —
verification stays O(stored bytes) with no recursion.

Two rules that do not change: **verification must never repair** what it
inspects, and the walk must follow **every parent**, not just the first, or a
merged branch goes unchecked.

### 4.6 Alignment

The problem: two networks can compute the identical function with hidden units
in a different order. Permute layer *k*'s output units and layer *k+1*'s input
weights the same way and nothing observable changes — but a byte diff sees
100% change.

**Permutation groups.** A group is one permutable axis, shared by every tensor
touching it. Permuting group *g* means:

- **rows** of every tensor the group produces (that layer's weight, bias,
  BatchNorm `weight`/`bias`/`running_mean`/`running_var`)
- **columns** of every tensor that consumes it (the next layer's weight)

Input and output groups are **pinned to identity** — you cannot renumber the
pixels or the classes.

**Column blocking.** After a conv→linear flatten, output channel *c* owns a
contiguous block of columns. One rule covers every case:

```
block = consumer.cols // producer.size

  linear → linear   cols = in_features        size = in_features  → 1
  conv   → conv     cols = in_ch·kh·kw        size = in_ch        → kh·kw
  conv   → linear   cols = in_ch·H·W          size = in_ch        → H·W
```

**The cost matrix** (Git Re-Basin Algorithm 1), for group *g*, target-major:

```
C = Σ over row members  W_target @ W_base.T
  + Σ over col members  W_target.T @ W_base        (blockwise)
```

Both terms are required. Rows alone tie whenever two units have identical
incoming weights; columns alone tie whenever two units are read identically
downstream.

**Coordinate descent.** Each group's cost depends on its neighbours', so there
is no order that solves each once with correct inputs. Instead: start all at
identity, solve one group at a time by linear assignment against whatever the
others currently claim, sweep until a full pass changes nothing.

Two properties fall out rather than being engineered:

- **Fine-tuning is the fast path.** Initialising `P ← I` means an unpermuted
  pair converges on sweep 1 — the ordinary termination condition, not a special
  case.
- **Monotonicity.** Solving a group maximises exactly the objective terms
  involving it, holding the rest fixed, so the global objective cannot decrease.

**Is it worth storing a delta at all?** Measure `‖target − aligned_base‖ /
‖target‖` — relative to the **target's own magnitude**, not to the
pre-alignment residual. Comparing post to pre would score every fine-tune at
exactly 1.00 (identity was already right, nothing improved) and flag the
easiest case in the system as not alignable. Against `‖target‖`: a fine-tune
scores ~0.01, two unrelated tensors score ~√2 ≈ 1.41. Threshold 0.9;
break-even is 1.0, where storing raw wins outright.

**Skippable:** `LayerWindow` (a two-layer sliding cache), objective tracking,
sweep shuffling, the dirty-set optimisation. The naive "re-solve every group
every sweep" is the same algorithm.

#### 4.6.1 Per-tensor normalisation of the cost

Each member's contribution is divided by `‖W_target‖ · ‖W_base‖` before it is
summed. Without that division the objective is decided by whichever member
holds the largest numbers, and on a BatchNorm group that is never the one
carrying the information.

A 1-D member — a bias, a norm scale, a running statistic — contributes
`outer(t, a)`, a **rank-1** matrix whose argmax is the same column for every
row. It says almost nothing about correspondence. A 2-D member contributes a
full-rank matrix that does. Measured on `g.features.28` of the 92M benchmark
between consecutive epochs, where identity is provably correct:

| member | shape | ‖contribution‖ | `argmax == i` |
|---|---|---|---|
| `features.28.weight` | (1792, 12096) | 5.75e+02 | **100.0%** |
| `features.29.running_var` | (1792, 1) | 2.17e+06 | 0.1% |

`running_var` holds variances, so its rank-1 term outweighed the convolution
kernel by 3783× and the solver optimised it instead: 668 of 1792 units moved
away from identity on a pair that had never been permuted. Downstream, every
one of those permutations had to be thrown away by the residual gate — after
the sweeps had already been paid for.

Normalising makes the members *comparable* rather than letting the biggest
win. The rank-1 terms are near-flat among units of similar magnitude, so the
full-rank term breaks the ties. Effects:

| | before | after |
|---|---|---|
| commit epoch01→02, aligned | 4 m 59.7 s | **23.9 s** |
| permuted-control recovery | 310.4 s (residual → 0.0000) | **19.2 s** (→ 0.0000) |
| fine-tune alignment | 424.4 s, 3 sweeps | **5.7 s, 1 sweep** |

The fine-tune fast path — "sweep 1 changed nothing, stop" — only began firing
on real data once this was fixed.

#### 4.6.1a Where the time actually goes

Neither the Hungarian solve nor the matmuls. Profiling one alignment of the
92M benchmark:

| | before | after |
|---|---|---|
| widening fp16/bf16 -> float32 | 7.69 s (51.6%) | 1.6 s |
| `group_cost` matmuls | 3.45 s (23.1%) | unchanged |
| `linear_sum_assignment` | 0.35 s (2.3%) | unchanged |
| **one-sweep alignment** | **15.2 s** | **6.0 s** |
| five-sweep (early epochs) | 58.3 s | 34.7 s |

Two fixes, and the second was larger than the first.

**`MatrixPair` was uncached.** One sweep performed **694 conversions of 144
distinct (tensor, side) pairs** — each group re-reads its members, a tensor in
two groups converts twice, and the norm and residual passes convert everything
again. It is now a byte-bounded LRU (`DEFAULT_CACHE_BYTES = 1 GiB`), the same
shape as `fuse/cache.py`. A cap rather than a switch: a 92M model caches
everything, a 7B model keeps what fits and degrades to re-reading the rest.

Alone this was only **1.36x**, which is why the profile was taken again.

**`relative_residual` widened to float64 before subtracting.** Two full float64
temporaries per call — 231 MiB each for the largest tensor — 144 calls per
alignment, **40% of the runtime** once the cache landed. The float64 exists to
protect the *accumulation*, which `norm`'s `einsum(..., dtype=np.float64)`
already does whatever the input dtype. Subtracting at float32 and letting
`norm` accumulate changes the answer by **2e-13** against a threshold of 0.9.

> Profile, fix the top item, profile again. The second bottleneck is invisible
> until the first is gone.

**A sweep is not free.** "Identity permutation detected — fast path" means
descent converged on sweep *1*; that sweep still built every cost matrix and
ran every solve. Late-epoch pairs converge in 1 sweep (~6 s); early ones take 5
(~35 s) and still land on identity, because early weights move enough to flip a
group before it settles back. An early-out comparing the solved objective
against identity's would collapse those — the largest remaining win here, and
unbuilt.

#### 4.6.2 GPU: worth it, but only after 4.6.1

Alignment wall-clock is graded at 8%. The inner loop is a matmul, so a GPU is
the obvious lever, and until 4.6.1 landed it was the wrong one: the assignment
step took 71% of a solve and has no GPU path. Two things changed that. Sweeps
dropped from 25 to 1, and each LAP got ~30× faster — Hungarian's runtime
depends on how *decisive* the cost matrix is, and a near-degenerate one is its
worst case. One full sweep over all groups of the 92M benchmark, measured:

| | cost build | LAP | total |
|---|---|---|---|
| numpy + CPU LAP | 8.73 s | 0.60 s | 9.33 s |
| torch/CUDA cost + CPU LAP | 1.84 s | 0.60 s | **2.43 s (3.84×)** |

Three details decide whether it is 3.8× or a disappointing 1.5×:

- **Ship fp16, widen on the device.** `as_matrix` widens to float32 on the CPU
  (0.32 s for one 1792×12096 pair) and then sends twice the bytes. Uploading
  the raw fp16 and calling `.float()` on device: 0.131 s → 0.074 s.
- **Do not use tensor cores.** An fp16 matmul measured *slower* here
  (0.280 s vs 0.074 s) and costs 4.5e-4 relative error. Widen to fp32.
- **Leave LAP on the CPU.** It is 25% of the GPU version and scipy has no
  device path. Auction or Sinkhorn would move it, but Sinkhorn returns a
  doubly-stochastic matrix rather than a permutation, and rounding it can
  break bijectivity that `is_permutation` then rejects.

Peak VRAM was 447 MiB, so the PS's budget is not a constraint here. torch must
stay an **optional** import with a numpy fallback — it costs ~474 MiB of RSS,
and nothing in the read path may pay that.

#### 4.6.3 How it scales with layer width

Everything above was measured on the 92M CNN benchmark, whose widest group is
**1792 units**. The PS evaluates to ~7B. The quantities that matter here scale
with the *width* of a layer, not with parameter count — the cost matrix is
`[n, n]` and the assignment solve is superlinear in `n` — so parameter count is
the wrong axis to extrapolate along.

**The two shapes land in completely different places.** A conv layer's
parameters are `out × in × kh × kw`, so a 3×3 kernel multiplies by 9 and you
reach 7B at roughly **4,000 channels**. A dense layer's are `h²` with no
multiplier, so a depth-24 MLP needs roughly **17,800 hidden units**. Same
parameter count, 4.5× the width. The risk was never "7B", it is "wide".

Measured with `tools/experiments/align_scaling.py` on synthetic MLPs with a
known ground-truth permutation, fp16, depth 3 (2 permutable groups), noise 0.01:

| width | params | align | sweeps | peak RSS | recovery |
|---|---|---|---|---|---|
| 512 | 0.36M | 0.05 s | 2 | 87 MB | 100% |
| 1,024 | 1.25M | 0.18 s | 2 | 117 MB | 100% |
| 2,048 | 4.6M | 0.98 s | 2 | 243 MB | 100% |
| 4,096 | 17.6M | 5.51 s | 2 | 658 MB | 100% |
| 8,192 | 68.7M | 31.8 s | 2 | 2.29 GB | 100% |
| **10,000** | **102M** | **50.8 s** | 2 | **3.36 GB** | **100%** |

Fitted on n ≥ 2048: **time ~ n^2.50**, RSS ~ n^1.66.

> **It is not n³.** Hungarian's runtime depends on how *decisive* the cost
> matrix is — §4.6.2 already noted that a near-degenerate matrix is its worst
> case. A cleanly recoverable permutation is its *best* case, and that is worth
> 0.5 in the exponent. Do not quote n³ for this system; it is the textbook
> bound, not the measured behaviour.

**Noise robustness**, at width 2048. Noise is relative:
`w += k · std(w) · N(0,1)`, so `k = 1.0` perturbs by as much as the weights
themselves.

| noise | recovery | residual after | sweeps | align |
|---|---|---|---|---|
| 0.0 – 0.3 | **100%** | 0.000 → 0.150 | 2 | ~1.0 s |
| 0.5 | **100%** | 0.234 | 3 | 1.9 s |
| 0.75 | 99.5% | 0.316 | 3 | 2.9 s |
| 1.0 | 95.3% | 0.371 | 4 | 4.4 s |
| 1.5 | 80.5% | 0.426 | 12 | 13.3 s |
| 2.0 | 62.0% | 0.455 | 15 | 18.8 s |

Perfect recovery holds until the perturbation reaches **half the weight
standard deviation**, then degrades smoothly — no cliff, and `not_alignable`
begins firing at 1.5, which is the residual gate correctly noticing it is in
trouble rather than returning a confident wrong answer.

**Width and noise multiply, they do not add.** This is the result to remember:

| | noise 0.01 | noise 1.0 |
|---|---|---|
| width 8192 | 31.8 s | **215.5 s** |
| sweeps | 2 | 5 |

6.8× the wall clock for 2.5× the sweeps — so the *per-sweep* cost rose too, for
the same reason the exponent is 2.5 rather than 3. A hard pair is expensive
twice over: more sweeps, and a slower solve inside each one.

The fine-tune fast path behaves as §4.6.1 claims — one sweep at every width,
and about 3× cheaper (10.7 s against 31.8 s at width 8192).

**What this means for a 7B fixture.** Extrapolating the fitted exponents from
n = 10,000:

```
                    align      peak RSS
n = 12,000          ~80 s        ~4.4 GB
n = 16,384         ~174 s        ~7.5 GB
n = 17,800         ~214 s        ~8.6 GB     <- the depth-24 7B MLP width
```

Those are for the *same* two-group model scaled in width only — they isolate
the width term, they are not a whole-model estimate.

Peak RSS is bounded by the *widest* layer, not the model, because groups are
solved one at a time — so a deep model does not multiply memory. Wall clock
does multiply by group count, so a depth-24 wide MLP is tens of minutes. A
CNN-shaped 7B model sits near n = 4,000 and costs seconds.

**Caveats, since these numbers will be quoted.** Weights are synthetic
Gaussians, not trained, and trained weights carry structure that could move the
cost matrix either way. Both checkpoints are passed as in-memory dicts, so peak
RSS includes them — a real commit passes `SafetensorsReader` and mmaps, and the
n² driver is the float32 widening, the `[n, n]` cost matrix and `residual.py`'s
temporaries rather than the file bytes. Depth 3 throughout, one seed per point.

---

## 5. Module guide

Twenty-odd files. For each: what it is, the core, and what you can leave out.

### 5.1 Foundations

**`errors.py`** — typed exceptions carrying `exit_code`. Imports nothing;
everything imports it (put it anywhere else and you get a cycle).

| class | exit |
|---|---|
| `SynapseError` | 1 |
| `UsageError` | 2 |
| `NoRepoError` | 3 |
| `IntegrityError` | 4 |
| `NotAlignableError` | 5 |
| `ConflictError` / `NetworkError` / `MountError` | 6 / 7 / 8 |

**Exit 4 is reserved for verification failures only.** Graders script against
it; never reuse it for I/O errors.

**`anchor.py`** — the per-tensor anchor policy (§4.3.2), one pure function of
`(ratio, depth)`. Deliberately free of any repository access: the commit-time
encode produces `ratio`, the on-disk manifest chain produces `depth`, and this
only has to combine them — which is why it is worth keeping out of both
`graph.py` (reads objects) and `codec/checkpoint.py` (does the encoding), and
why its thresholds can be swept in a test without building a repo.

**`store/atomic.py`** — the one durable-write primitive. The core is four
steps:

```python
fd = open(tmp_dir / f".tmp-{uuid4().hex}", "w+b")
fd.write(data); fd.flush(); os.fsync(fd.fileno()); fd.close()
os.rename(tmp, target)
fsync the parent directory
```

An observer sees either no file or the complete file. Needs a *streaming*
variant too (yield the open file), because the pack writer must seek back to
patch its record count and read back its own bytes for the trailer.

*Skippable:* the age-gated tmp GC. *Do not skip:* the uuid in the staging name,
or two concurrent commits collide.

**`store/objectstore.py`** — `objects/<ab>/<hash>`. Three methods:

```python
def put(data):  h = blake3(data).hexdigest()
                if not exists(path(h)): atomic_write(path(h), data)
                return h
def get(h):     return path(h).read_bytes()      # raise ObjectNotFoundError
def has(h):     return path(h).exists()          # a stat, NOT a re-hash
```

`put`'s existence check **is** the dedup mechanism. `has` must stay a stat —
`put` calls it on every object.

**`store/repo.py`** — `.synapse/` layout, `HEAD`, refs. `init_at`, `find`
(walk upward like git), `read_head` (§3.2's three states), `resolve_ref`,
`update_ref`, `set_head_branch`, `set_head_detached`, and the branch
list/create/delete/rename set.

`resolve_ref` order: `HEAD` → branch name → hex prefix (≥6). Branch-first, so a
branch named like a hash still resolves as a branch.

*Do not skip:* branch-name validation. It becomes a path component, so `..` and
`/` are a traversal hole.

### 5.2 `safetensors_io.py`

mmap reader. `TensorSpec(name, dtype, shape, width, num_rows, row_elems,
nbytes)` plus `header_bytes`, `names()`, `spec()`, `rows(name, start, stop)`,
`whole()`, `gather_rows()`.

`rows()` returns a **2-D unsigned view of raw bit patterns**, never floats.
There is no numpy dtype for bf16, so "return the real dtype" is not a contract
this can honour. Reinterpretation is the caller's job.

*Gotcha:* `mmap.close()` raises `BufferError` while any numpy view is alive.
Swallow it — the mapping frees when the last view dies.

### 5.3 `codec/chunk.py` and `codec/checkpoint.py`

`chunk.py` is §4.1 in code: `dtype_spec`, `zigzag`/`unzigzag`,
`shuffle`/`unshuffle`, `encode_chunk`/`decode_chunk`, `plain_stream`. (The
monotone key that used to sit in front of the delta step is gone -- it left the
encode path when byte shuffle replaced it, and its last consumer outside the
codec, `materialize._ulp_gap`, now compares raw bit patterns directly.)

`checkpoint.py` walks a checkpoint and produces manifests:

```python
for name in target.names():
    for row_start in range(0, num_rows, rows_per_chunk):
        target_rows = target.rows(name, row_start, row_stop)
        base_rows   = base.rows(name, row_start, row_stop) if base else None
        encoded     = encode_chunk(target_rows, base_rows, dtype=...)
        emit(record); chunk_entries.append({...})
```

*Do not skip:* the running `content_hash` digest (§2.1), and the
`base_tensor_manifest` iff-delta invariant (§3.4.3).

*Skippable:* the pending-buffer that holds identical chunks back until a
tensor's fate is known. It exists so an unchanged tensor emits no objects at
all; without it you write zero-delta chunks nothing references. Correct either
way.

`CheckpointResult.per_tensor` carries `(original, stored)` per tensor under the
same accounting rules as the aggregates (dedup excluded, a reused tensor
contributes `(original, 0)`). It is what §4.3.2's policy measures `ratio` from,
and it also makes `commit --json` able to say *which* layers are expensive
rather than folding everything into one number.

### 5.4 The chunk store

Replaces the whole `pack/` package. The entire lookup layer is:

```python
def path_for(h):  return objects / h[0:2] / h[2:4] / h[4:]
def has(h):       return path_for(h).exists()
def get(h):       return path_for(h).read_bytes()
def put(h, data): atomic_write(path_for(h), data, tmp_dir=tmp)
```

**The filesystem is the index.** Its own htree does the lookup, the kernel
maintains it, and it is already crash-safe. There is no fanout table, no
binary search, no `order` file, no index to lose or rebuild, and no
"verify must not repair the index" hazard.

#### What it costs, measured

Full cold read of a 129 MiB checkpoint, 810 chunks, `POSIX_FADV_DONTNEED`
between runs:

| | loose | pack | ratio |
|---|---|---|---|
| fetch, warm, 1 thread | 29.6 µs/chunk | 27.3 | 1.09× |
| fetch, **cold**, 1 thread | 302 µs/chunk | 105 | 2.87× |
| fetch, **cold**, 8 threads | 102 µs/chunk | 66 | 1.56× |
| **end-to-end cold** (fetch + zstd), 8 threads | 316 ms | 286 ms | **1.10×** |

The last row is the one that matters. zstd decode costs 233 ms of that and is
identical either way, so the storage layout moves ~10% of a full checkpoint
read. Warm, the difference is 9%.

Four things make cold loose reads slower, in rough order of size:

1. **No cross-file readahead.** The kernel prefetches aggressively *within* a
   file; it cannot know chunk #7 follows chunk #6 when they are separate files.
2. **Block amplification.** 653 of 810 chunks are under 4 KiB (median 394 B)
   and each costs a full 4 KiB block. In a pack they shared blocks.
3. **Path resolution** — four components, each a dentry lookup and, cold, a
   directory-block read.
4. **810 inodes instead of 1.**

#### Two mitigations, both worth building

**Fetch chunks in parallel.** NVMe wants queue depth; sequential loose reads
leave the device idle. Eight threads takes cold fetch from 302 to 102 µs/chunk
— a 3× win and the highest-value optimisation in this layer. Thirty-two threads
is *worse* than eight (135 µs, queue contention), and slurping a whole pack
sequentially is worse than parallel targeted reads (161 vs 66 µs/chunk), so
neither layout wants naive sequential access.

This is available to loose chunks specifically because **the manifest names the
exact file set in advance.** `posix_fadvise(WILLNEED)` over the whole set
before reading any of it gets 302 → 209 µs/chunk on its own.

**Cache decoded chunks in the FUSE daemon.** A 4 MiB chunk serves ~32 reads of
128 KiB, so fetch cost amortises toward zero and the gap all but disappears.
You need this cache regardless of layout — **and it must be keyed on the
chunk, not on the request**. See §4.4.1: keying it on the requested row range
looks equivalent and silently disables the cache entirely.

#### What it buys

- **`unlink` is a complete GC.** Reclaiming a chunk from a pack means rewriting
  the pack *and* rebuilding its index.
- **The fast verify tier becomes tamper-proof** (§4.5.2) — 694 MiB/s instead of
  239 for full substitution detection.
- **Interrupted sync resumes by existence check.** Each chunk is independently
  complete, which is exactly what PS 4b asks for.
- **Crash safety is simpler**: a partial pack must be discarded whole; a
  partially-written chunk cannot exist.
- **Dedup is a `stat`**, and roughly 950 lines of pack/index/packset code
  stop existing.

#### Where it stops working

Inode count. 20k chunks is comfortable; 1M chunks means 1M inodes plus ~500k
directories at two shard levels. If a checkpoint history ever gets there,
consolidate cold chunks into packs and keep the loose store as the write path —
which is what git does, and why `git gc` exists.

### 5.5 `graph.py`

The commit object graph. Writes the four object kinds; reads them back through
**`CommitCheckpoint`**, which is the key type:

> `CommitCheckpoint` implements `names()`, `spec()`, `rows()` — **the same
> three methods `SafetensorsFile` has.** Neither side knows which it holds.

That is what lets one code path serve "diff against a file" and "diff against
commit 4d8e2f", and it is the interface FUSE will read through. Keep the two
shapes identical.

Also: `walk_first_parent` (for `log`), `commits_since_full` /
`nearest_full_ancestor` (§4.3), `checkpoint_sizes` (derived, never stored —
baking a summary into a hashed object forks identity on a null repack).

For §4.3.2: `tensor_anchor(store, commit, name)` is `nearest_full_ancestor` at
*tensor-manifest* granularity — it follows `base_tensor_manifest` links rather
than commits, which is what lets one tensor's anchor differ from its siblings'.
`CommitCheckpoint.from_manifest(store, name, hash)` builds a single-tensor view
rooted directly at a manifest, bypassing the commit layer; every method reads
through `_tensor_manifests` and `store`, so only `header_bytes` and `total_size`
assume a real commit, and a per-tensor base source never calls either.

### 5.6 `materialize.py`

§4.4's file writer, plus `compare_sources` for accuracy inspection.
`max_ulp_diff` reuses the monotone key: the integer distance between two keys
**is** the ULP distance, which is the right measure for a codec claiming
bit-equality.

### 5.7 `verify.py`

§4.5. `verify_lineage(repo, roots, *, tier, ...)`. Takes **resolved commit
hashes**, not a ref — taking the root of trust as an argument is what PS §2.c
looks like in code.

Memoise verified objects and chunks by hash; measured 950 chunk references →
810 distinct.

### 5.8 `align/` — 13 files, self-contained

Imports nothing from the rest of the codebase; nothing imports it.

| file | role |
|---|---|
| `IR.py` | `TensorRef`, `LayerNode`, `PermutationGroup`, `Topology`, `validate()` |
| `axes.py` | the `block = consumer.cols // producer.size` rule |
| `config_parser.py` | checkpoint → `Topology`; shapes normative, `config.json` only corroborates |
| `group.py` | union-find over seed group ids; boundary groups found structurally and pinned |
| `objective.py` | the cost matrix |
| `lap.py` | SciPy assignment, `compose`, `invert`, `pack`/`unpack` |
| `coordinate_descent.py` | the sweep loop |
| `solver.py` | `align_checkpoints(...) -> AlignmentResult`; fans per-group answers to per-tensor |
| `residual.py` | `‖B−A‖/‖B‖`, threshold 0.9 |
| `reader.py` | its own mmap reader (fine to keep separate) |
| `report.py` | stderr rendering |

Two traps this code already handles:

- **Ordering.** safetensors sorts keys alphabetically, so `features.10` comes
  before `features.2` and `classifier.*` before `features.*`. Natural-sort
  fixes the first, not the second — so the layer chain is *type-checked*
  afterwards: each layer's column count must be a whole multiple of the
  previous layer's output size.
- **Norms attach by width, not position**, and BatchNorm's `running_mean` /
  `running_var` must join the host group. Leaving them behind is invisible
  under identity permutations and wrong under every real one.

`AlignmentResult.solved` distinguishes *"identity was the answer"* from *"a
group could not be solved and was left at identity."* These look identical in
output and must never be conflated.

`config_parser.infer_order` recovers the layer chain from **shape
divisibility** when the names do not sort topologically -- which is the normal
case, since safetensors sorts keys alphabetically and puts `head` before
`stem`. The layer whose column count is divisible by no other layer's output
size is the head of the chain; follow greedily from there.

### 5.9 `cli/`

`parser.py` registers commands (one module + one list entry each). `main.py` is
the **only** `sys.exit()` and the only `except SynapseError`. `output.py` is
the **only** stdout writer — the harness parses stdout, so a stray `print()` in
a command breaks it.

*Gotcha, if you use argparse:* with `parents=[global_parser]`, a subparser
parses into a *fresh* namespace and copies every attribute back, so a normal
default on `--json` silently clobbers a `--json` given before the subcommand.
Use `default=argparse.SUPPRESS` and apply baselines after parsing.
`set_defaults()` does not help — `parents=` shares Action objects rather than
copying them.

### 5.10 `networking/` — C++, standalone

`push`, `pull` and `serve` between two repositories over TCP. Built with
`make network`; there is no `__init__.py`, so setuptools does not treat it as
part of the Python package. It is C++ so a machine can host a repo without a
Python environment.

**Run it from inside `.synapse/`.** Every path it uses is relative
(`objects/...`, `refs/heads/...`), so the working directory *is* how it finds
the repo. From the repo root it silently finds nothing.

The protocol is git's negotiation. The side holding the branch walks the object
graph from the tip and computes the closure — commit -> checkpoint manifest ->
header, config and tensor manifests -> permutations and chunks, following
`parents` and `base_tensor_manifest` recursively — sends that hash list, the
peer replies with one byte per hash saying which it already has, and only the
missing objects go over the wire. A second push of a 25-epoch run transfers
almost nothing. Objects are received to `<path>.tmp` and renamed, so an
interrupted transfer leaves a complete object or none.

**It does not verify what it receives.** Bytes are written to the path named by
the hash the *sender claimed*, and never re-hashed. §4.5's trust chain rests on
every hash being checked against the value its parent named, and this layer
bypasses it. Hashes off the wire are validated as 64 lowercase hex characters —
a path-traversal guard, not an integrity check. **Run `synapsefs verify` after
a pull**; it is ref-anchored and runs at 544 MiB/s.

Five bugs found by running it against a real repo, all now fixed and worth
knowing because four are re-implementation drift:

| bug | consequence |
|---|---|
| `objects/<ab>/<62>` and a separate `objects/chunks/` subtree | every open failed; a pull transferred **nothing** |
| chunk loop bounded by `rho_len`, predicate `has_hash` | re-sent chunks; heap overflow if objects > chunks |
| `req_hash_objects.back()` on an empty vector | segfault whenever the peer sent nothing |
| `topology_config_hash` absent from the DAG walk | `config.json` never transferred; `log` worked, `verify` failed |
| `spp.cpp` negated the return value | success exited 1, failure exited 0 |

The paths were the packfile-era layout from the superseded `CHUNK_STORE.md`. A
Python implementation importing `ObjectStore.path_for` could not have had that
bug, or the config one — which is the argument for keeping the transport thin
and the format knowledge in one place.

**Was C++ worth it?** For speed, no: `send_file` reads a whole file into a
buffer then sends it, where `sendfile(2)` moves it in-kernel. Measured on
localhost, 1113 MiB/s buffered against 4638 MiB/s zero-copy — Python's stdlib
`socket.sendfile()` is 4.2x faster than what this does, in the slower language,
because the syscall pattern is what is being measured. Everything else here is
I/O-bound or kilobytes. The real argument is deployment: no Python needed on
the server.

---

## 6. Write ordering

Crash safety comes from ordering, not transactions. Two rules:

**Commit:** chunks → manifests → commit → **ref last**.

A manifest naming a chunk that is not yet on disk is a dangling reference. A
chunk nothing references yet is merely unreferenced, costs one file, and gets
deduped against next time. Only one of those is recoverable.

The loose layout makes this ordering *cheaper* to honour than it was with
packs: each chunk lands atomically and independently, so there is no
partially-written container to reason about and no index that could disagree
with what is on disk.

**Checkout:** materialise → **HEAD last**.

A decode failure must leave the repo untouched rather than pointing HEAD at a
commit whose file never landed.

Both reduce to: *the pointer moves last, and a crash before it leaves garbage
nobody can see.*

---

## 7. Traps

Things that look right, pass every test you would naturally write, and are
wrong.

### 7.1 Permutation direction and composition

`p[i]` is the base index for target index `i`; you gather `base[p]`; the read
path applies **the same gather, not the inverse**. Composing across a chain is
`p2[p1]`, not `p1[p2]`.

Both orderings are valid bijections of the correct length. Every chunk hashes
correctly. Shapes match. **Only the tensor `content_hash` catches it.** This is
the single most likely bug in the whole system and it will look like working
code.

### 7.2 Tensor write order

The safetensors header's JSON key order need not match its `data_offsets`
order. Writing tensors in `names()` order gives a file that parses, loads, and
is numerically identical — and is **not byte-identical**, failing the one
guarantee the project exists to provide. Always sort by `data_offsets[0]`.

### 7.3 Counting deduped bytes as stored

If a chunk already exists and you count its size as "stored", the residual
ratio is nonsense. This overstated storage 4× and reported 92.57% where the
truth was 23.14%. Keep `stored_bytes` and `deduped_bytes` separate.

### 7.4 Verifying a chunk against something the attacker also wrote

The general form of the trap, which the loose layout mostly removes: a check is
worthless if its reference value lives somewhere the attacker rewrites in the
same breath. Under packs that was the index checksum, the index trailer and the
pack trailer — all three could be recomputed, and all three would then agree.

With loose chunks the only reference values left are in the tensor-manifest,
which is hash-chained to the ref. Keep it that way: **never add a sidecar file,
xattr, or cache holding a chunk's expected hash.** The moment such a thing
exists, something will verify against it and the chain is broken again.

### 7.5 A tmp GC that deletes in-flight writes

If object-store construction sweeps `objects/tmp/`, it deletes the staging file
of a concurrent commit, which then fails at `rename`. Age-gate it or do not
sweep at all.

### 7.6 `safe_open` and bf16

It cannot read bf16. Both `get_slice` and `get_tensor` raise. No dtype
translation table fixes this; you must mmap.

### 7.7 `.half()` does not halve everything

BatchNorm's `num_batches_tracked` stays int64. Every real fp16 checkpoint is
mixed-dtype, so a codec that assumes one width per file breaks on the first
real model.

### 7.8 One variable, two meanings, one guard that destroyed the other

`codec/checkpoint.py` used `base_manifest_hash` for two unrelated things
depending on which branch a tensor took:

- tensor described here -> it is this manifest's `base_tensor_manifest`
  **pointer**, and must be null when no chunk actually references a base
  (§3.4.3's if-and-only-if rule);
- tensor **reused** (§4.5, unchanged since the base) -> it is the *identity of
  the manifest being reused*, and must be a real hash.

The guard enforcing the first rule ran unconditionally, so it nulled the second
one too, writing a literal `null` into the checkpoint-manifest's tensor map.

Reachable whenever a tensor is byte-identical to its base **and** every chunk
chose `raw` over a delta encoding -- which is not exotic. `encode_chunk` keeps
`is_identical` across its raw fallback deliberately (it describes the content,
not the storage), and on a small all-zero buffer raw genuinely wins: measured on
a real 448-element frozen bias, raw 19 bytes against escape 20. **One byte.**
Bias-free conv layers store an all-zero bias; the moment such a layer stops
training, the condition fires.

What made it dangerous rather than merely wrong: `commit` reported success and
exit 0, and only `restore`/`verify`/`log` -- later, separately -- crashed on the
null. On the 24-epoch 90M CNN, **6 of 24 commits were corrupted, including
HEAD**, with no error at write time. The MLP fixtures never caught it because
nothing in them ever freezes, so no tensor is ever both unchanged *and*
raw-encoded.

The general lesson is the one §7.3 and §7.4 also make: when a variable's meaning
depends on a branch taken later, a guard written for one meaning silently
corrupts the other. Regression tests pin both halves in
`tests/test_codec_checkpoint.py`.

---

## 8. How alignment is wired in

`commit` runs the aligner between resolving the base and encoding chunks:

```
resolve the base (the star's anchor)
  -> build TensorRefs from the base's specs
  -> config_parser.parse(refs, config)          # layer order inferred from shapes
  -> solver.align_checkpoints(base, target, topo)
  -> store each non-identity permutation as an object (lap.pack -> int32 LE)
  -> encode_checkpoint(..., alignment={name: TensorPermutation})
```

`CommitCheckpoint` grew three methods for this: `gather_rows` (a row
permutation turns a range request into a scattered one), `as_float` (the solver
scores whole layers and cannot work on raw bit patterns), and `refs`.

Measured on a permuted fixture: **residual ratio 87.96% -> 0.16%**, residual
norm 0.942 -> 0.287, every commit still byte-identical on reconstruction.

### 8.1 Two guards that are not optional

**Apply a permutation only when it reduced the residual.** The solver maximises
the weight-matching objective, which is *not* the same as minimising the delta.
Between consecutive epochs the right answer is identity, but early in training
another matching can score higher on the objective while making the residual
larger — measured 83.24% against 76.44% for identity. The gate is
`Assessment.helped` (post < pre), already computed by the residual pass.

**Disable §4.5 manifest reuse under a non-identity permutation.** The reuse rule
points an unchanged tensor at the *base's* tensor-manifest. Under a permutation
the chunks are identical **only after gathering the base through it**, so the
content is a reordering of the base's, not a copy. Reusing there yields a
repository that is entirely self-consistent: `verify --content` **passes**,
because the reused manifest carries the base's `content_hash` and
reconstruction faithfully reproduces the base's row order.

Only a byte comparison against the original file catches it. That is how it was
found, and why `tests/test_align_wiring.py` asserts on
`restore --compare --strict` rather than on internal consistency.

### 8.2 The direction, once more

`base_row_permutation[i]` is the base index that target index `i` came from.
Encode gathers `base[p]`; **decode applies the same gather, never the inverse.**
Both are valid bijections of the correct length, so no structural check sees a
reversal — only the tensor `content_hash` does.

---

## 9. Suggested build order

Each step is independently demonstrable — you can show it working before the
next one exists.

| # | build | demo |
|---|---|---|
| 1 | `errors`, `atomic`, `objectstore`, `repo` | `init`, refs round-trip |
| 2 | `safetensors_io` | read a real checkpoint, compare to `safe_open` where it works |
| 3 | `codec/chunk` | encode/decode round-trip, bit-exact |
| 4 | `codec/checkpoint` + the chunk store (§5.4) | `commit` a single file, see the residual ratio |
| 5 | `graph` + `CommitCheckpoint` | `commit` a second file against the first |
| 6 | `materialize` | **`checkout` gives back a byte-identical file** ← the guarantee |
| 7 | `verify` | corrupt a byte, watch it get caught |
| 8 | `log`, `branch`, `checkout`, `restore` | a real history |
| 9 | ~~wire `align/`~~ **done** | commit a permuted checkpoint, still byte-identical |
| 10 | FUSE | `safetensors.torch.load_file()` against the mount |
| 11 | `push`/`pull`/`serve` (§3.6, chunk at a time), `merge` | |

Step 6 is the milestone. Until `checkout` returns byte-identical bytes, nothing
above it is verifiable; after it, every later step has an oracle:

```
synapsefs restore <ref> --compare <original.safetensors> --strict
synapsefs verify --content
```

That pair is what will tell you step 9 is correct — nothing else can.

### 9.1 What to leave out

**Deleted outright by the loose-chunk move** — do not reimplement any of it:
the pack index (header, fanout, parallel arrays, binary search), index recovery
and rebuild-on-open, multi-pack probe order and the `order` file, packfiles in
*both* roles (storage and wire, §3.6), and `verify --packs`. That is
`pack.py` + `index.py` + `packset.py`, ~950 lines, gone.

**Still present in the reference implementation but not asked for by the PS:**
age-gated tmp GC, the identical-chunk pending buffer, the uncompressed `raw`
encoding, `LayerWindow`, alignment objective tracking and dirty-set pruning,
`--no-size`, and abbreviated-hash prefix resolution.

**Keep, even though they look like polish:** the write ordering (§6), the
iff-delta invariant (§3.4.3), tensor `content_hash`, `stored_checksum`
verification on read, and branch-name validation. Each is either a correctness
invariant or a security boundary.
