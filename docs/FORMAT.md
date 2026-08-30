# SynapseFS — On-Disk Format Specification

Status: **draft, frozen for Phase 1 implementation.** Anything marked `OPEN QUESTION`
is a real unresolved decision — do not silently pick a default; decide it explicitly,
then update this doc.

This document is the single source of truth for the object model. Track A (storage/CLI),
Track B (alignment/codec), and Track C (FUSE) all code against it. If an implementation
detail isn't decided here, it belongs here before it's decided in code.

> Supersedes `template.md`. Three things changed materially from that draft; each is
> called out inline with **CHANGED FROM DRAFT** so reviewers can find them:
> object hashes now cover *uncompressed* content (§2), chunks live in packfiles rather
> than one file per chunk (§4–5), and the `row_order` / `row_permutation` pair is
> replaced by an unambiguous target-order rule (§7).

---

## 1. Design principles

1. **Content-addressed, immutable, append-only.** Nothing on disk is ever modified in
   place. New content gets a new hash and a new path. This is what makes crash safety
   (§3) a two-line argument instead of a subsystem.
2. **Manifests are indexes, not payloads.** A manifest never contains tensor data —
   only hashes pointing at objects that do. Dedup, verification, and sync then work
   identically at every granularity (chunk → tensor → checkpoint → commit) with no
   per-level machinery. See §11.
3. **Byte-exactness comes from never re-deriving bytes we can store verbatim.** The
   `.safetensors` header is stored raw and replayed unmodified. Reconstruction is
   `header_bytes ‖ data_region`; nothing is re-serialized.
4. **Exact by construction, not by luck.** All delta math happens in an integer
   domain over raw bit patterns (§8). No floating-point operation occurs anywhere in
   the commit or reconstruction path.
5. **Identity is the fast path.** A real fine-tuning pair has an identity permutation.
   Every layer that can check for it — solver, manifest, reconstructor, FUSE read —
   must short-circuit on it rather than executing the general case with an identity
   argument.

---

## 2. Object addressing

- **Hash:** BLAKE3, 256-bit, lowercase hex (64 chars) when used in JSON; raw 32 bytes
  when used in binary structures.
- **Hash input: the object's UNCOMPRESSED, canonical content.**
  - `chunk` → the post-delta, pre-compression, pre-dictionary byte stream (§8 step 3).
  - all other kinds → the object's bytes exactly as they appear on disk.

> **CHANGED FROM DRAFT.** The draft hashed the *compressed* bytes of a chunk so that
> `verify` never had to decompress. That is unsafe here: once compressed output depends
> on anything besides the chunk's own content — a pack dictionary (§6), a zstd level, a
> library upgrade — the same logical chunk yields two different hashes and **dedup
> silently stops working**. Nothing errors; the residual ratio just quietly degrades.
> Hashing uncompressed content makes chunk identity independent of codec settings.
> The cost (verify must decompress) is recovered by the two-tier scheme in §12.

- **Directory layout:**

```
<SYNAPSE_DIR>/
  objects/
    tmp/<random>                 # write staging; never a final resting place
    <hash[0:2]>/<hash>           # loose objects: manifests, commits, headers, permutations
    pack/
      pack-<hash>.pack           # sealed, immutable chunk container (§5)
      pack-<hash>.idx            # its index (§6)
  refs/
    heads/<branch-name>          # text file containing a commit hash
  HEAD                           # "ref: refs/heads/<branch>" or a raw commit hash (detached)
```

- `<hash>` in a pack filename is the BLAKE3 of the `.pack` file's own bytes, so packs
  are content-addressed like everything else.

### 2.1 Which objects go loose, which go in packs

| Kind | Storage | Why |
|---|---|---|
| `chunk` | **pack** | Thousands per commit; needs read locality and shared compression |
| `header` | loose | One per checkpoint, small |
| `permutation` | loose | One per permutation group, dedups perfectly when identity-stable |
| `tensor-manifest` | loose | Tens–hundreds per commit, ~KB each |
| `checkpoint-manifest` | loose | One per commit |
| `commit` | loose | One per commit |

Rationale for the split: DAG traversal (`log`, `verify --shallow`, ref resolution) must
never depend on pack integrity. Keeping structural objects loose means the commit graph
is walkable even if a pack is missing or damaged, and it keeps §3's crash argument
trivially true for everything except bulk data.

---

## 3. Write protocol (crash safety)

Every **loose object** write:

1. Write full bytes to `objects/tmp/<random>` on the **same filesystem** as `objects/`.
2. `fsync(fd)`.
3. `rename()` to `objects/<h[0:2]>/<h>` — atomic on the same filesystem.
4. `fsync()` the containing directory so the rename itself survives a crash.

Every **pack** write (see §5 — packs are built whole and sealed, never appended to):

1. Build `<random>.pack` and `<random>.idx` in `objects/tmp/`.
2. `fsync()` both files.
3. `rename()` the `.pack` into `objects/pack/pack-<hash>.pack`, then the `.idx`.
   **Pack before index, always** — the index must never be visible while referencing
   bytes that aren't durable.
4. `fsync()` `objects/pack/`.

Refs follow the identical temp→fsync→rename→fsync-dir pattern, and **refs are always
the last thing updated** in any operation (commit, merge, pull).

**Invariants this buys:**

- An interrupted operation can leave orphaned objects or an orphaned pack. Harmless —
  nothing references them.
- A ref can never point at a partially-written or inconsistent commit.
- An index can never reference non-durable pack bytes.

**On startup:** delete everything in `objects/tmp/` (nothing durable ever references a
tmp path). Delete any `.pack` in `objects/pack/` with no matching `.idx`. If a ref file
is malformed or missing, **refuse to proceed** rather than guessing.

> Because packs are sealed and never appended to, the draft's crash-safety argument
> survives the packfile change unmodified. This is the reason for the one-pack-per-write
> rule in §5 — appending to a live pack would introduce a window where a crash between
> "index updated" and "pack durable" is genuine corruption, in exactly the scenario the
> PS grades ("recovery behavior after a simulated crash mid-write").

---

## 4. Object kinds

### 4.1 `chunk`

The row-range slice of one tensor, encoded per §8. Stored inside a pack; addressed by
the BLAKE3 of its uncompressed content (§2). Opaque to the storage layer — the
`encoding` field in the referencing tensor-manifest says how to decode it.

### 4.2 `header`

The raw, verbatim `.safetensors` header region for one checkpoint file:

```
[8 bytes, little-endian u64: header_len N]
[N bytes: the exact JSON header, including __metadata__, exact key order, exact whitespace]
```

Stored byte-for-byte from the source. **Never re-serialized.** Replay, don't rebuild.

### 4.3 `permutation`

A raw little-endian `int32` array, no header, length implied by object size / 4.
Content-addressed like any object, so an identity permutation shared across commits
stores exactly once — and two groups that happen to share a permutation share an object.

### 4.4 `tensor-manifest`

One per tensor (`layer1.weight` and `layer1.bias` are separate manifests). Full schema
in §7.

### 4.5 `checkpoint-manifest`

```json
{
  "header_object": "<blake3>",
  "tensors": {
    "layer1.weight": "<tensor-manifest-hash>",
    "layer1.bias":   "<tensor-manifest-hash>"
  },
  "topology_config_hash": "<blake3 of the config.json this checkpoint was aligned against>"
}
```

**Reuse rule:** when committing a new version, compute a candidate tensor-manifest for
every tensor. If a candidate's hash equals the parent's hash for that tensor name, do
not write a new object — reuse the existing hash. (Storage-layer dedup; the *compute*
skip is the identity fast path in §9.)

### 4.6 `commit`

```json
{
  "checkpoint_manifest": "<hash>",
  "parents": ["<hash>", "..."],
  "timestamp": "2026-08-25T12:00:00Z",
  "message": "commit message",
  "full": true,
  "checkpoint_name": "epoch03.safetensors"
}
```

`parents`: empty for root, one for a normal commit, two or more for a merge.

`full`: whether this commit stored its checkpoint outright rather than as a
residual — i.e. whether it is a star hub (§12A). It is derived data living
inside an immutable object, which is normally the wrong place for it; the
alternative is worse. Deciding whether the re-basing interval is due otherwise
means loading every tensor-manifest of every ancestor to check that all of
their `base_tensor_manifest` fields are null — an O(tensors × depth) walk to
answer a one-bit question, on the hot path of every single commit.

`checkpoint_name`: the **basename** of the file that was committed. `checkout`
with no `--out` has to restore the checkpoint into the working tree "under the
filename recorded in the commit" (CLI.md §4), and the safetensors header names
tensors, not files, so there is nowhere else to put it. Only the basename is
stored, and readers must re-strip it: a commit that carried
`../../.ssh/authorized_keys` here would otherwise be a write-anywhere
primitive against anyone who checked it out.

Commits written before either field existed are still readable: `full` defaults
to false (the walk falls back to `parents == []` to find a root) and
`checkpoint_name` falls back to `model.safetensors`.

---

## 5. Packfile layout

A pack is an immutable container for chunk objects. Two roles, one format:

- **Storage pack** — one per commit, holding every chunk that commit introduced.
- **Transfer pack** — built on demand during `push`/`pull` from a want-list, holding
  exactly the chunks the peer lacks.

The transfer role is why pack construction must be callable at any time, not only at
commit. It is also what satisfies the graded "only missing blocks are transferred"
requirement: sending whole storage packs would re-send chunks the peer already has
from another branch.

### 5.1 Byte layout

```
offset  size  field
------  ----  --------------------------------------------------------------
0       8     magic       "SYNPACK\0"
8       4     version     u32 LE, currently 1
12      4     flags       u32 LE   bit0 = dictionary present
16      4     count       u32 LE   number of chunk records
20      32    dict_hash   blake3 of the `dictionary` object, or 32 zero bytes
52      ...   records     count × record, back to back
EOF-32  32    trailer     blake3 of bytes [0, EOF-32)
```

Each **record**:

```
size  field
----  --------------------------------------------------------------
32    content_hash   blake3 of this chunk's UNCOMPRESSED content
4     stored_len     u32 LE, byte length of the payload that follows
4     plain_len      u32 LE, byte length after decompression
N     payload        stored_len bytes: the encoded chunk (§8)
```

The 40-byte record header is redundant with the index by design: it makes a pack
**self-describing**, so a lost or corrupt `.idx` can be rebuilt by a linear scan from
offset 52. Say this in the Q&A — it is the answer to "what if your index is damaged."
At 1 MB chunks the overhead is 0.004%.

### 5.2 Index-to-pack offset convention

Index offsets point at the **payload**, not the record header. The header for a chunk at
offset `o` occupies `[o-40, o)`. This makes the FUSE hot path a single `pread(fd, len, o)`
with no arithmetic. Recovery scans, which need the headers, start at 52 and walk
`40 + stored_len` at a time.

---

## 6. Pack index layout

Designed to be `mmap`ed and binary-searched in place. **Never load it into a Python
dict** — see §6.3.

### 6.1 Byte layout

```
offset          size      field
------          ----      --------------------------------------------------
0               8         magic     "SYNIDX\0\0"
8               4         version   u32 LE, currently 1
12              4         count     u32 LE = N
16              32        pack_hash blake3 of the .pack this indexes
48              1024      fanout    256 × u32 LE, cumulative
1072            N×32      hashes    raw 32-byte content hashes, sorted ascending
1072+32N        N×8       offsets   u64 LE, payload offset into .pack
1072+40N        N×4       stored_len u32 LE
1072+44N        N×4       plain_len  u32 LE
1072+48N        N×8       checksum   first 8 bytes of blake3(stored payload bytes)
EOF-32          32        trailer    blake3 of bytes [0, EOF-32)
```

Entry cost: 56 bytes. 280,000 chunks ≈ 15.7 MB, mmapped, near-zero resident.

`fanout[b]` = the number of entries whose first hash byte is `<= b`. Cumulative, so
`fanout[255] == count`.

### 6.2 Lookup

```
b  = hash[0]
lo = 0 if b == 0 else fanout[b-1]
hi = fanout[b]
i  = binary_search(hashes[lo:hi], hash)     # ~8 comparisons typical
-> offsets[i], stored_len[i], plain_len[i], checksum[i]
```

Parallel arrays (rather than interleaved records) are deliberate: the search touches
only the hash array, so a probe pulls in ~8 cache lines instead of ~56 bytes × log N
scattered across the file.

### 6.3 Multi-pack lookup

On repo open, enumerate `objects/pack/*.idx` and mmap each. A lookup probes each index
**newest-pack-first** (recent commits are the hot path). Maintain the order in
`objects/pack/order` (newline-separated pack hashes, newest first), rewritten with the
same temp→fsync→rename dance.

> **Implementation trap — do not skip this.** A Python `dict` keyed by 64-char hex
> strings costs ~50–80 MB resident at 280k entries, against a graded 7% peak-RSS
> metric. Keep hashes as raw 32-byte slices of the mmap and binary-search with
> `memoryview` comparisons. No per-entry Python objects.

`OPEN QUESTION` — probe cost grows linearly in pack count. Decide a repack trigger
(e.g. consolidate when pack count > 32) or add a per-pack bloom filter. Needed only if
demo repos exceed ~30 commits.

---

## 7. Tensor-manifest schema

```json
{
  "name": "layer1.weight",
  "dtype": "bf16",
  "shape": [4096, 4096],
  "base_tensor_manifest": "<hash>" | null,
  "base_row_permutation": "<permutation-object-hash>" | null,
  "base_col_permutation": "<permutation-object-hash>" | null,
  "col_block_size": 1,
  "chunks": [
    {"row_start": 0,    "row_end": 511,  "encoding": "delta-shuffle-zstd", "object": "<blake3>"},
    {"row_start": 512,  "row_end": 1023, "encoding": "raw-zstd",          "object": "<blake3>"},
    {"row_start": 1024, "row_end": 1535, "encoding": "delta-shuffle-zstd", "object": "<blake3>"}
  ]
}
```

### 7.1 Row order — the ambiguity, resolved

> **CHANGED FROM DRAFT.** The draft carried `row_order: "stored"` plus a
> `row_permutation` array, and warned at length about the "very quiet, very nasty bug"
> where a component assumes `row_start`/`row_end` are logical when they are physical.
> That ambiguity is now **deleted rather than documented**, by a single rule:

**Rows are always stored in the tensor's own logical order** — the row order of the
target `.safetensors` file this manifest describes. `row_start`/`row_end` are logical
row indices, inclusive. There is no stored-vs-logical translation step anywhere.

This is sound because the residual is defined in the *target's* index space:

```
R[i] = key(B[i]) - key(A[π(i)])          # i indexes B's rows, always
```

There is never a reason to store B's rows in A's order, so the two orders can be
defined to coincide by construction. The permutation is not a property of *this*
tensor's layout; it is a property of the *relationship* to the base — which is why it
now lives next to `base_tensor_manifest` and is named for it.

### 7.2 Field semantics

- **`row_start` / `row_end`** — inclusive logical row indices. Not byte offsets, not
  indices into any shared file. A reader given a byte range in the reconstructed
  logical tensor computes which chunk entries overlap it, and fetches only those.

- **`base_tensor_manifest`** — the tensor-manifest this one was diffed against, or
  `null` if stored in full (first commit for this tensor, or the not-alignable
  fallback of §9). Reconstruction requires first reconstructing the base.

- **`base_row_permutation`** — hash of a `permutation` object π where `π[i]` is the
  **base** row index that target row `i` was diffed against. `null` means identity —
  the common case for a fine-tuning pair, and a mandatory short-circuit at every read
  site. Only meaningful when `base_tensor_manifest` is non-null.

- **`base_col_permutation`** — hash of the permutation applied to the base's *column*
  axis before differencing. This is the previous layer's output permutation. Storing
  it here rather than as a group-ID reference into a separate topology document is
  deliberate: the PS requires the diff artifact be **self-sufficient** ("given it plus
  one of the two checkpoints, you must be able to reconstruct the other"). A group ID
  pointing into a document that isn't part of the artifact would not satisfy that.

- **`col_block_size`** — the column permutation acts on contiguous blocks of this many
  columns. This is how the conv→linear flatten gotcha is handled explicitly:

  | Tensor | Flattened shape | `col_block_size` |
  |---|---|---|
  | Linear `[out, in]` | `[out, in]` | `1` |
  | Conv `[out_ch, in_ch, kh, kw]` | `[out_ch, in_ch·kh·kw]` | `kh · kw` |
  | Linear after a conv flatten | `[out, in_ch·H·W]` | `H · W` |

  So permuting input channel `c` to position `c'` moves columns
  `[c·block, (c+1)·block)` to `[c'·block, (c'+1)·block)` as a unit. Getting this wrong
  produces a diff that reconstructs correctly *only* when the permutation is identity —
  which means your fine-tuning fixtures pass and your permuted fixtures fail. Test it
  against a permuted CNN fixture specifically.

- **`encoding`** — per chunk, not per tensor. Different chunks of one tensor may differ.

  | Value | Meaning |
  |---|---|
  | `raw` | Uncompressed tensor bytes. Used for base/root checkpoints so the π gather at read time is a page-cache `memcpy` rather than N decompressions. |
  | `raw-zstd` | zstd of tensor bytes, no delta. Used when `base_tensor_manifest` is null, or per-chunk when delta doesn't help. |
  | `delta-shuffle-zstd` | The residual path — §8. |
  | `raw-shuffle-zstd` | Stored in full, byte-shuffled — §8. |
  | `delta-zigzag-zstd`, `raw-zstd` | Legacy. Still decode; never written. |

  `OPEN QUESTION` — whether root checkpoints use `raw` or `raw-zstd` by default. `raw`
  makes the scattered-permutation gather cheap; `raw-zstd` saves disk (a real concern
  given 24 GB free, see PLAN §0). Benchmark both on a permuted fixture with a cold page
  cache before deciding, and record the numbers here.

- **Chunk axis** — chunks always partition the **row / output-channel** axis (dim 0
  after any conv→2D flatten). Chunk boundaries must fall on whole rows; byte-granular
  chunking destroys the row↔chunk correspondence and is forbidden.

- **Chunk size** — `OPEN QUESTION`. Target ~1–4 MB post-compression: large enough that
  zstd framing overhead doesn't eat the residual-ratio metric, small enough that a FUSE
  read doesn't decode much more than it needs. Tensors below one chunk's worth are a
  single chunk. Record the benchmarked default here.

---

## 8. Delta encoding (`delta-shuffle-zstd`)

> **REVISED 2026-08-28.** Two measured changes to this section's pipeline. A
> **byte shuffle** was added before compression (78% -> 72%), and the **zigzag**
> step was then removed, because shuffle and zigzag turned out to be
> substitutes and zigzag cost 0.6-0.8pp once shuffle was in place. The default
> zstd level dropped from 3 to **1**, which is both smaller and ~2x faster on
> this data — compression is non-monotone in level here. The **monotone key**
> was then removed too (worth +0.07pp at gap 1, −0.26/−0.47/−0.66pp at gaps
> 2/3/4), so the residual is now a plain modular subtraction of raw bit
> patterns and the `FLOAT`/`SINT` key-kind distinction has left the storage
> path. Full measurements, plus four rejected alternatives and the one
> principle that explains all of them, are in `ARCHITECTURE.md` §4.1. The old
> encodings still decode.


Given a chunk of target tensor `B` and the corresponding rows of base `A` (gathered
through `base_row_permutation`, and column-permuted through `base_col_permutation` /
`col_block_size`):

**1. Bit-pattern → monotone integer key.** Per element, on raw bit patterns, never on
decoded float values. The map depends on how the dtype lays out its sign, not on how
wide it is, so width and *key kind* are chosen separately. For an `n`-bit element, with
`MSB = 1 << (n-1)`:

| Key kind | dtypes | `key(x)` |
|---|---|---|
| float (sign-magnitude) | `F16` `BF16` `F32` `F64` `F8_*` | `bits ^ MSB` if sign clear; `~bits` if sign set |
| sint (two's complement) | `I8` `I16` `I32` `I64` | `bits ^ MSB` |
| uint (already ordered) | `U8` `U16` `U32` `U64` `BOOL` | `bits` |

Order-preserving over the value domain, so numerically small changes produce small
integer deltas. All three are xor with a mask, which is why the implementation is
branchless rather than three cases.

> Three correctness notes worth knowing before someone "fixes" this:
> - **NaN and Inf need no special case.** They are bit patterns like any other and
>   round-trip exactly, because nothing in this path interprets them as numbers.
> - **`-0.0` and `+0.0` must stay distinct.** `bits(-0.0) = 0x8000 → key 0x7FFF`;
>   `bits(+0.0) = 0x0000 → key 0x8000`. Any "normalization" of signed zero breaks
>   byte-exactness. Do not add one.
> - **The key kind is not inferable from the array.** numpy has no `bfloat16`, so a
>   BF16 chunk necessarily arrives as `uint16` and is indistinguishable from a real
>   U16 chunk by inspection. The safetensors dtype name must be passed in explicitly;
>   keying a BF16 tensor as `uint` still round-trips and silently destroys the ratio.

**2. Delta.** `delta = key(B) - key(A)`, element-wise, **at the native element width**,
wrapping mod `2**n`. Do not widen.

Wrapping loses nothing: `(a - b) + b == a` mod `2**n` for every pair, wraparound
included, so reconstruction stays exact. When `|true delta| < 2**(n-1)` — the
overwhelmingly common case, since aligned checkpoints differ slightly — the wrapped
value *is* the true delta. When it wraps, it aliases to the distance the short way
around, which is never larger than a widened delta would have been.

**3. Zigzag.** `zz(d) = (d << 1) ^ (d >> (n-1))`, also at native width — over the full
`n`-bit signed range this is a bijection onto the full `n`-bit unsigned range, so it
needs no headroom either.
**The output of this step is what the chunk's content hash covers** (§2).

> **No varint, no bitpacking.** An earlier draft of this section widened 16-bit keys to
> `int32` and then proposed varint or bitpack to win the doubled stream back. Steps 2–3
> at native width make both unnecessary: the residual stream is *exactly* the size of
> the tensor chunk it encodes, with no per-element work, and zstd in step 4 takes it
> from there. `residual_ratio` is therefore measured against a stream that never
> inflates. Reintroducing widening silently doubles `plain_len`;
> `test_stream_never_inflates` exists to catch that.

**4. Compress.** zstd, optionally with the pack's dictionary (§6). This produces the
record payload.

**Reconstruction** reverses each step exactly: decompress → un-shuffle
(`d = (zz >> 1) ^ -(zz & 1)`) → `key(B) = delta + key(A)` (native width, wrapping) →
invert `key()` → reinterpret as the dtype. Every step is integer arithmetic on raw bit
patterns; no float rounding occurs anywhere, which is what makes byte-exact
reconstruction unconditional rather than "usually exact."

`raw` and `raw-zstd` chunks skip steps 1–3 entirely. A chunk that *was* delta-encoded
but compressed worse than raw is stored `raw-zstd` instead (§7); note this also changes
its content hash to the hash of its raw content, which is the desirable outcome — a raw
chunk dedups against every identical raw chunk in the repo, whereas a residual only
ever matches a residual taken against the same base.

Implemented in `synapsefs/codec/chunk.py`; every claim above is pinned by a test in
`tests/test_codec_chunk.py`, including exhaustive round-trips over all 8- and 16-bit
patterns for all three key kinds.

### 8.1 Pack dictionary

`OPEN QUESTION`, default off until benchmarked. A zstd dictionary trained per pack
(`zstandard.train_dictionary(16384, samples)`) recovers cross-chunk redundancy that
independent per-chunk frames throw away, **without** the seekability loss of
compressing the pack as one stream. Measured 5% on synthetic data; expect more on real
correlated residuals and on the many small tensors. Store the dictionary as a loose
object and reference it by `dict_hash` in the pack header.

This is only safe because chunk hashes cover uncompressed content (§2) — with the
draft's compressed-bytes hashing, a per-pack dictionary would fork the hash of every
chunk and break dedup.

---

## 9. Not-alignable fallback

If the alignment engine's relative-residual-norm check (post-alignment vs.
pre-alignment) shows alignment did not materially reduce the difference, the tensor is
stored with `base_tensor_manifest: null` and all chunks `raw-zstd` — as if it were the
first commit for that tensor.

This **must be reported explicitly by the CLI**, not silently emitted as a low-quality
diff. The PS calls this out directly ("Correctly recognize when two checkpoints are not
meaningfully alignable and report that explicitly, rather than forcing a low-confidence
match and displaying it as a result").

`OPEN QUESTION` — the threshold. Needs a number backed by the fixture set, including
the deliberately non-alignable fixture. Record it here with the measurement.

---

## 10. Reconstruction path (normative)

To read logical byte range `[a, b)` of tensor `T` in checkpoint `C`:

1. `C` → `checkpoint-manifest` → `tensors[T]` → tensor-manifest `M`.
2. Convert `[a, b)` to a logical row range using `shape` and `dtype`.
3. Select the chunk entries in `M.chunks` overlapping that row range.
4. For each: index-lookup the chunk hash → `(pack, offset, stored_len, plain_len)` →
   `pread` → decompress.
5. If `M.base_tensor_manifest` is null, the result is the tensor bytes. Done.
6. Otherwise, recursively reconstruct **only the required rows** of the base:
   - rows needed = `π[i]` for each target row `i` in range (identity if
     `base_row_permutation` is null — take the contiguous fast path).
   - apply `base_col_permutation` with `col_block_size` if non-null.
7. Apply §8 reconstruction element-wise.

Steps 6–7 recurse down the delta chain. Chain depth is the cost of this design;
mitigate with periodic re-basing (store a full `raw`/`raw-zstd` checkpoint every N
commits). `OPEN QUESTION` — the re-basing interval N.

**Never materialize a whole tensor to satisfy a partial read**, and never write a
reconstructed checkpoint to disk on mount. The PS forbids the latter explicitly and it
is directly benchmarked from a cold page cache.

---

## 11. Why this yields dedup, fast verification, and resumable sync uniformly

Every level (chunk, tensor-manifest, checkpoint-manifest, commit) is a content-addressed
object referencing children by hash, so the same three properties hold at every
granularity with no separate mechanism:

- **Dedup** — identical content at any level hashes to one object and is stored once.
  Chunk-level dedup is index-mediated (§6) but semantically identical to loose-object
  dedup: before writing a chunk into a new pack, probe the global index; if present,
  just reference it.
- **Verification** — one recursive hash-check-and-descend from a commit, touching only
  residual blocks rather than a materialized multi-GB model. This is *why* verification
  stays fast at scale, and it is the argument to make for the graded metric.
- **Resumable sync** — the have/want protocol operates on object hashes uniformly. A
  partial transfer means some objects exist and some don't; resuming re-runs the same
  negotiation and naturally skips what's present.

---

## 12. Verification tiers

> **CHANGED FROM DRAFT (twice).** The draft claimed `verify` never needs to decompress;
> that depended on hashing compressed bytes, which §2 abandons. The capability was
> recovered by splitting verify into tiers. The *second* change is that the content
> tier is now the **default** rather than an opt-in `--deep` — see §12B for why the
> earlier default was indefensible.

| Tier | Does | Catches | Measured (25 commits, 152 MiB) |
|---|---|---|---|
| `--shallow` | Walks loose objects only: commit → checkpoint-manifest → tensor-manifests, re-hashing each; probes that every chunk reference exists. | Structural corruption, broken DAG links | 0.05 s |
| `--fast` | + for every referenced chunk: index lookup, `pread`, compare the 8-byte stored-payload checksum. **No decompression.** | Bit-rot, truncation, a damaged pack | 0.19 s, 694 MiB/s |
| **default** (`--deep`) | + decompress every chunk and check its full 32-byte content hash against `chunks[].object`. | **Malicious block injection** | 0.54 s, 239 MiB/s |
| `--content` | + reconstruct every tensor and check its manifest's `content_hash`. | A permutation applied in the wrong order | 2.03 s, 64 MiB/s |

Deep costs 2.8× the checksum tier and still finishes a 25-commit history in half a
second, which is what settled the open question the draft left here.

`--content` is deliberately *not* folded into the default. It is the only check that
requires **reconstruction** — a residual chunk must be applied to its base, so cost
scales with the reconstructed model, not with stored bytes. Deep hashes the
*decompressed stream* instead (the exact bytes `_finish` hashed), which needs no base
and no recursion. That is why deep stays O(stored bytes) and `--content` does not.

---

## 12B. Why the content tier must be the default

The tiers are usually described as increasing amounts of work. That framing is wrong
and it hid a real bug in the earlier default. What actually changes between tiers is
**whose hash you are trusting.**

PS §2.c fixes the root of trust: *"Trust is rooted at a locally accepted commit/ref
ID."* Everything else must be derived from it by walking down:

```
ref  (trusted by assumption)
 └─ commit hash             → re-hash the commit's bytes
     └─ checkpoint_manifest → re-hash
         ├─ header_object   → re-hash
         └─ tensor_manifest → re-hash
             └─ chunks[].object → decompress the payload, re-hash
```

Against that chain:

- **`--shallow`** is *parent-anchored* — sound, but covers only metadata.
- **`--fast`** is *index-anchored*. The index is a file an attacker rewrites alongside
  the payload. Self-certifying, therefore a rot scan and nothing more.
- **default** is parent-anchored again, now all the way down to bytes.

So the content tier is not "fast plus extra." It is **shallow's trust chain extended to
completion**; the checksum tier hangs off a different anchor entirely and is a detour,
not a step along the way.

The earlier framing — that the default tier merely "cannot detect a crafted payload
carrying a matching stored checksum" — undersold the gap badly. It implies the weakness
is an 8-byte collision, i.e. 2⁶⁴ work, i.e. not a real attack. The actual weakness is
that **every** checksum below the manifest level is attacker-writable:

1. Rewrite a chunk payload → recompute the index entry's 8-byte checksum
2. → recompute the index trailer
3. → recompute the pack trailer

All three now agree. `tests/test_verify.py` does exactly this and asserts that
`verify --fast --packs` reports **clean** on the tampered repo, while the default tier
names the substituted block, its pack, and the tensor and row range that reference it.
That pair of tests is the demonstration; the prose above is only its explanation.

Two consequences worth stating plainly:

- **`--packs` is not a security control.** Re-hashing a pack against its own trailer
  proves internal consistency, which the attacker ensured. It is off by default because
  at the content tier every referenced byte already has a stronger check, so it only
  covers framing and unreferenced regions at double the read volume.
- **Verification never repairs.** `PackSet` rebuilds a missing or unreadable index on
  open by default. `verify` disables that, because repairing the artifact under
  inspection and then reporting OK would make the command worthless.

---

## 12A. Commit topology: a star, not a chain

**Decision: every residual commit diffs directly against its group's full
checkpoint, and every 4th commit becomes a new full checkpoint.**

```
commit   1     2     3     4     5     6     7     8
stored   FULL  diff  diff  diff  FULL  diff  diff  diff
base     --    1     1     1     --    5     5     5
```

Contrast the chain this replaced, where commit 4's base was commit 3, whose base
was commit 2. Both bound the damage with the same `N`, but they bound different
things:

| | chain | star |
|---|---|---|
| decodes to reconstruct the Nth commit | N | **2**, always |
| what `N` limits | reconstruction depth | how far a group drifts from its hub |
| residual size | smallest possible | grows with distance from the hub |

### Why the star wins

**Re-measured 2026-08-29 on all 25 checkpoints of the 92M benchmark model**
(`tools/experiments/topology_star_vs_chain.py`). The table that stood here
before was from a 512x512 *synthetic* tensor, measured before byte shuffle, the
zigzag removal and the zstd level change, and it was wrong by 5x.

| | raw | star | chain | chain saves |
|---|---|---|---|---|
| features.0.weight (stem) | 90.70% | 70.50% | 68.30% | 2.20pp |
| features.31.weight (deepest) | 84.45% | 79.18% | 77.44% | 1.73pp |
| **all tensors** | **84.39%** | **79.27%** | **77.56%** | **1.71pp** |

Reconstruction, worst commit in a group (`features.31.weight`, 55.1 MiB):

| topology | decodes | time |
|---|---|---|
| star | 1 | **57.7 ms** |
| chain | 3 | 165.3 ms (**2.87x**) |

> **Superseded figures.** This section previously claimed the star costs "~9%
> more storage" and reconstructs "2.19x faster". The real numbers are **1.71pp**
> and **2.87x** -- the star's cost was overstated 5x and its benefit
> understated. The conclusion was right; the evidence was not.

Two things make the trade worth taking:

1. **The graded weights point this way.** `residual_ratio` is 7%.
   Reconstruction speed feeds `mmap read throughput` (8%), `POSIX compliance`
   (10%) and `daemon peak RSS` (7%) -- and chain depth multiplies RSS too,
   since each level holds a decoded array live while decoding the next.
2. **Partial reads are the real workload.** The 2.87x above is whole-tensor
   reconstruction. A 128 KiB FUSE read under a chain decodes a 4 MiB chunk at
   *every* level; under a star it decodes two. The gap there is wider.

### Consequences to be aware of

- **`N` is still needed.** It now bounds the star's radius rather than a chain's
  depth: without it, a long run drifts arbitrarily far from its hub and the
  residuals grow without limit. The reset is visible in practice -- residual
  ratio climbing 18.8% -> 20.6% -> 21.5% across a group, then dropping back at
  the next anchor.
- **The hub is load-bearing.** Every commit in a group depends on it, so losing
  a hub's pack costs the whole group rather than a suffix. Content addressing
  and `verify` detect that; they do not repair it.
- **A full commit is structurally identical to a root commit** -- every tensor
  `base_tensor_manifest: null`, all chunks `raw`/`raw-zstd` -- so no separate
  code path exists to write or read one.
- **A full commit costs far less than a whole checkpoint.** Its chunks are still
  content-addressed and deduped against every existing pack (section 6.3), and
  unchanged tensors reuse their manifests outright (section 4.5), so the cost is
  bounded by what actually changed since the last hub.

`N = 4` is a placeholder chosen for a bounded worst case, not a measured optimum.
It lives in exactly one place, `graph.REBASE_INTERVAL`.

**Planned successor: make the interval dynamic.** Rather than a fixed count,
start a new hub when the group's residuals stop being cheap -- the same signal
section 9's not-alignable check already computes. A commit whose residual ratio
jumps, or whose tensors increasingly fall back to `raw-zstd` per section 7, is
one where the group has drifted far enough that a fresh hub is cheaper than a
larger residual. That turns re-basing from a schedule into a response to the
data, and reuses a measurement the alignment stage has to make anyway.

`OPEN QUESTION` -- the dynamic trigger's exact statistic and threshold. Blocked
on the same measurements as section 9's not-alignable threshold; until then the
fixed `N = 4` stands.

---

## 13. Open questions to resolve before Phase 1 is "done"

- [ ] Default chunk size / rows-per-chunk (needs the codec benchmark, PLAN §1.2).
- [ ] Root-checkpoint encoding: `raw` vs `raw-zstd` (gather cost vs. 24 GB free disk).
- [ ] Pack dictionary on/off, and dictionary size, measured on real fixtures.
- [x] Commit topology and re-basing interval — **star, fixed N = 4** (§12A).
- [ ] Dynamic re-basing trigger to replace the fixed N (§12A), keyed on
      residual-ratio degradation / `raw-zstd` fallback rate.
- [ ] Not-alignable threshold, measured against the non-alignable fixture.
- [ ] Repack trigger / bloom filters if pack count exceeds ~32.
- [ ] Whether `verify --deep` becomes the default.
- [ ] Topology-IR schema (separate doc) — must be frozen before Track B's
      permutation-group resolver is written. Note it no longer needs to be referenced
      at *reconstruction* time (§7.2), only at *commit* time.
