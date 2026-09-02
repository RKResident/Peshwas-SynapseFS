---
title: synapsefs-storage-lineage.md

---

# SynapseFS — Storage & Lineage Architecture

**Revision 2.** Supersedes the previous revision in full. Appendix A lists every
substantive change and the failure each one prevents; read it first if you have
implemented against the previous revision.

---

## 1. Scope

This document specifies the storage engine and version-control lineage model: how
checkpoints are addressed, linked, reconstructed, branched, merged, verified, and
recovered after a crash.

It does **not** specify the alignment engine's internals (see the Alignment
Architecture doc) or the FUSE read path (see the Filesystem Access Layer doc), except
where those subsystems consume or produce objects defined here. Where this document and
`CLI.md` / `FileFormat.md` overlap, the byte layouts in `FileFormat.md` are normative
and reproduced here for convenience; the semantics below are normative.

### 1.1 The guarantee this design exists to provide

> Reconstruction of any committed checkpoint is **byte-identical** to the file that was
> committed, and remains so across branch switches, merges, network transfer, and
> process crashes.

Everything below — content addressing, verbatim header storage, write ordering,
determinism requirements — exists to hold that sentence true. Any proposed change that
weakens it is out of scope regardless of what it improves.

---

## 2. Conventions

### 2.1 Hashing

All object addressing uses **BLAKE3-256** (32-byte digest).

| Context | Representation |
|---|---|
| Binary structures (pack records, index entries, permutation files) | 32 raw bytes |
| JSON fields (commits, manifests) | 64 lowercase hex characters |
| Ref files, `HEAD` | 64 lowercase hex characters |
| Loose-object **paths** | split: 2 hex chars of directory, 62 hex chars of filename |

The loose-object path split is the one place a hash is not written as a contiguous
64-character string. `objects/<h[0:2]>/<h[2:64]>` — a 2-character shard directory and a
62-character filename that together spell the full 64-character digest. Sharding keeps
directory entry counts manageable on filesystems that degrade with very wide
directories. Code that reconstructs a hash from a path must concatenate both
components; code that computes a path from a hash must split it. Never store the full
64-character name inside the shard directory, and never shard on a different width.

### 2.2 JSON canonicalisation

Every JSON object that is content-addressed (`commit`, `checkpoint-manifest`,
`tensor-manifest`) is serialised canonically before hashing:

```
UTF-8 · separators=(',', ':') · sort_keys=True · no trailing newline
```

This is mandatory. Without a single canonical serialisation, semantically identical
content hashes differently, deduplication silently stops working, and the hash
comparisons in §8 and §11 produce false differences.

### 2.3 Integers and ranges

All multi-byte integers in binary structures are **little-endian**.

Row ranges in a `tensor-manifest` are **inclusive on both ends**: `row_start` and
`row_end` are both valid row indices. A tensor with `shape[0] == 4096` is covered by
chunks whose ranges union to `[0, 4095]`, with `chunks[0].row_start == 0` and
`chunks[-1].row_end == 4095`. There is no half-open form anywhere in the format.

### 2.4 Logical 2-D shape

Several rules below refer to a tensor's **logical 2-D shape**. Every tensor, whatever
its stored rank, has one:

```
rows = shape[0]  if shape else 1
cols = prod(shape[1:])  if len(shape) > 1 else 1
```

| Stored shape | Logical 2-D |
|---|---|
| `[out, in]` (linear) | `[out, in]` |
| `[out_ch, in_ch, kh, kw]` (conv2d) | `[out_ch, in_ch·kh·kw]` |
| `[n]` (bias, norm) | `[n, 1]` |
| `[]` (scalar) | `[1, 1]` |

`cols` is **`prod(shape[1:])`, not `shape[1]`**. For a conv weight `shape[1]` is
`in_channels`, which is not a column count. Every column-axis rule in this document is
stated in terms of `cols`.

---

## 3. Repository layout

```
<SYNAPSE_DIR>/
├── objects/
│   ├── tmp/<random>                 write staging; unconditionally cleared on startup
│   ├── <hh>/<remaining-62-hex>      loose objects, sharded by the first hash byte
│   └── pack/
│       ├── pack-<hash>.pack         sealed, immutable chunk container
│       ├── pack-<hash>.idx          mmap-able index for its pack (derived)
│       ├── pack-<hash>.bloom        membership filter for its pack (derived)
│       └── order                    newline-separated pack hashes, newest first
├── refs/
│   └── heads/<branch>               64 hex chars + "\n"
├── cache/                           entirely derived; safe to delete at any time
│   ├── generations                  commit hash -> generation number
│   └── verified/<branch>            verification high-water marks (§11.4)
└── HEAD                             "ref: refs/heads/<branch>\n"  or  64 hex + "\n"
```

`refs/heads/<branch>` is exactly 65 bytes (64 hex + newline). Detached `HEAD` is the
same 65 bytes; attached `HEAD` is `"ref: refs/heads/" + branch + "\n"`.

Everything under `cache/`, plus every `.idx` and `.bloom`, is **derived state**: it can
be deleted and rebuilt from the authoritative data without loss. Nothing in `cache/`
participates in crash durability, and no correctness property may depend on it being
present or current.

---

## 4. Object model

Three mutable files. Seven immutable object kinds. Every immutable object is
content-addressed: its filename (loose) or record hash (packed) is the BLAKE3-256
digest of its own canonical bytes.

### 4.1 Mutable files

| Name | Path | Contents | Update rule |
|---|---|---|---|
| Ref | `refs/heads/<branch>` | tip commit hash | atomic (§9.1) |
| HEAD | `HEAD` | `ref: …` or a bare hash | atomic (§9.1) |
| Pack order | `objects/pack/order` | pack hashes, newest first | atomic (§9.1) |

`order` is mutable and is updated whenever a pack is sealed. It was previously
described as if the repository had only two mutable files; it does not. `order` is
rebuildable — a missing or corrupt `order` is regenerated by listing `objects/pack/*.pack`
and sorting by mtime descending — but because probe order affects only latency and not
which object is found, a stale `order` is never a correctness problem.

### 4.2 Immutable object kinds

| Kind | Storage | Payload | Addressed by |
|---|---|---|---|
| `commit` | loose | canonical JSON | BLAKE3 of its bytes |
| `checkpoint-manifest` | loose | canonical JSON | BLAKE3 of its bytes |
| `tensor-manifest` | loose | canonical JSON | BLAKE3 of its bytes |
| `header` | loose | raw `.safetensors` header bytes | BLAKE3 of those bytes |
| `permutation` | loose | packed `int32` LE array | BLAKE3 of those bytes |
| `dictionary` | loose | zstd dictionary blob | BLAKE3 of the blob |
| `chunk` | **packed** | encoded tensor-row payload | BLAKE3 of the **pre-compression** stream |

`chunk` is the only kind stored inside sealed packs rather than as a loose file. Its
address is the hash of the payload **before** compression (§12.3) — this is what makes
deduplication work across different compression settings and dictionary states.

#### 4.2.1 `commit`

```json
{
  "checkpoint_manifest": "5e6f…12",
  "parents": ["0a1b…99"],
  "timestamp": "2026-08-25T12:00:00Z",
  "message": "epoch 3 checkpoint"
}
```

| Field | Type | Meaning |
|---|---|---|
| `checkpoint_manifest` | hash | the `checkpoint-manifest` this commit records |
| `parents` | array of hash | `[]` root · `[h]` ordinary · `[h1, h2, …]` merge |
| `timestamp` | string | RFC 3339, UTC, `Z` suffix |
| `message` | string | free text |

`parents` is always an array with no upper bound. There is no separate merge-commit
object kind — a commit is a merge purely by `len(parents) >= 2`. All traversal code is
written against the N-ary case; there is no special-cased "1 or 2 parents" branch
anywhere.

Generation numbers (§7.3) are **not** part of this object. They are derived, cached in
`cache/generations`, and excluded from the hash — a derived value inside the identity
of an immutable object would make the object's address depend on when it was computed.

#### 4.2.2 `checkpoint-manifest`

```json
{
  "header_object": "1a2b…cd",
  "tensors": {
    "layer1.bias":   "44de…01",
    "layer1.weight": "9f2c…a1"
  },
  "topology_config_hash": "77aa…bb"
}
```

| Field | Type | Meaning |
|---|---|---|
| `header_object` | hash | the `header` object holding this checkpoint's verbatim header bytes |
| `tensors` | object | every tensor name → its `tensor-manifest` hash |
| `topology_config_hash` | hash | BLAKE3 of the raw `config.json` bytes describing this checkpoint's graph |

The map is **flat**, matching `.safetensors`' own flat key space. There is no recursive
tree object: `.safetensors` has no subdirectories, so the generality would be unused.
The manifest is still a Merkle node — its hash depends on every tensor-manifest hash it
contains — so tamper-evidence is unaffected.

**Invariant:** `tensors` must name exactly the tensor set described by `header_object`,
no more and no fewer. A checkpoint-manifest whose key set differs from its header's key
set is rejected at write time; reconstruction (§6.4) walks the header's key order and
would otherwise encounter a name it cannot resolve, or silently omit one it can.

#### 4.2.3 `header`

Raw bytes copied verbatim from the source file's `[0, 8 + header_len)` range: the
8-byte little-endian length prefix followed by the space-padded header JSON.

**This object is never rebuilt from a parsed dictionary** for any checkpoint that
originated as a real file. Re-serialising parsed JSON does not reproduce the original
padding, key ordering, or `__metadata__` placement, and byte-for-byte reconstruction
fails. Store it and replay it.

**One exception, narrowly scoped (§8.5):** a *merge commit* may describe a tensor set
that no source file ever contained, so no stored header describes it. In that case only,
a header is synthesised, and the synthesis rule is fixed so it is reproducible (§8.5).
This does not weaken the guarantee in §1.1, because a merge commit's checkpoint never
existed as a file — there is no original for it to be byte-identical to. Every commit
that *did* originate from a file replays that file's header verbatim.

#### 4.2.4 `tensor-manifest`

```json
{
  "name": "layer1.weight",
  "dtype": "F16",
  "shape": [4096, 4096],
  "content_hash": "6d40…7e",
  "base_tensor_manifest": "9f2c…a1",
  "base_row_permutation": "3b71…0e",
  "base_col_permutation": null,
  "col_block_size": 1,
  "chunks": [
    {"row_start": 0,    "row_end": 2047, "encoding": "delta-zigzag-zstd", "object": "aa01…ff"},
    {"row_start": 2048, "row_end": 4095, "encoding": "raw-zstd",          "object": "bb02…ee"}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | exactly the tensor's key in the source header |
| `dtype` | string | per the `.safetensors` dtype table (`FileFormat.md` §1.4) |
| `shape` | array of int | row-major, exactly as in the source header |
| `content_hash` | hash | BLAKE3 of this tensor's **fully reconstructed bytes**, in the target checkpoint's own row order (§4.2.4.2) |
| `base_tensor_manifest` | hash or `null` | the manifest this one is diffed against; `null` ⇒ stored in full |
| `base_row_permutation` | hash or `null` | `permutation` aligning rows to the base; `null` ⇒ identity |
| `base_col_permutation` | hash or `null` | `permutation` aligning column blocks to the base; `null` ⇒ identity |
| `col_block_size` | int ≥ 1 | contiguous columns moving as one unit (§4.2.4.1) |
| `chunks` | array | ordered, non-overlapping, exhaustive cover of `[0, shape[0]-1]` |

##### 4.2.4.1 `col_block_size`

`col_block_size` has exactly three valid derivations, determined by the tensor's role in
the topology graph and **never chosen freely** by the alignment or storage layer:

| Tensor role | Logical 2-D shape | `col_block_size` |
|---|---|---|
| Linear, `[out, in]` | `[out, in]` | `1` — each input feature is its own column |
| Conv2d, `[out_ch, in_ch, kh, kw]` | `[out_ch, in_ch·kh·kw]` | `kh · kw` — a channel's whole spatial kernel moves as one block |
| Linear immediately following a conv flatten, `[out, in_ch·H·W]` | `[out, in_ch·H·W]` | `H · W` — a channel's whole flattened feature map moves as one block |

The third row is **not** a variant of the second: it applies to a *linear* layer, and it
is the case implementers omit. A codec that normalises an unrecognised `col_block_size`
to `1` passes every fixture where alignment is a no-op (fine-tuning without neuron
reordering) and produces a wrong reconstruction the moment a real permutation exists —
the aligner moves a channel's worth of columns while the codec moves one column.

`col_block_size` is a property of the tensor's **axis geometry**, not of the
permutation. It is emitted at its true value even when `base_col_permutation` is `null`
— a pinned conv column axis still reports `kh·kw`. Readers ignore it when the
permutation is `null`; they must not require it to be `1`.

##### 4.2.4.2 `content_hash`

BLAKE3 of the tensor's reconstructed bytes as they appear in *this* checkpoint's data
region — after all deltas, permutations, and decoding have been applied. It is
independent of `base_tensor_manifest`, of both permutations, of `col_block_size`, of
chunk boundaries, and of encoding.

It exists because **the manifest hash is not a content identity.** Two manifests with
the same hash certainly hold the same content; the converse is false, and merge (§8.2)
depends on the converse. Two branches can hold a byte-identical tensor whose manifests
differ because they were aligned against different bases: `main` diffs epoch-5 against
epoch-4, `experiment` diffs the same epoch-5 against epoch-3 — different base pointer,
different permutations, different chunk hashes, identical weights. Comparing manifest
hashes reports a conflict on a tensor nobody diverged on.

`content_hash` costs one hash of already-materialised bytes at commit time and doubles
as an end-to-end reconstruction check for `verify --deep` (§11.3) and for the
`checkout`/`mount` consistency test.

##### 4.2.4.3 Invariants — checked, not merely documented

Let `cols = prod(shape[1:])` per §2.4. Then:

1. `base_row_permutation`, when non-null, has exactly `shape[0]` elements.
2. `base_col_permutation`, when non-null, has exactly `cols / col_block_size` elements,
   and `cols % col_block_size == 0`.
3. Both permutations, when non-null, are **bijections** of `range(0, len)` — every index
   appears exactly once.
4. `col_block_size >= 1`.
5. `chunks` are ordered, non-overlapping, and exhaustive over `[0, shape[0]-1]`.
6. `base_tensor_manifest` is `null` **if and only if** every chunk's encoding is a
   non-delta encoding (`raw` or `raw-zstd`). A delta encoding with no base is
   unresolvable; a base with no delta chunk anywhere is a pointer to nothing.

Checks 1–4 run **whenever a `permutation` object is loaded, before it is applied to any
chunk decode.** This is the cheapest place to catch a `col_block_size` mismatch: it
fails immediately and deterministically, rather than producing a plausible-looking but
numerically wrong tensor that only a downstream consistency check or a human would
notice. A non-integer result in check 2 means `col_block_size` was computed from the
wrong axis or the wrong layer type; the manifest is rejected at write time, never
silently truncated or padded.

Note what these checks **cannot** catch: a permutation that is valid, correctly sized,
and simply *wrong* — for instance one composed in the wrong order across a base chain
(§6.3). Both orderings are bijections of the correct length. `content_hash` is the only
mechanism that catches that class of error.

#### 4.2.5 `permutation`

A packed array of `int32`, little-endian, no header of any kind. Element count is
`filesize / 4`.

```
p[0] : int32 LE   p[1] : int32 LE   p[2] : int32 LE   …
```

Example — `p = [2, 0, 3, 1]`, 16 bytes:

```
02 00 00 00  00 00 00 00  03 00 00 00  01 00 00 00
```

Semantics are defined entirely by the referencing field:

| Referencing field | Meaning of `p[i]` |
|---|---|
| `base_row_permutation` | the **base** tensor's row index that **target** row `i` was diffed against |
| `base_col_permutation` | the **base** tensor's column-block index that **target** column-block `i` was diffed against |

The direction is stated once, here, and every consumer follows it: reconstruction
gathers `base[p]` to bring the base into target order. No component in the system
inverts a permutation on the main path. Identity is represented by a `null` pointer,
never by a stored `[0, 1, 2, …]` array — an explicit identity array costs a loose
object nobody needs and forces every reader down the gather path.

Because permutations are content-addressed, one group's permutation is stored **once**
no matter how many tensors reference it. A hidden layer's permutation is simultaneously
the `base_row_permutation` of its own weight and bias and the `base_col_permutation` of
the next layer's weight; all three fields point at the same object.

#### 4.2.6 `dictionary`

An opaque zstd dictionary blob, referenced by a pack's `dict_hash` when that pack's
`flags` bit 0 is set. Chunks compressed against a dictionary must be decompressed with
that same dictionary. The dictionary is content-addressed like any other object, so a
corrupted dictionary is detected the same way a corrupted chunk is — and it is a GC
root through any pack that names it (§13).

#### 4.2.7 `chunk`

The terminal payload leaf: encoded bytes for one tensor's row range, under one of the
encodings in §12.3. A chunk exists only as a record inside a sealed pack, addressed by
the BLAKE3 of its **pre-compression** stream.

**Chunks split on whole rows, never on byte offsets.** Row↔chunk correspondence is what
makes partial reconstruction (§6) and permutation mapping possible at all; splitting
mid-row breaks permuted fixtures while leaving fine-tune fixtures passing.

---

## 5. The dual-axis DAG

Every commit sits at the intersection of two graphs.

**Data axis** — one commit's spatial state, resolved top-down. Note the two edges the
previous revision's diagram omitted: the base chain and the permutation objects.

```
[ commit ]
    │ checkpoint_manifest
    ▼
[ checkpoint-manifest ] ──header_object──► [ header ]
    │ tensors: name -> tensor-manifest hash
    ▼
[ tensor-manifest ] ──base_row_permutation──► [ permutation ]
    │      │        ──base_col_permutation──► [ permutation ]
    │      │
    │      └──base_tensor_manifest──► [ tensor-manifest ]   ← recursive; see 6
    │                                        │
    │                                        └──► … until base is null
    │ chunks[]
    ├── rows 0–2047    ──► [ chunk ]  resolved via pack index
    └── rows 2048–4095 ──► [ chunk ]
```

**Time axis** — history, resolved via `parents`:

```
[ C1 ] ◄── [ C2 ] ◄── [ C4 ]        C4.parents = [C2, C3]
       ◄── [ C3 ] ◄──┘
```

A branch ref points at exactly one commit, the tip. Reachable history is found by
following `parents` transitively to commits with `parents == []`.

The two axes are traversed by different algorithms with different cost shapes (§6, §7)
and must not be conflated: a commit is one node on the time axis and the root of an
entire data-axis subgraph.

---

## 6. Reconstruction

### 6.1 The base-manifest chain

Reconstructing any byte range of a tensor is a **recursive walk backward through
`base_tensor_manifest` pointers**, not a single-hop lookup:

1. Resolve the tensor-manifest for the requested tensor at the requested commit.
2. Convert the requested byte range to a logical row range via `shape` and
   `itemsize(dtype)`.
3. Select the `chunks` entries overlapping that row range; decode each per §12.3.
4. If `base_tensor_manifest` is `null`, stop — the tensor is stored in full here.
5. Otherwise map the needed rows through `base_row_permutation` and the needed column
   blocks through `base_col_permutation`/`col_block_size` (§6.2), then recurse into
   step 1 for **only that reduced set** of base rows — never the full base tensor.
6. Apply the delta decode of §12.3 at each level on the way back up.

### 6.2 Index mapping, and the locality it destroys

For a requested set of target rows `R`, the base rows needed are `{ p[i] for i in R }`.

**This set is not contiguous.** With `p` null the mapping is the identity and a
contiguous target range is a contiguous base range. With a real permutation, locality is
gone. Measured on a 4096-row tensor chunked at 512 rows:

| | base chunks touched |
|---|---|
| 64 contiguous target rows, identity permutation | 1 of 8 |
| 64 contiguous target rows, real row permutation | **8 of 8** |

A small read at the target level can require decoding the *entire* base level. This is
inherent to permutation-aligned deltas, not a defect — but it must be designed around
rather than discovered:

- The FUSE layer's random-access latency metric is measured against permuted
  checkpoints, not only fine-tuned ones. Budget for the amplified case.
- §6.5's memoisation is not an optional nicety under permutation; it is what keeps
  repeated small reads from re-decoding the same base chunks.
- The fast path matters disproportionately. A `null` row permutation means contiguous
  base access and one chunk decode, which is why `null` — rather than a stored identity
  array — is the representation (§4.2.5).

Column mapping has no equivalent locality cost: chunks partition **rows**, so a column
permutation is applied within rows already being decoded.

### 6.3 Composing permutations across levels

When the recursion descends more than one level, per-level permutations **compose**.
Given `p1` on the target manifest and `p2` on its base:

```
target row i  →  base row p1[i]  →  base² row p2[p1[i]]
```

so the composed target→base² mapping is **`p2[p1]`**, gathering `p2` at the positions
named by `p1`. Equivalently, for any array `A`:

```
A[compose(f, s)] == A[f][s]
```

Composing in the other order yields a permutation that is a valid bijection of the
correct length and is numerically wrong — §4.2.4.3's invariants cannot detect it, and
only `content_hash` will. Implementations must use the alignment track's
`lap.compose(first, second)` and `lap.invert(p)`, which carry this semantics and a test
asserting the gather identity above, rather than re-deriving the order.

`null` composes as identity at any level, so an all-identity chain costs no gather at
all.

### 6.4 Whole-file reconstruction

```
file = header_object bytes  ‖  concat(tensors in header key order)
```

Each tensor's placement in the data region is given by its `data_offsets` in the stored
header. **Use those offsets; do not recompute the layout.** Recomputation reproduces the
common case and diverges on padding and ordering edge cases, which is precisely the
byte-exactness failure §4.2.3 exists to prevent.

### 6.5 Bounded depth and memoisation

**Chain depth — periodic full snapshots.** The number of recursion levels equals the
number of commits since that tensor was last stored in full. If every commit diffs
against its immediate parent forever, checking out the tip of a thousand-commit history
walks a thousand levels per tensor, turning an O(rows needed) operation into
O(rows needed × history length) — and, under §6.2's locality loss, closer to
O(full tensor × history length).

The system therefore defines a snapshot interval `K`: on a commit that would push a
tensor's chain depth past `K` since its last full-storage point, that tensor is stored
in full (`base_tensor_manifest: null`) even though a smaller delta was possible —
a keyframe. This bounds reconstruction depth, and therefore permutation-composition
depth, to `K` regardless of history length. `K` is a tunable trade-off between
reconstruction latency and storage size and must be chosen, measured, and documented
explicitly rather than left as an accidental consequence of "always diff against the
parent."

**Intra-reconstruction memoisation.** When several row ranges of the same tensor are
reconstructed within one call — the FUSE layer serving overlapping `read()`s, or
`checkout` materialising a whole tensor — the resolver caches decoded base results keyed
by `(tensor-manifest hash, row range)` for the duration of that call. Without it, two
requests whose ranges recurse into overlapping base rows decode and re-permute the same
data twice. Under §6.2 this is not a marginal saving.

---

## 7. History traversal

### 7.1 Reachability

Reachable history from a tip is a backward BFS/DFS via `parents` with a visited-set
keyed by commit hash. The visited-set makes the walk correct regardless of how many
merges or shared ancestors exist, with no special-casing on parent count.

Cost is O(reachable commits). `log` and full `verify` both pay it; §11.4 bounds `verify`
by a high-water mark, and a paginated `log` should bound itself the same way rather than
re-walking full history per invocation.

### 7.2 Merge-base discovery

Given tips A and B, the merge base is a **best common ancestor**: a commit reachable
from both, with no other common ancestor reachable from it.

The naive "symmetric BFS, stop at the first commit seen from both sides" is **not
correct in general.** BFS explores by distance from each tip, and the first commit
reached from both sides may be an *ancestor of* the true best common ancestor rather
than the best one itself, producing a merge base that is too old and reporting spurious
changes on both sides. Criss-cross histories can also have **several** best common
ancestors, none of which is an ancestor of the others.

The procedure is therefore:

1. Walk backward from both tips, marking each visited commit with which side(s) reached
   it. Continue until every frontier commit is marked from both sides.
2. Collect **candidates**: commits marked from both sides.
3. **Reduce**: discard any candidate reachable from another candidate. What remains is
   the set of best common ancestors.
4. If exactly one remains, it is the merge base.
5. If more than one remains, pick deterministically — lowest generation number, ties
   broken by lexicographically smallest commit hash — and record the choice in the
   merge commit's message. Determinism matters more than which one is chosen: two peers
   merging the same pair must reach the same result.

If A is reachable from B or vice versa, the merge is a **fast-forward**: no new
checkpoint-manifest is computed and the trailing ref simply moves to the leading tip.

### 7.3 Generation numbers

`generation(c) = 0` for a root commit, otherwise `1 + max(generation(p) for p in parents)`.

Computed incrementally at commit-write time from the parents' values, which are already
durable, and cached in `cache/generations`. It is **not** part of the commit's hashed
JSON — it is derived, and embedding it would make an immutable object's address depend
on when it was computed.

During §7.2's search, a commit whose generation is below the lowest generation still in
either frontier cannot be a common ancestor of anything currently being explored and is
skipped. This prunes the walk instead of visiting every reachable node — the same
optimisation Git's commit-graph provides for `merge-base`.

Because the cache is derived, a missing or stale `cache/generations` must degrade to
recomputation, never to a wrong answer. Implementations verify a cached generation
against the commit's parents on load, or rebuild the cache wholesale.

---

## 8. Merge

### 8.1 Preconditions

A merge is refused before any classification when:

- the two sides' `topology_config_hash` values differ — the checkpoints describe
  different graphs, and a per-tensor merge across different architectures is meaningless;
- either side fails verification at the default tier.

### 8.2 Per-tensor classification

For every tensor name appearing in the merge base's, A's, or B's checkpoint-manifest,
compare **`content_hash`**, not the tensor-manifest hash (see §4.2.4.2 for why).

| At base | At A | At B | Classification | Result |
|---|---|---|---|---|
| `H` | `H` | `H` | no-op | `H` |
| `H` | `H` | `H′` | B-only edit | B's manifest |
| `H` | `H′` | `H` | A-only edit | A's manifest |
| `H` | `H_a` | `H_b`, `H_a == H_b` | convergent edit | A's manifest |
| `H` | `H_a` | `H_b`, `H_a ≠ H_b` | **conflict** | halt (§8.3) |
| `H` | `H` | absent | B deleted, A untouched | delete |
| `H` | absent | `H` | A deleted, B untouched | delete |
| `H` | absent | absent | both deleted | delete |
| `H` | `H_a ≠ H` | absent | **conflict** (edit/delete) | halt |
| `H` | absent | `H_b ≠ H` | **conflict** (delete/edit) | halt |
| absent | `H_a` | absent | A-only addition | A's manifest |
| absent | absent | `H_b` | B-only addition | B's manifest |
| absent | `H_a` | `H_b`, `H_a == H_b` | convergent addition | A's manifest |
| absent | `H_a` | `H_b`, `H_a ≠ H_b` | **conflict** | halt |

When a side is chosen, its **tensor-manifest hash** is reused verbatim — no
re-storage, no recompression, no re-alignment. Under convergent edits the two manifests
may differ while their content matches; taking A's is arbitrary but deterministic, and
costs nothing because the chunks of both already exist.

Deletions were absent from the previous revision's table entirely. They are reachable:
any commit whose checkpoint has a different tensor set than its parent's produces them.

### 8.3 Conflict policy

A tensor conflicts when both sides changed it to different content since the merge base,
or when one side edited what the other deleted.

SynapseFS does **not** attempt numerical reconciliation. There is no meaningful analogue
of a three-way text merge for independently trained floating-point weights, and
automatic blending — averaging, say — produces a tensor corresponding to no state either
branch ever trained, which silently undermines §1.1.

On conflict the merge halts before producing any commit and reports every conflicting
tensor name. Resolution is explicit, per tensor:

- `--ours <tensor>` — take A's version
- `--theirs <tensor>` — take B's version
- supply a replacement `.safetensors` fragment, which is aligned and stored as a new
  tensor-manifest like any other update

A merge with unresolved conflicts never produces a commit. Exit code 6.

### 8.4 Producing the merge commit

With every tensor resolved to a single tensor-manifest hash: build the merged
checkpoint-manifest, write it as a loose object, write the commit object, then update
the ref via §9.1 — refs last, always.

```json
{
  "checkpoint_manifest": "<merged manifest hash>",
  "parents": ["<tip A>", "<tip B>"],
  "timestamp": "<now, RFC 3339 UTC>",
  "message": "<merge message>"
}
```

No chunk data is rewritten for any tensor whose resolved manifest already exists. A
merge, like any commit, pays storage only for content that is actually new.

### 8.5 The merge header problem

A checkpoint-manifest needs a `header_object`, and the previous revision did not say
which one a merge uses. Three cases:

1. **The merged tensor set, shapes, and dtypes exactly match one side.** Reuse that
   side's `header_object` verbatim. This is the common case, including every
   fast-forward.
2. **They match both sides identically.** Either; they are the same object.
3. **They match neither** — A added a tensor, B added a different one, or a deletion
   occurred. No stored header describes the merged set, so one is **synthesised**:

   ```
   __metadata__ first if present (union of both sides; conflicting keys are a conflict),
   then tensor keys sorted lexicographically,
   data_offsets assigned in sorted key order, contiguous from 0,
   header JSON space-padded so that (8 + header_len) % 8 == 0
   ```

   This is the sole exception to §4.2.3's "never rebuild a header." It is sound because
   a merge commit's checkpoint never existed as a file: there is no original for it to
   be byte-identical to. The synthesis rule is fixed and deterministic so that two peers
   merging the same pair produce the same header object and therefore the same commit.

---

## 9. Crash safety

The orderings below guarantee that a process killed at any point leaves the repository
either in its pre-operation state or its fully-completed post-operation state — never an
intermediate a subsequent `verify` cannot detect.

### 9.1 Atomic pointer writes

Every write to a ref, to `HEAD`, or to `objects/pack/order`:

1. Write the new content to a temp file **in the same directory** as the target.
2. `fsync` the temp file's descriptor.
3. `rename()` onto the target path — atomic when both are on the same filesystem.
4. `fsync` the containing **directory's** descriptor, making the rename durable.

Step 4 is not optional. Without it the rename may be lost on power failure even though
the file contents were synced.

This sequence is always the **last** step of any state-changing operation. Every object
the new ref will point to must already be durably written and hash-confirmed first.

### 9.2 Loose object writes

1. Write to `objects/tmp/<random>`.
2. `fsync`.
3. Hash the bytes; compare against the address the object is intended to occupy.
4. `rename()` into `objects/<hh>/<remaining-62>` only after step 3 succeeds.
5. `fsync` the shard directory.

If the destination already exists, the write is a no-op — content addressing makes it
provably the same bytes. Skip the rename and keep the existing file.

### 9.3 Pack construction

A pack is **not** built by renaming individual objects into it — a temp file cannot be
renamed "into" another file. A pack is assembled whole:

1. Build the complete pack in `objects/tmp/<random>`: 52-byte header, all records
   back-to-back, then the 32-byte trailer over `[0, EOF-32)`.
2. `fsync` the temp pack.
3. Hash it; `rename()` to `objects/pack/pack-<hash>.pack`; `fsync` the directory.
   **The pack is now sealed and immutable, and is never appended to.**
4. Build `pack-<hash>.idx` from the sealed pack, and `pack-<hash>.bloom` from its
   hashes. Write both via the same temp→fsync→rename→fsync-dir sequence.
5. Update `objects/pack/order` via §9.1.

**Packs before indexes, always.** The `.idx` and `.bloom` are derived; the `.pack` is
not. Reversing the order can leave an index referencing a pack that does not exist.

### 9.4 Commit write order

```
chunks → packs → indexes → order → loose objects (permutations, header,
tensor-manifests, checkpoint-manifest, commit) → ref
```

Each stage is durable before the next begins. The ref is the last write in the entire
operation; until it lands, the new commit is unreachable and the repository is exactly
as it was.

### 9.5 Recovery on startup

In order:

1. Delete everything under `objects/tmp/`. Anything there was never hash-confirmed and
   never referenced by anything in the DAG; discarding it can never lose committed data.
2. For each `.pack`, check for its `.idx`:
   - **`.idx` missing or corrupt, `.pack` trailer valid** → the pack is complete;
     **rebuild** the index and bloom filter by rescanning records from offset 52. This
     is not data loss.
   - **`.pack` trailer invalid or absent** → the pack was never sealed; it is a crashed
     partial write. No ref can reference it, because refs are written last. Discard it.

   The previous revision's "drop any `.pack` with no `.idx`" is wrong for the first case:
   it discards a sealed pack full of committed chunks over a missing derived file.
3. For each ref, confirm the target commit object exists and its bytes hash to that
   address.
4. On failure at step 3, do **not** attempt automatic repair. Mark that branch
   unavailable and return an explicit error naming the branch and the missing or
   mismatched hash. Do not silently roll back or substitute a tip.

Given §9.1–9.4, a ref never points at a commit whose object graph is not already fully
durable — so a step-3 failure indicates external interference or a filesystem that does
not honour the rename/fsync guarantees this design assumes, not an ordinary crash.

---

## 10. Determinism requirements

Content addressing deduplicates identical bytes. Anything that makes identical inputs
produce different bytes silently degrades the storage ratio with **no error and no
failing test**. Three sources, all of which must be pinned:

### 10.1 Canonical JSON

§2.2. Non-negotiable.

### 10.2 Alignment seeding

The permutation solver is a **search with multiple valid answers.** Coordinate descent
converges to a *local* optimum, and a different group visiting order can yield a
different — equally valid — permutation. A different permutation means a different
residual, different chunk hashes, and dedup that quietly stops working.

Therefore: the solver is seeded from a **fixed constant** (default `0`), never from
wall-clock time, PID, or `os.urandom`. The seed is recorded in the commit's `--json`
output. Re-running `commit` on the same `(base, target, config)` triple must produce
byte-identical residual chunks.

### 10.3 Chunk boundaries

Chunk boundaries are a deterministic function of `(shape, dtype, chunk_size)` alone —
never of available memory, thread count, or arrival order. Two peers committing the same
tensor with the same configured chunk size must produce the same chunk set, or
differential transfer (§14) degenerates to a full transfer.

---

## 11. Verification

`verify` confirms that every object reachable from a tip hashes to its claimed address,
without re-hashing a multi-gigabyte history on every call.

### 11.1 Two distinct hashes — do not conflate them

| Name | Covers | Where stored | Requires decompression? |
|---|---|---|---|
| **Content hash** | the chunk's **pre-compression** payload stream | the chunk's address, in `tensor-manifest.chunks[].object` and in the pack record header | **yes** |
| **Stored checksum** | the chunk's **compressed** bytes as they sit in the pack | 8-byte field in the `.idx` | no |

The previous revision described the index field as "an 8-byte prefix of the record's
full 32-byte BLAKE3 hash," verified against "a fresh partial hash of the payload." Both
halves are wrong:

- There is no partial BLAKE3. Obtaining the first 8 bytes of a digest requires
  processing **every** input byte; the prefix truncates the *output*, not the *work*. A
  prefix comparison saves 24 bytes of `memcmp`, not any hashing.
- If the index field were a prefix of the *content* hash, verifying it would require
  decompressing every chunk — contradicting the "no decompression" property the cheap
  tier exists to have.

The index field is a checksum over the **stored** bytes. The cheap tier's real savings
are that the compressed payload is smaller than the plaintext and that no zstd pass runs
at all.

### 11.2 Tiers

| Tier | Checks | Catches | Cannot catch |
|---|---|---|---|
| `--shallow` | DAG structure: commit → checkpoint-manifest → tensor-manifests → header/permutations, each re-hashed | broken links, structural corruption, tampered loose objects | anything inside a pack |
| *(default)* | above + each referenced chunk's stored checksum from the `.idx`, no decompression | bit-rot, truncation, damaged packs | **crafted payloads** |
| `--deep` | above + decompress each chunk and compare its content hash **against the hash recorded in the tensor-manifest** | malicious block injection | — |

**`--deep` must compare against the manifest's `chunks[].object`, not against the
32-byte hash in the pack record header.** An attacker rewriting a pack rewrites the
record header too, and the pack trailer, and the index checksum — all of it is inside
the artefact they control. Trust is rooted at the **ref**, and the only chain from the
ref down to the bytes runs commit → checkpoint-manifest → tensor-manifest →
`chunks[].object`. Verifying a pack against itself proves nothing.

For the same reason: the pack trailer detects bit-rot, not tampering.

Be accurate about this in `verify` output and in the README. The default tier **cannot**
detect a crafted payload carrying a matching stored checksum. Only `--deep` can. Exit 0
clean, exit 4 on any mismatch.

### 11.3 End-to-end check

`--deep` additionally re-derives each tensor's `content_hash` (§4.2.4.2) after full
reconstruction and compares it. This is the only check that catches a *semantically*
wrong but structurally valid reconstruction — most importantly a permutation composed in
the wrong order across a base chain (§6.3), which passes every structural invariant.

### 11.4 Incremental verification, and what it does not cover

`verify` maintains a per-branch, **per-tier** high-water mark in `cache/verified/<branch>`:
the most recent commit confirmed clean, and at which tier. A later call walks backward
from the tip only until it re-reaches that mark, then advances it. The first call on a
branch is a full walk; subsequent calls amortise to new history.

A mark set at one tier does not satisfy a request at a stronger tier. A `--shallow` mark
never short-circuits a default or `--deep` run.

**The limit, stated plainly:** the previous revision justified this by arguing that a
commit which passed verification "can never later fail without external tampering." That
conflates two different things. The *intended content* of a commit is immutable; the
*stored bytes* are not. Bit-rot is neither external nor tampering, and it can corrupt
already-verified history at any time. Incremental verification therefore covers only new
history and provides **no ongoing guarantee** about old history.

Consequently:

- `verify --full` ignores the high-water mark and re-walks everything. It exists
  precisely because incremental mode cannot detect rot in verified history.
- Long-lived repositories should scrub periodically with `--full`, on a schedule
  proportional to how much they care about cold data.
- The high-water mark is in `cache/`. Deleting it is always safe and forces a full walk.

---

## 12. Packfile and index

### 12.1 Pack — `pack-<hash>.pack`

```
+========== HEADER (52 bytes) ===============+
|  0   8   magic          "SYNPACK\0"        |
|  8   4   version        u32 = 1            |
| 12   4   flags          u32, bit0 = has dictionary |
| 16   4   count          u32                |
| 20  32   dict_hash      BLAKE3 or 32 zero bytes |
+========== RECORDS x count =================+
|     32   content_hash   BLAKE3 of the PRE-compression stream |
|      4   stored_len     u32                |
|      4   plain_len      u32                |
|      N   payload        stored_len bytes   |
+========== TRAILER (32 bytes) ==============+
|          BLAKE3 of bytes [0, EOF-32)       |
+============================================+
```

Records run back-to-back from offset 52; stride is `40 + stored_len`. **Index offsets
point at `payload`**, not at the record header — the header for a payload at offset `o`
occupies `[o-40, o)`. Packs are immutable once sealed.

### 12.2 Index — `pack-<hash>.idx`

Designed to be mmapped and binary-searched in place.

| Offset | Size | Field |
|---|---|---|
| `0` | 8 | magic `"SYNIDX\0\0"` |
| `8` | 4 | version, u32 = 1 |
| `12` | 4 | count, u32 = N |
| `16` | 32 | pack_hash |
| `48` | 1024 | fanout, 256 × u32, cumulative |
| `1072` | 32N | hashes, raw 32-byte, ascending |
| `1072+32N` | 8N | offsets, u64, payload offset into the pack |
| `1072+40N` | 4N | stored_len, u32 |
| `1072+44N` | 4N | plain_len, u32 |
| `1072+48N` | 8N | **stored checksum** — 8-byte BLAKE3 prefix over the *compressed* payload (§11.1) |
| `EOF-32` | 32 | trailer, BLAKE3 of `[0, EOF-32)` |

Entry cost is **56 bytes**; 280,000 chunks ≈ 15.7 MB. `fanout[b]` is the count of
entries whose first hash byte is `<= b`; `fanout[255] == count`. Arrays are **parallel,
not interleaved**, so a search touches only the `hashes` array.

```python
b  = h[0]
lo = 0 if b == 0 else fanout[b-1]
hi = fanout[b]
i  = bisect(hashes, h, lo, hi)      # compare 32-byte memoryview slices
```

**Do not build a Python dict of hex-string keys.** 280k entries costs 50–80 MB resident
against a graded peak-RSS metric. Binary-search the mmap.

### 12.3 Chunk encodings

| `encoding` | Payload | Decode |
|---|---|---|
| `raw` | uncompressed tensor bytes for the row range | memcpy |
| `raw-zstd` | zstd frame over tensor bytes | zstd decompress |
| `delta-zigzag-zstd` | zstd frame over zigzag-varint deltas | below |

`delta-zigzag-zstd`, encoding target chunk `B` against base rows `A` after applying the
row and column permutations:

```
1. key(x) = bits(x) ^ 0x8000   if the sign bit is clear      (16-bit dtypes)
   key(x) = ~bits(x)           if the sign bit is set
2. delta = int32(key(B)) - int32(key(A))
3. zz = (delta << 1) ^ (delta >> 31);  varint-pack zz    ← CONTENT HASH COVERS THIS
4. zstd(step-3 bytes)                                    ← this is the stored payload
```

Generalise step 1 to width `w` as `1 << (w-1)`. Decode reverses: zstd → un-varint →
un-zigzag → `key(B) = delta + key(A)` → invert `key()` → reinterpret as the dtype.

**No float arithmetic anywhere in encode or decode.** The entire path is integer, which
is what makes round-trip bit-exactness a property of the format rather than of floating-
point luck. (The alignment engine *is* permitted floating-point — it only chooses a
permutation, and a suboptimal choice costs ratio, never correctness.)

`-0.0` (`0x8000`) and `+0.0` (`0x0000`) are distinct bit patterns and must both survive
round-trip. Never normalise signed zero. NaN and Inf need no special-casing; adding any
is where the round-trip bug will be.

The content hash covers the **step-3** stream, never the zstd output. Hashing compressed
bytes would make dedup depend on compression level and dictionary state, so identical
content stored under different settings would fail to deduplicate.

### 12.4 Multi-pack lookup and the Bloom filter

Resolving a chunk hash probes packs in `objects/pack/order`, newest first. Recent chunks
hit in the first pack or two; old chunks, or genuine misses, degrade to probing every
pack.

Each pack therefore carries `pack-<hash>.bloom`, built at seal time and small enough to
keep all filters resident simultaneously (unlike `.idx`, mmapped on demand). A filter
negative skips that pack's real `.idx` search; a positive falls through to it.

**Integrity constraints — the filter is the one structure whose corruption content
addressing cannot catch:**

- Bloom filters have no false negatives *by construction*, but bit-rot flipping a `1` to
  a `0` manufactures them. The result is an existing object reported absent — a failure
  with **no hash mismatch to signal it**, unlike every case in §15.
- The filter is derived data in the same class as `.idx`: rebuildable by rescanning the
  pack, outside the crash-durability boundary, and carrying a trailing checksum over its
  own bytes. A filter failing its checksum is discarded and rebuilt, never trusted.
- A negative is conclusive **only for skipping one pack's probe**. If every pack returns
  a negative, the lookup falls through to at least one real `.idx` search before
  reporting the object absent. A corrupted filter must cost latency, never correctness.

---

## 13. Garbage collection and reachability

If a GC pass is implemented, reachability from the set of all refs plus `HEAD` follows
**every** edge below. The data-axis edges beyond the first hop are the ones that get
missed:

```
ref / HEAD
  └─ commit
       ├─ parents[*]                              (transitively)
       └─ checkpoint_manifest
            ├─ header_object
            └─ tensors[*]  →  tensor-manifest
                 ├─ base_tensor_manifest          (transitively — the chain, 6.1)
                 ├─ base_row_permutation
                 ├─ base_col_permutation
                 └─ chunks[*].object  →  chunk in a pack
                                          └─ that pack's dictionary, if flags bit 0
```

The base-manifest chain is a real edge. Pruning an "unreferenced" older tensor-manifest
because no *checkpoint-manifest* names it breaks checkout of every commit that diffs
against it. Likewise a `dictionary` is reachable only through the pack that names it, not
through any manifest.

Packs are immutable, so GC cannot delete individual chunks. Reclaiming space inside a
pack means writing a new pack containing only the reachable records and retiring the old
one — which is a repack, not a delete, and must follow §9.3's ordering.

---

## 14. Differential transfer

Push/pull negotiate on object identity, which content addressing makes trivial: the
client sends ref tips, the server computes the commit set, sends an object-ID list, the
client filters against what it already holds and requests only the remainder.

Two requirements follow from this document:

- **Build a transfer pack from the want-list.** Do not ship whole storage packs — the
  peer may already hold some of their chunks from another branch, and the graded
  property is that only missing blocks move.
- **Verify each object's hash on receipt, before writing it past §9.2's staging step.**
  Refs update last, exactly as in a local commit.

Resumability falls out of content addressing: a partial transfer leaves some objects
present, and re-running fetches only what is still missing. §10's determinism
requirements are what make this work between two peers rather than only within one
repository.

---

## 15. Threat model

**Covered:** accidental bit-rot, and malicious block injection by a compromised peer.

**Explicitly out of scope:** a peer presenting an entirely different, internally
self-consistent history — ref rollback or wholesale history replacement. Trust is rooted
at the locally accepted ref. This design guarantees that stored bytes match their
claimed hashes under the currently accepted lineage; it does not establish that the
lineage itself is authoritative against an adversarial peer.

### 15.1 Loose object tampering

Any byte change makes the object's BLAKE3 differ from the path it is stored under. A
reader that re-hashes on load detects this and raises a fatal mismatch rather than
returning corrupted content. The object that embeds this one's hash becomes unresolvable
at the point it is walked — surfaced as a broken link, never ignored.

### 15.2 Pack record tampering

A changed byte inside a sealed pack makes the record's content hash disagree with the
address the manifest requested. The engine rejects that record at read time; other
records in the same pack, independently addressed, remain readable. Note §11.2: catching
this requires `--deep`, because the cheap tier's checksum lives inside the artefact the
attacker controls.

### 15.3 Malicious injection during sync

A peer cannot get fabricated content accepted under a hash it does not match: the
receiver verifies before writing past staging (§14). A compromised peer can withhold or
refuse to serve data, but cannot cause a receiver to accept content that does not hash
to the requested address.

### 15.4 Cascading detection

Content addressing at every layer means a corrupted object is detectable at every point
the chain is walked, so every commit whose object graph transitively includes it becomes
unverifiable. (The previous revision said "every ancestor commit that references it";
ancestors do not reference descendants — the affected commits are those whose *data-axis
subgraph* contains the object, which on the time axis are its descendants.) There is no
partial-trust state: an object either verifies against its address or it does not.

---

## 16. Design rationale

| Decision | Alternative | Why |
|---|---|---|
| `parents` as an unbounded array | Separate merge-commit kind with two parent fields | One schema, one traversal path; costs nothing for `len == 1` and supports octopus merges unchanged |
| Flat `checkpoint-manifest` | Recursive tree object | `.safetensors` has no subdirectories; a flat map is still a Merkle node, so tamper-evidence is identical and the recursion is pure overhead |
| `content_hash` alongside the manifest hash | Compare manifest hashes in merge | The manifest hash covers the *encoding*; two branches can encode identical weights against different bases. Without it, merge conflicts on tensors that are byte-identical |
| Halt on merge conflict | Automatic numerical blending | No meaningful three-way merge exists for independently trained weights; blending produces a state neither branch trained |
| Chunks in sealed packs, mmap + binary search | One loose file per chunk | Hundreds of thousands of small files cost inode overhead or an in-memory table that blows the peak-RSS budget |
| Stored-payload checksum in the index | Prefix of the content hash | A content-hash prefix cannot be checked without decompressing, which defeats the cheap tier's entire purpose |
| Content hash over the pre-compression stream | Hash the zstd output | Hashing compressed bytes makes dedup depend on compression level and dictionary state |
| Bloom negative is advisory on a full miss | Bloom negative is conclusive | A corrupted filter otherwise produces "not found" with no hash mismatch — the one failure content addressing cannot catch |
| Incremental verify + explicit `--full` | Incremental verify alone | Immutable *intent* is not immutable *bytes*; rot in verified history is invisible to incremental mode |
| Header synthesised for merge commits only | Never rebuild a header | A merge checkpoint never existed as a file, so there is no original to be byte-identical to; every file-derived commit still replays verbatim |
| Fixed alignment seed | Whatever the solver picks | Coordinate descent has multiple valid optima; an unseeded solver silently breaks dedup with no error |
| Rebuild `.idx` from a sealed `.pack` | Drop a `.pack` with no `.idx` | The index is derived; discarding a sealed pack over a missing derived file destroys committed data |

---

## 17. Object reference

| Kind | Storage | Mutable | Addressed by |
|---|---|---|---|
| Ref | file, atomic replace | yes | branch name |
| HEAD | file, atomic replace | yes | fixed path |
| Pack order | file, atomic replace | yes | fixed path (rebuildable) |
| `commit` | loose | no | BLAKE3 of canonical JSON |
| `checkpoint-manifest` | loose | no | BLAKE3 of canonical JSON |
| `tensor-manifest` | loose | no | BLAKE3 of canonical JSON |
| `header` | loose | no | BLAKE3 of raw header bytes |
| `permutation` | loose | no | BLAKE3 of the raw int32 array |
| `dictionary` | loose | no | BLAKE3 of the blob |
| `chunk` | packed | no | BLAKE3 of the pre-compression payload stream |
| `.idx`, `.bloom`, `cache/*` | derived | rebuildable | not addressed |

---

## Appendix A — Changes from the previous revision

Ordered by severity. Each entry names the failure it prevents.

| # | Section | Change | Failure prevented |
|---|---|---|---|
| 1 | §4.2.4.3 | Column-permutation length invariant uses `prod(shape[1:])`, not `shape[1]` | As written, the invariant **rejected valid conv manifests**: `c2.weight` with `shape (8,6,3,3)` and `col_block_size 9` gave `6/9 = 0.667` and failed, where `54/9 = 6` is correct |
| 2 | §11.1, §12.2 | Index field is a checksum over **stored/compressed** bytes; "partial BLAKE3" claim removed | There is no partial BLAKE3 — a prefix truncates the output, not the work. The claimed saving did not exist, and a content-hash prefix would require decompression, defeating the cheap tier |
| 3 | §11.2 | `--deep` compares against the hash in the **tensor-manifest**, not the pack record | Verifying a pack against hashes stored inside that same pack proves nothing against an attacker who rewrote it |
| 4 | §6.3 | Permutation composition across the base chain specified, with direction (`p2[p1]`) | Composing the other way yields a valid bijection of the right length that is numerically wrong; no structural invariant catches it |
| 5 | §6.2 | Row permutation destroys base-side chunk locality — measured, 1 chunk → 8 of 8 | The random-access latency budget was written as if permuted and identity cases cost the same |
| 6 | §4.2.4.2, §8.2 | `content_hash` added; merge compares it | Merge reported conflicts on byte-identical tensors aligned against different bases |
| 7 | §8.2 | Deletion rows added to the classification table | Deletions were unrepresentable; a commit changing the tensor set had no defined merge behaviour |
| 8 | §8.5 | Merge header resolution, incl. the synthesis exception | Undefined: a merged tensor set matching neither side had no `header_object` and no rule permitted creating one |
| 9 | §7.2 | Merge base is a *best* common ancestor: candidates, reduction, deterministic tie-break | "First commit seen from both sides" can return an ancestor of the true base, and criss-cross histories have several |
| 10 | §11.4 | Incremental verify does not cover rot in verified history; `--full` added | The old rationale conflated immutable *intent* with immutable *bytes*; bit-rot is neither external nor tampering |
| 11 | §12.4 | Bloom filter: checksummed, rebuildable, advisory on a full miss | A corrupted filter produced "object not found" with no hash mismatch — the one corruption content addressing cannot catch |
| 12 | §9.5 | A sealed `.pack` with a missing `.idx` is **rebuilt**, not dropped | "Drop any `.pack` with no `.idx`" destroys committed chunks over a missing derived file |
| 13 | §9.3 | Pack construction described as whole-file assembly | The old text said temp files are "renamed into a pack under construction," which is not an operation |
| 14 | §2.1, §3 | Object path is `<2 hex>/<62 hex>`, stated once | §2 said paths hold 64 hex chars while §4.2 said 62 — the same object had two addresses |
| 15 | §4.1 | Three mutable files, not two — `objects/pack/order` included | `order` was mutated while the doc asserted only two files were ever modified in place |
| 16 | §2.3 | Row ranges are inclusive on both ends, stated once | The old text said both `[0, shape[0])` and "inclusive on both ends" |
| 17 | §10 | Determinism requirements: alignment seed, chunk boundaries, canonical JSON | An unseeded solver silently breaks dedup; nothing required it to be seeded |
| 18 | §13 | GC reachability spelled out, incl. base chain, permutations, dictionaries | Pruning by checkpoint-manifest references alone breaks checkout of every commit diffing against a pruned base |
| 19 | §4.2.2 | Manifest tensor set must match the header's key set | Reconstruction walks header key order; a mismatch silently omits or fails to resolve a tensor |
| 20 | §4.2.4.3 | `base_tensor_manifest` null ⟺ no delta encoding anywhere | A delta chunk with no base is unresolvable; a base with no delta is a pointer to nothing |
| 21 | §8.1 | Merge refuses on differing `topology_config_hash` | Per-tensor merge across different architectures is meaningless |
| 22 | §4.2.1, §7.3 | Generation numbers explicitly excluded from the commit hash; cache verified on load | A derived value inside an immutable object's identity makes its address depend on when it was computed |
| 23 | §15.4 | "every ancestor commit that references it" corrected | Ancestors do not reference descendants; the affected commits are descendants |
| 24 | §4.2.4.1 | `col_block_size` emitted at its true value even when the permutation is `null` | Readers requiring `1` alongside a null permutation would reject valid conv manifests |
| 25 | §1.1 | The byte-exactness guarantee stated once, at the top | It was implicit, so trade-offs elsewhere had nothing to be checked against |
