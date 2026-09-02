# SynapseFS — On-Disk Format Specification

Status: **draft, frozen for Phase 1 implementation**. Anything marked `OPEN QUESTION`
is a real unresolved decision the team needs to make explicitly — do not silently
pick a default for those; ask, then update this doc.

This document is the single source of truth for the object model. Track A (storage/CLI),
Track B (alignment/codec), and Track C (FUSE) all code against this. If an implementation
detail isn't decided here, it belongs here before it's decided in code.

---

## 1. Design principles

1. **Everything is an object.** The content-addressed store (CAS) has exactly one
   physical concept: a file at `objects/<hash>` whose name is the hash of its own
   bytes. There is no separate database. "Object kind" (blob / header / tensor-manifest
   / checkpoint-manifest / commit) is a distinction the *application code* makes when
   interpreting an object's bytes — the filesystem does not know or enforce it.
2. **Manifests are indexes, not payloads.** A manifest object never contains tensor
   data itself. It contains hashes pointing at the objects that do. This gives uniform
   dedup, verification, and sync behavior at every granularity (chunk, tensor,
   checkpoint, commit) for free — see §7.
3. **Byte-exactness comes from never re-deriving bytes we can store verbatim.** The
   `.safetensors` header is stored as raw bytes and replayed unmodified. Reconstruction
   is `header_bytes ‖ data_region`, nothing is re-serialized.
4. **Reconstruction must be exact by construction, not by luck.** All delta math
   happens in an integer domain (§5). No floating-point operation is ever performed
   during commit or reconstruction that could introduce a rounding difference.

---

## 2. Object addressing

- **Hash algorithm:** BLAKE3, 256-bit output, lowercase hex encoding (64 chars).
- **Hash input:** the object's bytes **as stored on disk** — i.e. for `blob` objects,
  the hash is over the *compressed* bytes, not the decoded tensor data. This is why
  `verify` never needs to decompress or reconstruct anything (§9).
- **Directory layout:**

```
<SYNAPSE_DIR>/
  objects/
    tmp/<random>              # write staging, never a final resting place
    <hash[0:2]>/<hash>        # final immutable objects, sharded by first byte
  refs/
    heads/<branch-name>       # text file containing a commit hash
  HEAD                        # text file: "ref: refs/heads/<branch>" or a raw commit hash (detached)
```

- **Immutability:** once an object exists at `objects/<h[0:2]>/<h>`, it is never
  modified or deleted (except by an explicit, out-of-scope GC). New content always
  gets a new hash and a new path.

---

## 3. Write protocol (crash safety)

Every object write, regardless of kind, follows this sequence:

1. Write full bytes to `objects/tmp/<random>` on the **same filesystem** as `objects/`.
2. `fsync(fd)` the file.
3. `rename(objects/tmp/<random>, objects/<h[0:2]>/<h>)` — atomic on the same filesystem.
4. `fsync()` the containing directory (`objects/<h[0:2]>/`) so the rename itself survives a crash.

Refs follow the identical pattern (write-temp → fsync → rename → fsync directory),
and **refs are always the last thing updated** in any operation (commit, merge, pull).
This means an interrupted operation can leave orphaned objects (harmless — nothing
points at them yet) but can never leave a ref pointing at a partially-written or
inconsistent commit.

**On startup:** scan `objects/tmp/`. Any file there is the product of an interrupted
write and is safe to delete (nothing durable ever references a tmp path). If a ref
file itself is malformed or missing, refuse to proceed rather than guessing.

---

## 4. Object kinds

### 4.1 `blob`

Raw bytes of a single compressed chunk. Opaque to the storage layer — the
`encoding` field in the referencing tensor-manifest chunk entry tells the reader
how to decode it. See §5–6 for what's inside.

### 4.2 `header`

The raw, verbatim `.safetensors` header region for one checkpoint file:

```
[8 bytes, little-endian u64: header_len N]
[N bytes: the exact JSON header, including __metadata__, exact key order, exact whitespace]
```

Stored byte-for-byte as taken from the source file. Never re-serialized. This is
what makes header reconstruction trivially byte-exact — replay, don't rebuild.

### 4.3 `tensor-manifest`

One tensor-manifest per **tensor**, e.g. one for `layer1.weight`, a separate one
for `layer1.bias`, a separate one for `layer2.weight`. It is the index of that
tensor's chunks — see §5 for full schema and worked example.

### 4.4 `checkpoint-manifest`

One per checkpoint version. Maps every tensor name to its tensor-manifest hash, and
points at the header object. Schema:

```json
{
  "header_object": "<blake3-hash>",
  "tensors": {
    "layer1.weight": "<tensor-manifest-hash>",
    "layer1.bias":   "<tensor-manifest-hash>",
    "layer2.weight": "<tensor-manifest-hash>",
    "...":           "..."
  },
  "topology_config_hash": "<blake3-hash of the config.json this checkpoint was aligned against>"
}
```

**Reuse rule:** when committing a new checkpoint version, compute a candidate
tensor-manifest for every tensor. If a candidate's hash equals the parent commit's
hash for that same tensor name, **do not write a new tensor-manifest object** —
reuse the existing hash in the new checkpoint-manifest's `tensors` map. Alignment
and delta-encoding compute still run (this is a storage-layer optimization, not a
compute-skip — see §8's identity fast path for the compute-skip), but no new object
is written for unchanged tensors.

### 4.5 `commit`

```json
{
  "checkpoint_manifest": "<hash>",
  "parents": ["<hash>", ...],
  "timestamp": "2026-08-24T12:00:00Z",
  "message": "commit message"
}
```

`parents`: empty array for the root commit, one entry for a normal commit, two or
more entries for a merge commit.

---

## 5. Tensor-manifest schema (detailed)

```json
{
  "name": "layer1.weight",
  "dtype": "bf16",
  "shape": [4096, 4096],
  "row_order": "stored",
  "row_permutation": [<int>, ...] | null,
  "col_permutation_group": "<group-id>" | null,
  "base_tensor_manifest": "<hash>" | null,
  "chunks": [
    {"row_start": 0,    "row_end": 511,  "encoding": "delta-zigzag-zstd", "object": "<blake3-hash>"},
    {"row_start": 512,  "row_end": 1023, "encoding": "raw-zstd",          "object": "<blake3-hash>"},
    {"row_start": 1024, "row_end": 1535, "encoding": "delta-zigzag-zstd", "object": "<blake3-hash>"}
  ]
}
```

### Field semantics — read this carefully, it's the part that's easy to get subtly wrong

- **`row_start` / `row_end`** are **row indices in `row_order`, inclusive**, *not* byte
  offsets and *not* an index into any shared file. Each chunk entry's `object` field
  points at an **independent** `blob` object containing only that chunk's compressed
  bytes. There is no concatenated file that `row_start`/`row_end` slice into — the
  chunk boundary information exists purely so a reader can compute, given a byte range
  in the *reconstructed logical tensor*, which chunk objects overlap that range.

- **`row_order` is always `"stored"`.** This is the load-bearing decision that
  resolves the stored-vs-logical ambiguity: `row_start`/`row_end` index into the
  physical on-disk row order used when this tensor-manifest was written — i.e. the
  order chunks are laid out and hashed in. They do **not** index into the logical
  neuron ordering of the target checkpoint. **Any code that needs logical row `k`
  must first apply `row_permutation` to find which stored row holds it, then locate
  the chunk covering that stored row.** Do not skip this translation step, and do not
  let any component (FUSE reconstruction, alignment, verify) assume `row_start`/`row_end`
  are already logical — that assumption is the "very quiet, very nasty bug" this field
  exists to prevent.

- **`row_permutation`**: an array of length `shape[0]` where `row_permutation[i]` is
  the **logical row index** stored at **stored row `i`**. `null` means identity (stored
  order == logical order) — this is the common case for a real fine-tuning pair and
  should be the fast path everywhere it's checked, not just in the alignment solver.

- **`col_permutation_group`**: a reference into the topology IR's permutation groups
  (see the topology IR doc — not this file). Column permutation of a weight matrix is
  never encoded here directly; it's resolved by looking up which group this tensor's
  input axis belongs to and reading that group's permutation off the *previous* layer's
  output axis. This keeps permutation state in exactly one place per group instead of
  duplicated across every tensor that shares it.

- **`base_tensor_manifest`**: the tensor-manifest this one was diffed against, or
  `null` if this tensor is stored in full (first commit for this tensor, or the
  alignment engine determined "not alignable" and fell back to raw storage — see §8).
  Reconstructing this tensor requires first reconstructing the tensor at
  `base_tensor_manifest`.

- **`encoding`** (per-chunk, not per-tensor — different chunks of the same tensor
  may use different encodings):
  - `"delta-zigzag-zstd"` — monotone-int delta against the corresponding chunk of
    `base_tensor_manifest`, zigzag-encoded, then zstd-compressed. See §6.
  - `"raw-zstd"` — no base, no delta; raw tensor bytes for this chunk, zstd-compressed.
    Used when `base_tensor_manifest` is `null`, or per-chunk when delta encoding
    doesn't help (see the codec benchmark decision in §6).

- **Chunk axis**: chunks always partition the tensor's **row / output-channel axis**
  (dim 0 after any conv→2D flattening — see the topology IR doc for the flatten rule).
  This is what makes a row permutation a pure manifest edit (reorder `row_permutation`
  and, if needed, reorder which chunk entries are listed) rather than a data rewrite.

- **Chunk size**: `OPEN QUESTION` — needs a benchmarked default (target: keep
  post-compression chunk size in the ~1–4 MB range so zstd frame overhead doesn't
  eat the residual-ratio metric on fine-grained changes, while staying small enough
  that FUSE reads don't decode more than necessary). Record the chosen default and
  the benchmark that justified it here once §1.2's codec benchmark is run.

---

## 6. Delta encoding (per chunk, `delta-zigzag-zstd`)

Given a chunk of the target tensor `B` and the corresponding chunk of the base
tensor `A` (same shape, same stored row range):

1. **Bit-pattern → monotone integer key** (per element, operating on raw fp16/bf16
   bit patterns, never on decoded float values):
   ```
   key(x) = bits(x) ^ 0x8000     if sign bit clear
   key(x) = ~bits(x)             if sign bit set
   ```
   This is order-preserving over the value domain, so numerically small changes
   between `A` and `B` produce small integer deltas.

2. **Delta:** `delta = int32(key(B)) - int32(key(A))`, element-wise.

3. **Zigzag encode** each delta to an unsigned integer (standard zigzag:
   `zz(n) = (n << 1) ^ (n >> 31)`), then bitpack/varint.

4. **zstd-compress** the resulting byte stream. This is the `blob` object's content.

**Reconstruction** reverses each step exactly: zstd-decompress → un-zigzag →
`int32(key(B)) = delta + int32(key(A))` → invert `key()` back to the original
bit pattern → reinterpret as fp16/bf16. Every step is integer arithmetic on raw
bit patterns; no float rounding occurs anywhere in this path, which is what makes
byte-exact reconstruction unconditional rather than "usually exact."

`raw-zstd` chunks skip steps 1–3 entirely and zstd-compress the tensor bytes directly.

---

## 7. Why this gives dedup, verification-speed, and resumable sync uniformly

Because every level (blob, tensor-manifest, checkpoint-manifest, commit) is a
content-addressed object referencing children by hash, the same three properties
hold at every granularity without separate mechanisms:

- **Dedup**: identical content at any level (a chunk, a whole tensor, a whole
  checkpoint) hashes to the same object and is stored once. See §4.4's reuse rule.
- **Verification**: `verify` is one recursive hash-check-and-descend from a commit
  down through checkpoint-manifest → tensor-manifests → blobs. It only touches
  compressed bytes, never a materialized multi-GB model — this is the argument for
  why verification stays fast at scale (the graded verification-time metric).
- **Resumable sync**: the have/want protocol (see the networking doc) operates on
  object hashes uniformly. A partial transfer just means some objects in the DAG
  exist and some don't; resuming is re-running the same negotiation, which
  naturally skips objects already present.

---

## 8. Not-alignable fallback

If the alignment engine's relative-residual-norm check (post-alignment vs.
pre-alignment) shows alignment did not materially reduce the difference, the
tensor is stored with `base_tensor_manifest: null` and all chunks `raw-zstd` —
i.e. treated as if it were the first commit for that tensor. This must be reported
explicitly by the CLI (not silently produced as a low-quality diff) per the PS's
requirement to recognize and report non-alignable pairs.

---

## 9. Open questions to resolve before Phase 1 is "done"

- [ ] Default chunk size / row-count per chunk (needs the §1.2 codec benchmark).
- [ ] Exact topology-IR permutation-group schema and its JSON shape (separate doc,
  referenced by `col_permutation_group` here — must be frozen before Track B's
  permutation-group resolver is implemented).
- [ ] Whether `row_permutation: null` (identity) is enforced as a required
  optimization (skip storing an explicit `[0,1,2,...,n-1]` array) or always
  materialized — affects tensor-manifest size for the common identity case.
