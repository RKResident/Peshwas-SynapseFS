# Loose chunk store

Status: **specification only.** Packs (`docs/FORMAT.md` §5–6) are what ships
today. This describes replacing the *storage* role of packfiles with
content-addressed loose files, keeping the pack format for its *transfer* role.

```
.synapse/objects/
    <ab>/<62-or-64 hex>              commits, manifests, headers, configs  (unchanged)
    chunks/<shard...>/<64 hex>       chunk payloads                        (new)
    pack/                            transfer packs only, transient
    tmp/                             staging for atomic writes
```

---

## 1. Path layout

```
objects/chunks/ <s₁> / <s₂> / ... / <s_d> / <64-hex content hash>

s_i = content_hash_hex[2(i-1) : 2i]          # one byte per level
d   = SHARD_DEPTH
```

For `abcdef01…`: depth 1 → `chunks/ab/abcdef01…`; depth 3 →
`chunks/ab/cd/ef/abcdef01…`.

The hash is BLAKE3-256 of the chunk's **uncompressed stream** — identical to
what `chunks[].object` records in a tensor-manifest today, and to the
`content_hash` a pack record carries. Chunk identity does not change, so a
repository can be converted in either direction without re-encoding anything.

### 1.1 SHARD_DEPTH

A repository-wide constant recorded in `.synapse/config` at `init`, because a
reader must know where to look and must not guess by probing.

| depth | leaf buckets | directories at N=20k | at N=100k | use when |
|---|---|---|---|---|
| **1** *(default)* | 256 | 256 → 1.0 MiB | 256 → 1.0 MiB | N < ~1M |
| 2 | 65,536 | 17,499 → 68 MiB | 51,510 → 201 MiB | N > ~1M |
| 3 | 16.7M | 37,490 → **146 MiB** | 151,192 → **591 MiB** | N > ~100M |

Directory counts are measured, not estimated (uniform random 32-byte keys).

The trap is that sharding depth is usually chosen to bound *entries per
directory*, and at depth 1 that is `N/256` — 78 files per directory at N=20k,
which sounds like a lot. It isn't: ext4 has indexed directories (`dir_index`,
on by default), so lookup in a 78-entry directory is a htree probe, not a scan.
Meanwhile each *extra* level costs a 4 KiB directory inode per distinct prefix,
and because hashes are random, N ≪ 256^d means "one leaf directory per chunk".
Depth 3 spends more on directory metadata than on chunk content until roughly
100M chunks.

Depth is a parameter rather than a constant so this can be revisited with a
number rather than an argument. **Default 1.**

### 1.2 Filename

The leaf filename is the **full 64-hex hash**, not the remainder after the
shard prefix.

This is a live disagreement in the team — `docs/kris_docs.md` specifies
`<2 hex>/<62 hex>` for loose objects, this document specifies `<2 hex>/<64 hex>`
for chunks, and the two are mutually unreadable. It has to be settled once for
*both* stores before either is implemented.

The case for the full hash: a chunk file is then self-identifying. It can be
`cp`'d out, listed by `find`, checked by `b3sum *`, or recovered from a
half-copied tree, all without reconstructing its identity from its path. A
truncated name saves 2 bytes per filename and makes every one of those
operations require path context. The redundancy is the point.

---

## 2. What moves into the tensor-manifest

Four fields lived in the pack index. Three are recovered for free; one has to
move, and moving it is an improvement.

| field | new home |
|---|---|
| `offset` | gone — a chunk file starts at 0 |
| `stored_len` | `stat().st_size` |
| `plain_len` | **tensor-manifest** `chunks[].plain_len` (new) |
| `checksum` | **tensor-manifest** `chunks[].stored_checksum` (new) |

```json
{
  "row_start": 0,
  "row_end": 95,
  "encoding": "delta-zigzag-zstd",
  "object": "38d3dea1…",
  "plain_len": 1536,
  "stored_checksum": "bbaa17e69c87186d"
}
```

`plain_len` could be read from the zstd frame header instead, but that makes
the format depend on the compressor writing a content-size field. Recording it
is explicit and costs ~20 bytes of JSON per chunk.

### 2.1 `stored_checksum` makes the fast tier tamper-proof

This is the most consequential change in the document and it is a side effect,
not a goal.

Today the 8-byte checksum of a chunk's compressed bytes lives in the pack
index. FORMAT.md §12B explains why that makes `verify --fast` a rot scan and
nothing more: the index is a file an attacker rewrites alongside the payload,
so the check is self-certifying.

Moved into the tensor-manifest, that same checksum becomes **ref-anchored** —
covered by the manifest hash, which is covered by the checkpoint-manifest,
which is covered by the commit, which is the trusted root (PS §2.c). An
attacker who substitutes a chunk file cannot adjust the checksum without
changing the commit hash.

So under this layout the tiers become:

| tier | anchored to | detects | rate |
|---|---|---|---|
| `--shallow` | ref | structure, broken links | — |
| `--fast` | **ref** | **rot *and* tampering** | ~694 MiB/s |
| `--deep` | ref | as above, plus a forged 8-byte prefix (2⁶⁴ work) | ~239 MiB/s |

`--fast` stops being a compromise. Full tamper detection without decompressing,
at ~2.9× the throughput, with `--deep` remaining as the belt-and-braces tier.
Whether `--deep` should stay the default becomes an open question again.

---

## 3. Operations

**Write.** `atomic_write(chunks/<shard>/<hash>, payload, tmp_dir=objects/tmp)`.
Same primitive as every other durable write. A partially written chunk cannot
be observed; a crash leaves the file absent, and the commit that would have
referenced it never landed its ref.

**Read.** `open(path) → read() → close()`, or a bounded LRU of open fds in the
FUSE daemon. Measured cost against a held-open pack fd: **50.4 µs vs 28.0 µs
per chunk, ~3 syscalls vs 1** (warm cache; the gap widens cold).

**Existence / dedup.** `path.exists()`. No index probe, no mmap, no fanout.

**Enumeration.** `os.walk(objects/chunks)`. Needed by GC and `fsck`.

**Delete.** `unlink`. Unreferenced chunks are removed individually with no
rewrite of anything else.

---

## 4. Packs after this change

`pack.py` stays; `index.py` and `packset.py` mostly go.

FORMAT.md §5 already gives packs two roles — *storage pack* (one per commit)
and *transfer pack* (built on demand from a want-list). This change deletes the
first and keeps the second. A transfer pack is written, streamed, unpacked into
loose chunks by the receiver, and deleted. It is never indexed, never mmap'd,
never read at random, and never persists — so it needs no `.idx`, no fanout, no
`order` file and no recovery path.

That removes, in full: the pack index format (v1 and the proposed v2),
`PackSet`, `recover_packs`, the `order` file, index rebuild-on-open, and the
"verify must not repair" hazard that comes with it.

---

## 5. Migration

Both directions are pure re-arrangement — chunk identity is unchanged, so
nothing is re-encoded, re-compressed or re-hashed.

- **pack → loose**: `scan_pack` each pack; write each payload to its shard
  path; add `plain_len`/`stored_checksum` to each manifest chunk entry, which
  rewrites the tensor-manifests and therefore every ancestor object up to the
  ref. Commits change hash. This is a history rewrite and must be presented as
  one.
- **loose → pack**: feed the store to `PackWriter` + `write_index`.

The manifest change is not backward compatible: a reader without
`stored_checksum` cannot run the ref-anchored fast tier, and a reader without
`plain_len` must fall back to the zstd frame header. Bump the tensor-manifest
schema version.

---

## 6. Trade-offs

### For

1. **Deletes the index entirely.** The filesystem is the index — its own htree
   is the lookup structure, maintained by the kernel, already crash-safe. About
   270 lines of `index.py` plus `packset.py`'s lookup path, and with them the
   whole v1/v2 fanout question, the `order` file, and index rebuild-on-open.
2. **GC becomes `unlink`.** Reclaiming an unreferenced chunk from a pack means
   rewriting the pack and its index — the known-painful part of git's design.
   Here it is one syscall. Given the PS grades a `gc`-adjacent story not at all
   but a *demo* heavily, "delete an old branch and watch the repo shrink" is a
   thing that suddenly works.
3. **`verify --fast` becomes ref-anchored** (§2.1) — full tamper detection at
   ~694 MiB/s instead of ~239. This is the strongest technical argument and it
   is not obvious from the outside.
4. **Crash recovery gets smaller.** No index can be lost, so nothing has to be
   rebuilt, so `verify` no longer has to defend against repairing what it
   inspects.
5. **Dedup is a `stat`.** No index open, no mmap, no 1 KiB fanout per pack.
6. **Fixes the current 35% index overhead** by deleting the thing that has it —
   one pack per commit holding ~32 chunks, each with a 1,072-byte header.

### Against

1. **Read path is ~1.8× slower and 3× the syscalls** (50.4 µs vs 28.0 µs per
   chunk, warm). Measured against a held-open pack fd; cold, the gap widens,
   and PS module 3 grades cold-cache read throughput and daemon CPU.
2. **Content addressing destroys locality, and this exposes it.** 810 chunks
   land in 248 directories. Chunks read together — consecutive row ranges of
   one tensor — are scattered maximally, because a hash has no relationship to
   position. In a pack they are adjacent in row order and one readahead
   collects several. No sharding scheme fixes this; deeper sharding worsens it.
3. **No `mmap` of the whole store.** A pack maps once; 20,000 files cannot.
4. **fsync amplification**: 2 per chunk vs 2 per commit — 1,620 vs 2 at 810
   chunks. Measured cost on NVMe is only **1.5×** wall-clock, so this is real
   but modest, and would matter far more on slower storage.
5. **Directory inodes**, per §1.1 — bounded to 1 MiB at depth 1, catastrophic
   at depth 3.
6. **Internal fragmentation**: +1% at depth 1 measured. 81% of chunks are under
   4 KiB (median 394 B) and each takes a full block, but a few multi-MB chunks
   carry the byte total. Weaker than it looks, and it shrinks at real
   checkpoint sizes where chunks approach the 4 MiB target.
7. **Inode consumption**: one per chunk, ~20k at realistic scale. Fine on ext4,
   worth knowing on a filesystem provisioned with a fixed inode table.
8. **The transfer path still needs packs**, so the pack writer cannot be
   deleted — only its index and its persistence.

### Net

The read-path cost (1, 2, 3) is paid on the graded metric. Everything in the
"for" column is paid in complexity, which is graded only indirectly — through
the Q&A, where "we deleted the index because the filesystem already is one" is
a much better answer than a fanout table nobody can justify at N=32.

§2.1 is the item that does not fit that framing: it is a straight improvement
to the integrity module, worth 20% of the grade, and it is unavailable in the
pack design at any price.
