# SynapseFS pack index, version 2

Status: **specification only** — v1 (`docs/FORMAT.md` §6) is what ships today.
This document is written to be implementable from scratch, in C++ or anything
else, without reading the Python.

v2 is v1 plus an optional second-level radix table. Every field v1 carries is
carried here: `pack_hash`, `count`, the full 32-byte hashes, payload offsets,
`stored_len`, `plain_len`, the 8-byte payload checksum, and the trailer.
Nothing is dropped, and no field changes width or meaning.

---

## 1. What the second level is

A trie over a fixed-length key, flattened into an array, is a **radix table**:
the pointer structure disappears because the child's index is computable from
the key instead of stored. For 32-byte BLAKE3 hashes — uniformly random, no
shared prefixes — this is strictly better than a node-and-pointer trie:

| | pointer trie | flattened radix |
|---|---|---|
| lookup | 2–3 *dependent* loads (pointer chase) | 2 *independent* array loads |
| parsing | node headers, child bitmaps | none — offset arithmetic |
| allocation | per node | none, `mmap` and cast |
| space at 2 levels | 256 B–1 KiB per node, × nodes | 1 KiB + 256 KiB, fixed |

Both are "a trie over two characters." Only one of them is a good idea when
the keys are random.

**One level (v1) is already enough for small packs.** `log₂₅₆(N)` is the
expected number of key bytes needed to isolate one entry among N random keys:
1.7 at 10k, 2.3 at 280k. Measured on a real 25-commit repository, the v1
fanout narrows a lookup to a mean of **1.13 entries — 1.08 binary-search
probes.** The second level exists for the case where packs are consolidated
and N reaches tens of thousands, not for today's 32-entry packs.

---

## 2. Byte layout

All integers are **little-endian**, all fields fixed-width. There are no
varints and no length-prefixed anything, so a reader may `mmap` the file and
cast. Let `N = count`.

```
offset            size        field
------            ----        -----------------------------------------------
0                 8           magic        "SYNIDX\0\0"
8                 4           version      u32 = 2
12                4           count        u32, N
16                32          pack_hash    blake3 of the .pack this indexes
48                4           flags        u32; bit0 = L2 present, others 0
52                4           _reserved    u32, must be 0

56                1024        L1           256   x u32, cumulative
1080              262144      L2           65536 x u32, cumulative
                              (present if and only if flags bit0 is set)

A                 N * 32      hashes       content hashes, ascending
A + 32N           N * 8       offsets      u64, payload offset within the pack
A + 40N           N * 4       stored_len   u32, bytes as stored (compressed)
A + 44N           N * 4       plain_len    u32, bytes after decompression
A + 48N           N * 8       checksum     first 8 bytes of blake3(stored payload)

EOF - 32          32          trailer      blake3 of bytes [0, EOF-32)
```

```
A = 1080 + (262144 if L2 present else 0)
file size = A + 56N + 32
```

`_reserved` at offset 52 is padding, not decoration: it makes `A` a multiple
of 8 in both variants (1080 = 8·135, 263224 = 8·32903), so the `offsets` array
is naturally 8-byte aligned and a C++ reader can
`reinterpret_cast<const uint64_t*>` it without an unaligned load.

### 2.1 Field meanings

`offsets` points at the **payload**, not at the record header. FORMAT.md §5.2:
the 40-byte record header occupies `[offset-40, offset)`. This makes a read a
single `pread(fd, buf, stored_len, offset)` with no arithmetic.

`checksum` is blake3 of the **stored (compressed)** bytes, truncated to 8. It
answers "did these bytes rot?" and is checked without decompressing.
It is *not* the chunk's identity — that is the `hashes` entry, which covers the
**uncompressed** stream. The two hash different bytes on purpose: identity must
survive recompression at a different level, a rot check must not.

Dropping `checksum` would delete `verify --fast` and the cheap integrity check
on the FUSE read path. Do not.

---

## 3. Fanout semantics

Both tables are **cumulative counts**, in the git sense:

```
L1[b] = number of entries whose hashes[i][0]      <= b          for b in 0..255
L2[k] = number of entries whose (h[0]<<8 | h[1])  <= k          for k in 0..65535
```

Cumulative rather than per-bucket because it yields both bounds from two
adjacent loads with no sentinel and no branch:

```
lo = (k == 0) ? 0 : L2[k-1];
hi = L2[k];
```

Invariants a validator must check:

1. `L1[255] == N` and, if present, `L2[65535] == N`
2. both tables non-decreasing
3. `L1[b] == L2[(b << 8) | 0xFF]` for all `b` — L1 is redundant with L2 and
   must agree with it
4. `hashes` strictly ascending (byte-lexicographic), which also forbids
   duplicates
5. `trailer == blake3(file[0 .. EOF-32))`

L1 is written **always**, even when L2 is present. It costs 1 KiB, it keeps the
no-L2 path and the L2 path structurally identical, and it answers abbreviated-
hash prefix queries without paging in 256 KiB.

---

## 4. When to write L2

Write L2 when `N >= 16384`; otherwise leave flags bit0 clear and omit it.

The threshold is where the table stops being overhead. L2 is a fixed 262,144
bytes and the rest of the file is `56N + 1112`, so L2's share drops below 25%
at `N ≈ 14,000`; 16384 is the next power of two above that. Below it, L1
narrows to `N/256`
entries — 64 at N = 16k — and binary-searching 64 sorted hashes costs ~6
probes over ~2 KiB, which beats faulting in a 256 KiB table that is 99.9%
zeros.

A reader must handle both variants. A writer may use any threshold; the flag,
not the threshold, is normative.

---

## 5. Lookup

```cpp
// mm: mmap'd index. h: 32-byte content hash. Returns index or NOT_FOUND.
uint32_t lookup(const Index& ix, const uint8_t h[32]) {
    uint32_t lo, hi;

    if (ix.flags & FLAG_L2) {
        const uint32_t k = (uint32_t(h[0]) << 8) | h[1];
        lo = k ? ix.L2[k - 1] : 0;
        hi = ix.L2[k];
        // hi - lo is expected N/65536 < 1; a linear scan is optimal here and
        // a binary search would only add a branch.
        for (uint32_t i = lo; i < hi; ++i)
            if (std::memcmp(ix.hashes + 32 * i, h, 32) == 0) return i;
        return NOT_FOUND;
    }

    lo = h[0] ? ix.L1[h[0] - 1] : 0;
    hi = ix.L1[h[0]];
    while (lo < hi) {                      // ~log2(N/256) probes
        const uint32_t mid = lo + (hi - lo) / 2;
        const int c = std::memcmp(ix.hashes + 32 * mid, h, 32);
        if (c < 0)      lo = mid + 1;
        else if (c > 0) hi = mid;
        else            return mid;
    }
    return NOT_FOUND;
}
```

With L2 this is O(1) expected: two loads and an expected 0.15 comparisons at
N = 10,000. Without it, O(log(N/256)).

Note `mid = lo + (hi - lo) / 2`, not `(lo + hi) / 2` — the latter overflows
u32 only past 2³¹ entries, but the correct form is free.

### 5.1 Prefix lookup (abbreviated hashes)

`Repo._lookup_hash` resolves user-typed prefixes. Because `hashes` is sorted,
a prefix query is `lower_bound(prefix) .. upper_bound(prefix)`, and the result
is ambiguous iff that range holds more than one entry. Use L1/L2 to seed the
bounds when the prefix is at least 1 or 2 bytes. This is the one operation a
trie would do natively and a plain hash table could not do at all.

---

## 6. Reading a whole index

`entries()` — used by `verify` and by the pack/index cross-check test — is a
sequential walk of the five parallel arrays. No random access, no fanout.

The arrays are parallel rather than interleaved for a reason worth preserving:
a lookup touches only `hashes`, so its probes pull in only hash bytes.
Interleaved 56-byte records would drag `offsets`, both lengths and the
checksum into cache on every probe, none of which the search reads.

---

## 7. Compatibility

`version` is the discriminator. A v1 reader must reject `version == 2` rather
than attempt to parse it; the array base moved, so a v1 reader that ignored the
version would read the fanout as hashes and silently return wrong offsets.

v1 and v2 are otherwise field-for-field identical, so an upgrade is a rewrite
from `scan_pack()` output with no re-encoding of any chunk — the same path
`recover_packs()` already uses to rebuild a lost index.

---

## 8. Test vectors an implementation must pass

1. **Empty index** (`N == 0`): all fanout entries 0, no L2, file is exactly
   1112 bytes (`1080 + 0 + 32`). Lookup of anything returns NOT_FOUND.
2. **Single entry**: `L1[b] == 1` for all `b >= hash[0]`, 0 below.
3. **Round trip against a linear scan**: for every entry produced by
   `scan_pack()`, `lookup()` returns the identical `(offset, stored_len,
   plain_len, checksum)`. This is the pack layer's central test — an index
   must be checkable against something other than itself.
4. **Both variants agree**: build one index with L2 forced on and one forced
   off over the same entries; every lookup must return identical results.
5. **Trailer**: flipping any single byte in `[0, EOF-32)` must make
   verification fail.
6. **Boundary buckets**: hashes beginning `00 00` and `ff ff` must resolve —
   these are the two indices where the `k == 0` branch and the array end are
   exercised.
