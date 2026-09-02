---
title: SynapseFS

---

# Alignment

## 1. Core Concepts

### 1.1 Groups

A permutation isn't defined per layer or per tensor, it's per **shared ordering**.
When we move a hidden layer's neurons, following should be moved with it following similar permutation:

- the rows of its weight and bias,
- any batch-norm parameters/buffers riding on that same axis,
- the columns of whatever layer reads it next.

The set containing all those elements is a `Group`. 

### 1.2 Pinned axes

The very first axis (network input) and the very last (network output)
cannot be shuffled, so there's no "other side" to reconcile them against. These are marked `pinned` and always solved as identity.

### 1.3 Column blocks

A permutation moves **units**, but a tensor's column count is often a
multiple of unit count, e.g a conv reads `in_ch * kh * kw` columns for
`in_ch` incoming channels. `col_block_size` records how many raw columns
move together as one logical unit, so a permutation can be applied to the
right span of columns instead of one at a time.

---

## 2. The Weight Matching Algorithm

**1. The Cost Matrix**
Build a cost matrix `C` where `C[i, j]` tells how well target unit `i` matches base unit `j`. Two units are considered the same if they play the same *score*. The score is calculated using two terms:

- **Row term**: for tensors whose rows this group owns (a layer's weight,
  bias), compare target's row `i` to base's row `j`. 
- **Column term**: for tensors whose columns this group owns (the next
  layer's weight), same comparison on the other axis.


**2. The Hungarian Algorithm**
`C` is handed to the Hungarian algorithm
(`scipy.linear_sum_assignment(C, maximize=True)`), which returns the
one-to-one pairing that maximizes total similarity. Its output directly *is*
the permutation.

**3. The Coordinate Descent**
It's not possible to get the exact permutation for each group in a single run, So Instead:

1. Start every group at identity.
2. Do a **sweep** (visit groups in a shuffled order); for each,
   rebuild its cost matrix against whatever the others claim, and re-solve.
3. Repeat sweeps. Each round, every group's neighbor information gets a bit
   more accurate.
4. **Stop** the moment one full sweep changes nothing, that's convergence.
5. **Give up** after `MAX_SWEEPS = 10` if it never settles, real fixtures
   converge in ≤5, so an unrelated pair that never converges is capped
   rather than looping forever.
6. A ratio `||Target - Base|| / || Base ||`, if it is less than 0.05 then there is no requirement for the coordinate descent.

---

## 3. Key Design Decisions

| Decision | Why |
|---|---|
| Memory-map | Allows us to open multi-GB files for near zero open cost |
| `THRESHOLD = 0.5` (for alignable/not) | After testing, on a 92 million parameter model, we found unrelated tensors measure ~1.41 (independent samples, doubled variance); real fixtures land far below (<0.03).|
| `RATIO_THRESHOLD = 0.05` | Came to this decision after more than 30 commits on a 92 million parameter model, further checked on a model with width of 10000. |
| `MAX_SWEEPS = 10` | Real convergence happens in ≤5; the cap only bounds wasted work on unrelated pairs |
| `AlignmentResult.solved` checked before `identity` | Distinguishes "identity because it's correct" from "identity because we gave up", the two must never be reported the same way |

## 4. Testing Statistics
 - Time goes as n^2.50 in layer width. Hungarian is faster when the cost matrix is decisive, and a recoverable permutation is exactly that. At n = 10,000 a full alignment is 50.8 s and 3.36 GB at 100% recovery.
 - Recovery survives noise up to half the weight standard deviation, then degrades smoothly, 99.5% at 0.75, 95.3% at 1.0, 80.5% at 1.5. There is no cliff, and is_alignable starts rejecting before the answers get bad.
 - Width and noise multiply. Width 8192 costs 31.8 s at noise 0.01 and 215.5 s at noise 1.0, 6.8× for 2.5× the sweeps, because a less decisive matrix also makes each individual solve slower.

# Compression

## 1. Core Concepts

### 1.1 Bit-exact residuals


To calculate the residuals, we used the following approach:
- Reinterpret each element's bits as an **unsigned integer** of its
native width (2 bytes for F16/BF16, 4 for F32, 8 for I64) and subtract
those, modulo `2ⁿ`. Integer subtraction never rounds and it's a bijection, so
it's exactly reversible: `new_bits = old_bits + delta` which recovers the original
without any approximations.

### 1.2 Byte shuffle 

A weight's residual is one small number occupying the low byte, plus a high
byte, `0x00` if the weight drifted up, `0xff` if it
drifted down.
Drift direction is locally correlated across a trained layer, so those high
bytes come in long runs which is exactly the structure an LZ compressor wants.

To counter the problem of alternation of low and high bytes (`lo hi lo hi …`), **Byte Shuffle** transposes the buffer, keeping the low bytes together and high bytes together


### 1.3 Chunks — the unit of storage

Tensors aren't handled whole; each is cut into **chunks**, whole rows only,
sized to a byte budget (4 MiB by default).
Chunks are the unit of:

- **deduplication**: an unchanged chunk is stored once, referenced twice
- **partial reads**: reading rows 100–200 fetches only the chunks covering
  them
- **memory**: nothing loads a whole multi-GB model into RAM at once

### 1.4 Raw fallback

After encoding a delta, the raw tensor is also encoded, and whichever is
smaller is kept. Two unrelated tensors produce a delta *larger* than the
tensor itself, so after encoding a delta, the raw tensor is also encoded and compared, then the smaller one is stored.

---

## 2. The Residual Codec Algorithm

To turn two checkpoints of the same tensor into the smallest exact
residual (a blob), combined with the old checkpoint, reproduces the new
one bit-for-bit.

**1. Reinterpreting the bits as unsigned integers.**
To ensure that the reconstructed bits are exact, we first map the floating points to unsigned ints while preserving ordering. 

**2. Subtract, mod `2ⁿ`.**
`delta = target_bits - base_bits`. This is exact and fully
reversible as this is a bijection and doesn't round off unlike the float subtraction .

**3. Byte shuffle.**

It transposes the buffer, keeping the low bytes together and high bytes together

On the decoding side, this is done in Cython as the decoding process had the python decoding function (decode_chunk in codec/chunk.py) as the bottleneck.

**4. Compress with zstd, at level 1.**

Empircally found that L1 did better than L2 and abive on both speed and size

**Worked example** (fp16, three elements):

```
  value(target)  bits  |  value(base)     bits  |  delta (mod 2^16)  high byte
       1.0       3c00  |  1.0009765625    3c01   |          65535       0xff
  1.00195312      3c02  |  1.0            3c00   |              2       0x00
       0.5       3800  |  0.4995117188    37fe   |              2       0x00
```

Row 1 wraps to `65535`, i.e. `-1` — the wraparound isn't tolerated noise,
it's the mechanism: one ULP of drift costs one unit, in either direction,
with the sign landing cleanly in the high byte.

Because arithmetic stays in the tensor's native width, the residual stream
is exactly the size of the tensor region — no widening, no varints.

**The bf16 variant.** `bf16` has 7 mantissa bits against fp16's 10, so the
same weight movement lands ~8× fewer ULPs away — median residual falls from
**510 to 27**, and 86.5% of residuals now fit in a single byte (vs 35.3%
for fp16). That makes a second encoding worth it: **zigzag** the signed
delta so both directions are small (`-1→1`, `+1→2`), write one byte per
weight, and escape anything oversized with a `0xFF` marker — with the
escaped values collected in a **separate plane at the end**, not spliced in
inline (inline breaks the run of small bytes the compressor is matching
against; moving them out is worth 2.6 percentage points). This drops bf16
from 60.88% (plain byte shuffle) to **52.72%**. On fp16 the same scheme
loses — only 35% of residuals fit a byte — so the codec falls back to plain
byte shuffle there. Any conclusion about which scheme wins is conditional
on mantissa width; it does not transfer between formats.

**Raw fallback.** Always encode raw alongside the delta and keep whichever
is smaller.

---

## 3. Key Design Decisions

| Decision | Why |
|---|---|
| Bit-integer subtraction| Float subtraction rounds; `(a-b)+b ≠ a` in general. Integer subtraction mod `2ⁿ` is a bijection which is exactly reversible and doesn't round off.|
| Byte shuffle | Only a whole-byte permutation preserves LZ matches; splitting finer mixes multiple elements into one output byte and breaks repeated-value matching entirely |
| zstd at level 1, not a higher level | Compression is non-monotone here: level 1 gets both the best ratio (74.79%) and far higher throughput (398 MB/s) than level 3 or 9, which switch match-finders and lose on run-heavy data |
| Zigzag + escape-byte plane, bf16 only | bf16's narrower mantissa means most residuals fit one byte; zigzag makes negative residuals fit too, taking bf16 from 60.88% to 52.72%. The same scheme *loses* on fp16, where it's correctly left out |
| Per-chunk selection by counting, not by dtype | A value costs 1 byte fitting, 3 escaping — mean `3 − 2p` only beats a 2-byte residual once `p > 0.5`. A dtype rule gets this wrong whenever a bf16 commit sits far from its anchor and the escape fraction climbs |
| 1 MiB default chunk size | 1MiB lead to most cache hits on the mount's LRU cache, and overall minimum RSS on read and mmap benchmarks, while not affecting performance much. |
| Raw fallback always computed | Two unrelated tensors produce a delta larger than the tensor itself; without this, an unrelated pair would cost more to store as a "diff" than as a plain value |

## 4. Testing Statistics

Achieved compressed commit sizes with multiple methods. Tested on 25 checkpoints of a 92M parameter model.

| scheme | fp16 | bf16 |
|---|---|---|
| byte shuffle (the fp16 answer) | **72.56%** | 60.88% |
| zigzag + byte shuffle | 72.48% | 56.79% |
| zigzag + PFor bitmap | 71.65% | 52.77% |
| zigzag + escape byte, wide plane split | 74.32% | **52.72%** |

# Version Control System

## 1. Core Concepts

### 1.1 Content addressing

Every piece of data is hashed (BLAKE3-256) and stored under its own hash
`objects/<h[0:2]>/<h[2:4]>/<h[4:64]>`, one scheme for every object kind,
chunks included. This allows us the following:

- **Deduplication.** Nothing is ever stored twice.
- **Tamper detection.** Re-Hashing the file and finding disagreement in comparing the name allows us to detact if it is changed or not.
- **Chains of trust.** If object A stores object B's hash, verifying A also
  pins down B — B cannot be swapped without A's hash changing. Chain enough
  of these and one trusted hash verifies everything beneath it (§2.4).


### 1.2 Commits form a DAG

Commits point backwards at **parents** — one for a
normal commit, none for the root, two or more for a merge. Because a commit
can have two parents and two commits can share one parent (a branch), the
shape is a **DAG** (Directed Acyclic Graph), directed because
links have a direction, acyclic because a commit's hash depends on its
parents' hashes, so a loop back to where you started is structurally
impossible.


### 1.3 The manifest tree — what a commit actually points at

```
commit  ──►  checkpoint-manifest  ──►  the original file header (stored verbatim)
                    │
                    └──►  one tensor-manifest per tensor
                                 │
                                 └──►  the hashes of that tensor's chunks
```

A **manifest** is just a small JSON file listing what something
is made of. Critically, `content_hash` lives on the tensor-manifest
and is computed from reconstructed data only, never from a chunk hash or a
base reference which is what keeps two identical tensors recognizably
identical even when they were diffed against different history.

### 1.4 Star topology

If commit 40 diffs against 39, which diffs against 38, and so on,
reconstructing 40 means undoing 40 residuals which is slow, and one broken link
breaks everything after it. Instead, every `REBASE_INTERVAL`-th commit stores its checkpoint **in full** and becomes a hub; the commits in between each diff **directly against that hub**.
```
chain:  A(full) <- B-A <- C-B <- D-C      reconstructing D = 3 decodes
star:   A(full) <- B-A                    reconstructing D = 1 decode
        A       <----- C-A
        A       <---------- D-A
```

---

## 2. The Commit Algorithm

**1. Reading the checkpoint's header.**

**2. Decide what to Diff against**
Follow first-parents to the nearest full ancestor (the current hub):

```
anchor      = nearest_full_ancestor(base_hash)
since_full  = commits_since_full(head)
store_full  = anchor is None or since_full >= REBASE_INTERVAL - 1
```

**3. Align.** Work out the neuron correspondence between the new
checkpoint and the anchor wiich is needed before any
diff can mean anything, since two functionally identical checkpoints can
have their neurons in unrelated orders.

**4. Compression.

**5. Hash each result** 
This is where deduplication actually happens — silently, as a side effect of content addressing, not as a separate pass.

**6. write manifests bottom-up.** A tensor-manifest per tensor, then
a checkpoint-manifest referencing all of them, then a commit object
referencing that.

**7. move the branch pointer last, always.**
If the process dies at any earlier step, there are some unreferenced
objects lying around that nobody points at — harmless, cleaned up later. If
the pointer moved *before* the data finished, a branch would point at a
commit whose data was never written — broken and unrecoverable. **The
pointer always moves last**, which is also why every writer in this system
writes to a `.tmp` path and atomically `rename()`s into place: a crash
leaves either the old file or the new one, never a truncated one that
hashes to nothing and poisons the store.

**Verifying a commit (the reverse walk, rooted at trust):**

```
ref  (trusted by assumption)
 └─ commit hash             → re-hash the commit's bytes
     └─ checkpoint_manifest → re-hash
         ├─ header_object   → re-hash
         └─ tensor_manifest → re-hash
             └─ chunks[].object → decompress the payload, re-hash
```

Every comparison is against a hash that came from the object's *parent* —
never a value stored beside the thing it checks (§1.4). There are four
tiers, each catching strictly more:

```
--shallow : object exists + links resolve                     (rot, broken links)
--fast    : + compressed-bytes checksum matches manifest       (rot AND substitution)
default   : + decompressed-stream hash matches manifest         (+ forged checksum, 2^64 work)
--content : + tensor content_hash recomputed from reconstruction (+ wrong-order permutation)
```

Two rules that never change regardless of tier: **verification must never
repair** what it inspects, and the walk must follow **every** parent, not
just the first — skipping a merge's second parent leaves half of history
unchecked.

---

## 3. Exact File Formats

### 3.1 Commit Fomat
            
```        
{
  "checkpoint_manifest": "5e6f...12",
  "parents": ["0a1b...99"],
  "timestamp": "2026-08-25T12:00:00Z",
  "message": "epoch 3 checkpoint"
}
```

### 3.2 Checkpoint Manifest
```
{
  "header_object": "1a2b...cd",
  "tensors": {
    "layer1.bias":   "44de...01",
    "layer1.weight": "9f2c...a1"
  },
  "topology_config_hash": "77aa...bb"
}
```
            
### 3.3 Tensor Manifest

```
{
  "name": "layer1.weight",
  "dtype": "F16",
  "shape": [4096, 4096],
  "base_tensor_manifest": "9f2c...a1",
  "base_row_permutation": "3b71...0e",
  "base_col_permutation": null,
  "col_block_size": 1,
  "chunks": [
    {"row_start": 0,    "row_end": 511,  "encoding": "delta-zigzag-zstd", "object": "aa01...ff", "stored_checksum": "128581ai"},
    {"row_start": 512,  "row_end": 1023, "encoding": "raw-zstd",          "object": "bb02...ee", "stored_checksum": "a102289a"},
  ]
}
```

### 3.4 Chunks

Each chunk is stored as a collection of compressed bytes, stored as one loose object. The byte offsets in the tensor manifest were a legacy from when we used pckfiles.

---

## 4. Key Design Decisions (and why)

| Decision | Why |
|---|---|
| Three distinct hash types (chunk content, stored checksum, tensor content) | Conflating them breaks things silently — e.g. comparing manifest hashes instead of tensor `content_hash` would report a conflict on identical weights simply diffed against different bases |
| Star topology (hub + spokes), not a chain | Reconstructing any commit costs one decode instead of walking back through every intermediate commit; measured 2.87x faster reconstruction for ~1.7 percentage points more storage |
| Branch pointer moved last, writes are `tmp` + atomic rename | A crash mid-commit leaves harmless unreferenced objects, never a branch pointing at incomplete data, and never a half-written file that poisons the store |
| No persistent packfile/pack index; chunks are loose objects | packfiles provided no appreciable performance gain despite the added complexity |
| `stored_checksum` moved from the pack index into the tensor-manifest | Now covered by the manifest→commit→ref hash chain, so an attacker can no longer substitute a chunk and adjust its checksum to match — upgrades a rot scan into a real tamper check, at no extra decompression cost |
| Verification tiers are additive, not a single fixed check | Lets the caller trade cost for guarantee explicitly — `--fast` alone is already sound against any attacker who can't do 2⁶⁴ work, so paying for full reconstruction (`--content`) becomes optional rather than forced |
| Per-tensor anchoring left opt-in, not default | Measured only a 0.79pp storage gain over the flat per-checkpoint policy on real drift patterns — too thin a margin to justify the added reconstruction-depth complexity as the default |

---

# Cryptographic Verification

Every hash is checked against a value stored in the object that *points at* it,
never against a value stored beside it. The chain is anchored at the ref:

```
refs/heads/<branch>  ->  commit  ->  checkpoint-manifest  ->  tensor-manifest  ->  chunk
```

## Tiers

Each tier is a superset of the one above it. `--deep` is the default.

**`--shallow`** — Re-hashes every loose object (commits, manifests, headers,
permutations) against the hash its parent named, and probes that every chunk
referenced by a manifest exists on disk. Reads no chunk payload at all.

*Catches:* structural corruption, broken links, missing objects, a truncated
object graph.
*Misses:* anything about the contents of a chunk.

**`--fast`** — Adds each chunk's stored, still-compressed bytes checked against
the `stored_checksum` its tensor-manifest records. No decompression. Because
that checksum is reached by walking down from the ref, this tier detects
substitution as well as bit-rot.

*Catches:* bit-rot, substituted chunk payloads.
*Misses:* a payload that decompresses to something other than what it claims.

**`--deep`** *(default)* — Adds decompressing every chunk and re-hashing the
plain stream against the hash its tensor-manifest names in `chunks[].object`.

*Catches:* malicious block injection — a payload crafted to pass the checksum
but decode to different weights.
*Misses:* a structurally valid checkpoint whose permutations were applied in the
wrong order.

**`--content`** — Adds reconstructing every tensor and checking it against its
manifest's `content_hash`. This is the only tier that performs *reconstruction*:
a residual chunk must be applied to its base, so it costs more than the tiers
above, which hash stored bytes and stay O(stored bytes).

*Catches:* a permutation applied in the wrong order, or inverted — the one
failure that produces a valid-looking checkpoint of scrambled weights.

**`--all`** — Verifies every branch rather than only `<ref>`'s ancestry

## Cost

25 commits, 6075 chunks, 3.4 GiB of stored objects, cold page cache:

| Tier | Cold | Warm | Payload rate | Peak RSS |
|---|---|---|---|---|
| `--shallow` | 1.10 s | 0.92 s | — (reads no payload) | 85 MB |
| `--fast` | 8.27 s | 8.14 s | 503.7 MiB/s | 86 MB |
| `--deep` *(default)* | 13.54 s | 13.44 s | 293.5 MiB/s | 87 MB |
| `--deep --content` | 32.89 s | 35.90 s | 116.8 MiB/s | 127 MB |

`--content` costs about 2.5× `--deep`.

Cold and warm times are close because the work is CPU-bound on hashing and
decompression rather than I/O-bound.

Memory is flat at ~86 MB across the first three tiers: verification streams
chunk by chunk and never holds a whole checkpoint. It rises only for
`--content`, which materialises one tensor at a time.

Objects and chunks are memoised across the walk, so a 25-commit history verifies
far fewer distinct objects than it references — tensor-manifests are shared
between commits by construction, and chunk dedup shares payloads too.

# FUSE



## 3. Key Design Decisions 

| Decision | Why |
|---|---|
| Cache keyed on the chunk span, not the requested byte/row range | The kernel's 128 KiB reads don't match the 4 MiB chunk unit; keying on the request makes every read a unique miss — measured at 31.6x read amplification versus keying on the chunk |
| Cache bounded by bytes (512 MiB default), not entry count | Entries range from ~200 bytes to 4 MiB; "keep N entries" is not a memory bound, "keep 512 MiB" is one that can actually be checked |
| `read()` hopped to a worker thread (`trio.to_thread.run_sync`) | The daemon's single-threaded event loop would otherwise stall every other in-flight request while one chunk decodes |
| Header bytes replayed verbatim, never re-serialized | Guarantees byte-exact reconstruction — key order and whitespace in the original JSON header survive exactly |
| Element width derived (`byte_span / element_count`), not mapped from dtype name | A dtype this code has never seen still reconstructs correctly |
| `(parent, name) → inode` index for allocation | Without it, allocating a new inode scanned every inode allocated so far — quadratic in commits touched |
| `lookup_count` incremented on every `lookup` reply, decremented by `forget` | The kernel caches lookup answers and holds a reference until it calls `forget`; skipping this leaks one entry per path ever looked up — a slow leak invisible in casual testing |
| `open()` sets `keep_cache=True` | Mounted content is immutable, so the kernel is safe to page-cache it itself |
| This module imports only the checkpoint object, the repo, and its own cache | Never touches the codec or walks the commit graph directly — why it survived unrelated internal migrations with no edits here |
| Double fork + pipe handshake on background mount | The daemon must detach from the shell (a second fork prevents reacquiring a controlling terminal), but the parent still needs to report a real error instead of a false success if mounting failed — the pipe carries that result back |

# Networking

TCP networking for synchronizing objects and branches between two SynapseFS
repositories.

The networking executable, `spp`, provides three operations:

- `serve` — listen for incoming requests.
- `push` — send a local branch to a remote server.
- `pull` — fetch a remote branch.

---

## Commands

### `serve`

```text
spp serve <port> [ro=<0|1>]
```

Starts a TCP server listening on all IPv4 interfaces.

Examples:

```bash
spp serve 9000
spp serve 9000 ro=1
```

The optional `ro` flag controls whether the server accepts pushes:

- `ro=0` — normal read/write server; pushes and pulls are accepted.
- `ro=1` — read-only server; pulls are accepted but pushes are rejected.

The server continues accepting connections until it is terminated.

### `push`

```text
spp push <ip> <port> <branch>
```

Sends the requested local branch to the remote server.

Example:

```bash
spp push 192.168.1.10 9000 main
```

The sender walks the branch's reachable object graph. The receiver reports
which objects it already has, and only missing objects are transferred.

### `pull`

```text
spp pull <ip> <port> <branch>
```

Fetches the requested branch from the remote server.

Example:

```bash
spp pull 192.168.1.10 9000 main
```

After all required objects have been received, the local
`refs/heads/<branch>` reference is updated to the branch tip.

---

## How synchronization works

SynapseFS stores repository data as content-addressed objects. A branch
references a commit, and commits and manifests reference other objects.

When serving a branch, the networking layer recursively walks the reachable
object graph:

```text
branch
└── commit
    ├── checkpoint manifest
    │   ├── header object
    │   ├── topology config
    │   └── tensor manifests
    │       ├── base tensor manifest
    │       ├── row permutation
    │       ├── column permutation
    │       └── chunks
    └── parent commits
        └── ...
```

The exact traversal is implemented by:

- `branch_get_objects()`
- `commit_get_objects()`
- `checkpoint_get_objects()`
- `tensor_get_objects()`

Referenced parents and base objects are followed recursively.

`HashList` maintains an ordered list of objects while preventing duplicates.

### Object negotiation

The sender first sends the hashes of all required objects:

```text
[uint32 object count]
[64-byte hash]
[64-byte hash]
...
```

The receiver responds with one status byte per hash indicating whether that
object is already present.

The sender then transmits only the missing objects.

Consequently, synchronizing a repository that already shares most of its object
graph requires little additional data.

---

## TCP protocol

The protocol uses raw TCP sockets. TCP provides the reliable ordered byte
stream; the application protocol defines the structure of the messages sent
over that stream.

### Connection setup

The client:

1. Connects to the server.
2. Sends the requested operation.
3. Sends the branch name.
4. Receives an acceptance or rejection status.
5. Performs the push or pull protocol.

Operation values are defined in `network_common.hpp`.

### Strings

Strings are transmitted as:

```text
[uint32 length]
[length bytes of string data]
```

The length is transmitted in network byte order.

### Hashes

A `Hash` is transmitted as its 64-character hexadecimal representation,
occupying exactly 64 bytes.

Valid hashes consist only of lowercase hexadecimal characters:

```text
0123456789abcdef
```

### Objects

An object is transmitted as:

```text
[uint32 size]
[size bytes of object data]
```

The size is transmitted in network byte order.

The current implementation buffers an entire object in memory while sending
or receiving it. This is appropriate for the project's current object sizes.

### Errors

The high 16 bits of the 32-bit object-size field are reserved for an error
marker:

```text
0xFFFF0000 | error code
```

This allows the receiver to distinguish an error response from a normal
object length.

The error codes are defined by the `Error` enum in `network_common.hpp`.

---

## Repository layout

The networking code expects the standard SynapseFS layout:

```text
.synapse/
├── objects/
│   ├── <hash prefix>/
│   │   └── ...
│   └── tmp/
└── refs/
    └── heads/
        └── <branch>
```

Objects are located from their hashes. Branch references are stored under
`refs/heads/`.

The networking layer does not define or modify the internal contents of
SynapseFS objects; it transfers them according to the object relationships
understood by the main SynapseFS implementation.

---

## Safe writes

Received objects are not written directly to their final paths.

Instead, the receiver:

1. Creates the destination directories if necessary.
2. Writes the complete object to a temporary file.
3. Closes the file and checks that the write succeeded.
4. Renames the temporary file to the final object path.

The branch reference is updated using the same temporary-file-and-rename
approach.

This prevents an interrupted transfer from leaving a partially written object
or branch reference at its final path.

---

## Validation

### Branch names

Branch names may contain:

- letters,
- digits,
- `_`,
- `-`,
- `/`.

Slashes cannot appear at the beginning or end of a branch name, and consecutive
slashes are rejected.

This validation is important because branch names are used to construct
filesystem paths.

### Hashes

Hashes received from a peer are checked to ensure that they are exactly
64 lowercase hexadecimal characters before being used as object paths.

This prevents malformed hashes from being interpreted as filesystem paths.

**Hash-format validation is not content-integrity verification.**

The networking layer does not currently re-compute the cryptographic hash of
every received object. A peer can therefore send valid-looking hash names
containing incorrect bytes.

If repository integrity must be checked, use the SynapseFS verification
facilities after synchronization.

---

## Read-only mode

A server started with:

```bash
spp serve 9000 ro=1
```

rejects `push` requests before any objects are transferred.

`pull` requests remain available, allowing the server to act as a read-only
source of repository data.

---

## IPv4 and networking requirements

The client currently accepts IPv4 addresses such as:

```text
127.0.0.1
192.168.1.10
```

The networking executable does not currently provide hostname or IPv6 handling
through its command-line interface.

For two machines on a network, ensure that:

- the server is running,
- both sides use the same TCP port,
- the client can reach the server's IPv4 address, and
- any firewall permits the selected port.

---

## Source files

| File | Responsibility |
|---|---|
| `network_common.hpp` | Shared protocol definitions, hashing/path helpers, validation, TCP helpers, and file transfer |
| `push.cpp` | Resolves a branch's object graph and sends required objects |
| `pull.cpp` | Negotiates and receives required objects, then updates the branch |
| `serve.cpp` | Accepts TCP connections and dispatches push/pull requests |
| `spp.cpp` | Command-line interface and client connection setup |

# Why we used averaging in merge?

Git's model: if only one side changed something, take that side; if both changed it, that's a real conflict. We classified tensors by their content hash rather than manifest hash, since two branches can hold identical weights with different manifests just from being diffed against different bases, and comparing manifests would flag conflicts on tensors nobody touched.

That left the actual hard question: what to do when a tensor genuinely changed on both sides.

We considered resolving that layer-by-layer or weight-by-weight, but both break down for the same reason: a single weight or even a whole layer's neuron doesn't mean anything on its own; its meaning only comes from its position relative to everything else, and two independently trained models don't agree on those positions.

Layer-by-layer would produce a broken, internally inconsistent model since it never fixes that misalignment, and per-weight would additionally be unworkable in practice, generating millions of individual conflicts with no sane way to resolve any of them independently.

So we went with the only point where the two sides are actually comparable at all: align the conflicting tensor into the same basis first, which only works on whole rows or columns at once, matching how the alignment groups already operate, and then average, keeping that behind an explicit `--average` flag since averaging still isn't guaranteed to produce a good model, only a deterministic and reproducible one.

## CLI Reference

### Global Conventions & Flags

Global options can appear before or after the subcommand:
- `-C, --repo <path>`: Operate on repository at `<path>` (defaults to searching upward for `.synapse/`).
- `--json`: Output machine-readable JSON on `stdout`.
- `-q, --quiet`: Suppress progress output on `stderr`.
- `-v, --verbose`: Enable detailed tensor-level logs.
- `--no-color`: Disable ANSI color codes.

#### Standard Exit Codes
- `0`: OK / Success
- `1`: General Error
- `2`: Usage / Argument Syntax Error
- `3`: Not a SynapseFS Repository
- `4`: **Integrity Failure** (hash mismatch, corrupted pack, tamper detection)
- `5`: Not Alignable (under `--strict`)
- `6`: Merge Conflict
- `7`: Network / Protocol Failure
- `8`: Mount / FUSE Daemon Failure

---

### Commands

#### `init`
Initializes a new, empty SynapseFS repository.
```bash
synapsefs init [<path>] [--branch <name>]
```

#### `commit`
Ingests a `.safetensors` checkpoint, aligns it against the base commit, computes integer residuals, and writes a new commit object.
```bash
synapsefs commit <model.safetensors> -m "Commit message" \
                 [--config <config.json>] [--base <ref>] \
                 [--no-align] [--chunk-size <bytes>] [--strict]
```
*(Note: `--config` is required on the first root commit to establish the model topology).*

#### `checkout`
Switches branches or reconstructs a commit's checkpoint byte-identically into the working tree.
```bash
synapsefs checkout <branch|commit> [--out <path>] [--no-materialize]
```

#### `branch`
Lists, creates, deletes, or renames repository branches.
```bash
synapsefs branch [-a] [-d <branch>] [-m <old> <new>] [<new-branch>]
```

#### `log`
Displays commit history graph, messages, timestamps, and commit hashes.
```bash
synapsefs log [<ref>] [-n <count>] [--oneline]
```

#### `verify`
Cryptographically verifies DAG integrity and chunk content hashes.
```bash
synapsefs verify [<ref>] [--shallow] [--fast] [--deep] [--content] [--full]
```

#### `merge`
Performs a three-way merge between the current branch and another branch tip.
```bash
synapsefs merge <branch> [-m <msg>] [--no-commit]
```

#### `mount`
Mounts a read-only virtual filesystem exposing commits and branches as virtual `.safetensors` files without writing them to disk.
```bash
synapsefs mount <mountpoint> [--ref <ref>] [--foreground] \
              [--cache-size <bytes>] [--allow-other] [--debug-fuse]
```

#### `unmount`
Unmounts a mounted virtual filesystem, cleanly stopping the background daemon with fallback to `fusermount3 -u`.
```bash
synapsefs unmount <mountpoint>
```

#### `restore`
Directly reconstructs a commit to a destination file and optionally validates it against a reference file.
```bash
synapsefs restore <ref> --out <dest.safetensors> [--reference <ref.safetensors>]
```

---