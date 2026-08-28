# SynapseFS — Team Instructions

One section per aspect. Read your own section fully; skim the others' **"What you can
assume"** and **"What you must not break"** — those are the contracts between you.

Specs: `FileFormat.md` (byte layouts, quick lookup) · `FORMAT.md` (design + rationale) ·
`CLI.md` (command contract) · `PLAN.md` (schedule, ownership).

**Universal rules:**

1. If a spec doesn't say, **ask and update the spec** — don't decide it in code.
   `OPEN QUESTION` markers in `FORMAT.md` are decisions, not suggestions.
2. Commit early and often. Full commit history is graded, and a fresher with three
   commits at the end is a scoring problem regardless of what they actually did.
3. **You will be asked to defend your module with no notes.** A feature you cannot
   explain is scored as absent. Write your own code; use Claude to explain concepts and
   review, not to produce modules you then have to reverse-engineer.
4. No `import torch` outside `tests/`. Peak RSS is a graded metric and torch costs
   hundreds of MB for nothing.

---

## A. Compression / Codec Team

**Owns:** the residual codec — everything between "here are two aligned tensors" and
"here are bytes to store." Chunking, delta encoding, zigzag/varint, zstd, the pack
dictionary, and the benchmark that justifies all of it.

**Files:** `synapsefs/codec/`, `synapsefs/pack/`

### What you can assume

- The alignment team hands you: target tensor `B`, base tensor `A`, and permutations
  π_row / π_col (or `None` for identity). Alignment is already solved — you never run
  a matching algorithm.
- Tensors arrive as `numpy` arrays or raw `bytes` with a dtype string. `BF16` arrives
  as `uint16`; treat it as opaque bits.
- The storage team gives you `put_chunk(content_hash, payload) -> None` and a pack
  writer. You decide *what* the payload is.

### What you must not break

| Invariant | Consequence if broken |
|---|---|
| **Round-trip is bit-exact, always.** No float arithmetic anywhere in encode or decode. | Byte-exactness fails → the core deliverable fails |
| **Content hash covers the pre-compression stream** (§7.1 step 3), never the zstd output | Dedup silently stops working; nothing errors, the residual ratio just quietly degrades |
| **`-0.0` (`0x8000`) and `+0.0` (`0x0000`) stay distinct** | Reconstruction differs from target in a way that only shows on files containing negative zero |
| **NaN/Inf need no special-casing** — do not add any | Special-case code is where the round-trip bug will be |
| **Chunks split on whole rows**, never byte offsets | Row↔chunk correspondence breaks; permuted fixtures fail while fine-tune fixtures pass |
| Manifest JSON is canonical: `separators=(',',':')`, `sort_keys=True`, UTF-8, no trailing newline | Identical content hashes differently; dedup breaks |

### Build order

1. `key()` / `unkey()` for 16-bit dtypes, with a property test: for all 65536 bit
   patterns, `unkey(key(x)) == x`. Run it exhaustively — it's 65k iterations, it's free,
   and it proves the foundation.
2. Delta + zigzag + varint encode/decode, with the same exhaustive round-trip test over
   random chunk pairs.
3. zstd wrap. Then `raw` and `raw-zstd` paths.
4. Pack writer + reader (`FileFormat.md` §5), sealed-and-immutable.
5. Pack index writer + mmap reader (`FileFormat.md` §6). **Binary-search the mmap; do
   not build a dict of hex strings.**
6. Dictionary support, gated behind a flag, default off until benchmarked.

### Deliverable: the codec benchmark

This is the single highest-leverage artifact either team produces — it is simultaneously
the *residual ratio* metric (7%), the README trade-offs section (5%), and your Q&A
material. Produce a table over the fixture set:

| Scheme | Size vs. original | Encode time | Decode time |
|---|---|---|---|
| (a) raw zstd on B | — | — | — |
| (b) XOR(Â, B) + zstd | — | — | — |
| (c) monotone-int delta + zigzag + zstd | — | — | — |
| (d) each of the above **without** alignment | — | — | — |
| (e) (c) + pack dictionary | — | — | — |

(d) is what quantifies the alignment engine's value. Don't skip it — without it you
cannot answer "how much did permutation matching actually buy you?"

### Then close these `OPEN QUESTION`s in `FORMAT.md`

- Default chunk size (target ~1–4 MB post-compression)
- Root-checkpoint encoding: `raw` vs `raw-zstd`
- Dictionary on/off + dictionary size

### Q&A you will be asked

- Why an integer domain instead of subtracting floats?
- Why is the monotone key order-preserving, and why does that matter?
- What happens to NaN? To `-0.0`?
- Why hash uncompressed bytes rather than compressed?
- Why does zstd not give you random access, and what did you do about it?

---

## B. mmap / FUSE Team

**Owns:** the read-only virtual mount. `mount`, `unmount`, and every syscall path that
`safetensors.torch.load_file()` exercises.

**Files:** `synapsefs/fuse/`

**Start here:** `main.py` in the repo root is the pyfuse3 hello-world. Build from it.

### What you can assume

- The codec team gives you `read_tensor_range(manifest, row_start, row_end) -> bytes`.
- The storage team gives you `resolve(commit_ref) -> checkpoint_manifest`.
- `FileFormat.md` §9 is your normative reconstruction recipe. Follow it exactly.

**Unblock yourself on day 1:** implement the whole mount against a *stub* store that
returns hardcoded bytes. Do not wait for Tracks A or B. Swap the real reconstructor in
later — the interface is `read_tensor_range`.

### What you must not break

| Invariant | Consequence if broken |
|---|---|
| **No pre-materialization.** Never write a reconstructed checkpoint to disk on mount. | Explicitly forbidden by the PS; benchmarked from a cold page cache |
| **Read-only.** Reject `O_WRONLY` / `O_RDWR` with `EACCES`. | Scope creep, and write paths are where corruption lives |
| **`getattr().st_size` must be exact** | `mmap` truncates or over-reads; `load_file()` fails in a confusing way |
| **Decode never runs on the trio event loop** | All concurrent readers serialize; you fail the concurrency half of the POSIX metric |
| **Chunk cache has a hard byte cap** | Peak RSS climbs unbounded under load — a graded 7% |

### The pyfuse3 traps, specifically

1. **pyfuse3 is a single-threaded trio event loop.** Any CPU-bound work (zstd decode,
   numpy gather) must go through `await trio.to_thread.run_sync(...)`. If you call
   `zstd.decompress()` directly in `async def read()`, every other reader blocks. This
   is the single most likely reason your concurrency test fails.
2. **`mmap` reaches you as ordinary `read()` calls** from the kernel's page-fault
   handler. There is no separate mmap op to implement — get `read` and `getattr` right
   and mmap works. But it means reads arrive in page-sized units, possibly out of order
   and in parallel.
3. **Return exactly the bytes requested.** Short reads are legal for pipes, not for
   regular files. Returning fewer bytes than `size` when not at EOF produces truncated
   tensors.
4. Set `st_size`, `st_mode`, `st_ino`, `st_nlink`, and the three timestamps on every
   `EntryAttributes`. Missing `st_nlink` on directories confuses some tools.
5. `readdir` must honor `start_id` for resumption; returning everything every call
   makes `ls` loop on large directories.

### Namespace

```
<mountpoint>/
├── <branch-name>/<file>.safetensors        e.g. main/model.safetensors
└── commits/<commit-hash>/<file>.safetensors
```

### Build order

1. Stub mount: static file, correct `getattr` / `lookup` / `opendir` / `readdir` /
   `open` / `read` / `release`. Verify `cat` and `xxd` work.
2. Wire in the real store. Verify `sha256sum` of the mounted file equals the original.
3. `safetensors.torch.load_file(<mount>/main/model.safetensors)` must succeed unmodified.
   Then `transformers.AutoModel.from_pretrained` if a fixture supports it.
4. Thread offload + LRU cache with a configurable cap.
5. Concurrency: N readers × M threads reading overlapping and disjoint ranges,
   asserting byte-identical results. Run it under load, not once.

### Benchmarks you own

Run each from a **cold page cache** (`sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`):

- Sequential read throughput (MB/s) — mount vs. reading the original file directly
- `mmap` random-access latency (µs per page fault)
- Daemon peak RSS during sustained reads, at several cache caps → produce the tradeoff curve
- Throughput with 1, 2, 4, 8 concurrent readers

### The consistency test (PS requirement, make it CI)

```
synapsefs checkout <commit> --out /tmp/a.safetensors
synapsefs mount /mnt/syn && sha256sum /mnt/syn/main/model.safetensors
# must equal sha256sum /tmp/a.safetensors
```

### Q&A you will be asked

- Where does `mmap` actually enter your code?
- What happens with four processes reading the same tensor at once?
- Why is decode on a thread and not in the event loop?
- How do you serve a 4 KB read without decoding the whole tensor?
- Show me that nothing is written to disk on mount.

---

## C. Alignment Team

**Owns:** topology IR, permutation-group resolution, the weight-matching solver,
not-alignable detection.

**Reference:** `rebasin_paper.pdf` §3.2 + Algorithm 1 (`PermutationCoordinateDescent`).
Use **weight matching** — it is the data-free method, and the PS ships no data.
Activation matching (§3.1) and STE (§3.3) both require it and are out.

Key points:

- Algorithm 1's LAP objective has **two** terms, so you need four matrices resident:
  `W_ℓ^A, W_ℓ^B, W_{ℓ+1}^A, W_{ℓ+1}^B`. Slide a two-layer window; never `load_file()`.
- The paper already initializes `P ← I`. Your fast path is therefore **convergence
  detection at iteration 0**: if the first sweep leaves every `P_ℓ = I`, stop.
- Permutation is per **group** (a layer's output axis), shared by `W_ℓ` rows, `bias_ℓ`,
  norm affine params, and `W_{ℓ+1}` columns. Emit one `permutation` object per group.
- Input and output layers are **pinned** to identity.
- conv→linear flatten: a channel permutation moves *blocks* of columns. Emit
  `col_block_size` per `FileFormat.md` §4.3.1. **Test against a permuted CNN fixture** —
  this bug is invisible under identity permutations.
- Not-alignable: compare relative residual norm pre- vs. post-alignment. Below
  threshold ⇒ `base_tensor_manifest: null`, all chunks `raw-zstd`, and **report it on
  the CLI**. The PS grades explicit reporting, not silent degradation.

Deliverable: permutation-recovery accuracy against the ground-truth permuted fixtures,
plus alignment wall-clock at several model sizes.

---

## D. Storage / DAG Team

**Owns:** CAS, atomic writes, crash recovery, commit DAG, branches, merge, `verify`.

- Write protocol is `FORMAT.md` §3, verbatim: temp → fsync → rename → fsync-dir.
  **Refs are always the last write.** Packs before indexes, always.
- Startup: purge `objects/tmp/`, drop any `.pack` with no `.idx`, refuse on a malformed ref.
- `verify` has three tiers (`FORMAT.md` §12). Know which threat each catches — the
  default tier catches bit-rot but **cannot** catch malicious block injection, because
  an attacker rewriting a pack can forge a matching stored-payload checksum. Only
  `--deep` checks the content hash. State this correctly; a judge will probe it.
- Crash test: `SIGKILL` mid-commit, at several points, in a loop. After each, the repo
  must either verify clean or refuse to proceed. Never silently corrupt.
- Tamper tests, one per attack, each named: flip a byte in a pack payload; truncate a
  pack; substitute a chunk with valid framing; corrupt an index; corrupt a manifest.

---

## E. Networking Team

**Owns:** `serve`, `push`, `pull`, have/want negotiation.

- Protocol: client sends ref tips → server computes the commit set → sends object-ID
  list → client filters against what it has → requests missing objects in batches →
  **verifies each hash on receipt before writing** → updates refs last.
- **Build a transfer pack from the want-list.** Do not ship whole storage packs — the
  peer may already hold some of their chunks from another branch, and the graded
  requirement is "only missing blocks are transferred."
- Resumability falls out of content addressing: a partial transfer means some objects
  exist. Test it by killing the client mid-transfer and re-running; assert no object is
  re-fetched and neither peer's history is corrupted.
- Transport is explicitly **not graded** — use `http.server` or raw sockets. Do not
  spend time here. What *is* graded is that the block-diffing is yours and not
  delegated to rsync/rclone.
- Document the `serve` invocation (address/port/config) in the README — the PS asks for
  it by name.
