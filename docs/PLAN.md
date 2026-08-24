# SynapseFS — Build Plan

Deadline: **2 Sep 2026, 22:00** · Today: 24 Aug 2026 · **~9 days**, 8 people (≥3 Y26, ≥3 Y25).

---

## 0. Environment findings (read this first)

| Thing | Status |
|---|---|
| Python 3.14, torch 2.13+cu130, safetensors 0.8, pyfuse3 3.5, trio 0.34, zstandard 0.25, numpy 2.5 | ✅ installed |
| `fusermount3`, gcc, libfuse3 3.10.5 | ✅ present |
| `blake3` (cp314 manylinux wheel exists) | ⬜ `pip install blake3` — do it |
| CPU / RAM | 24 cores / **15 GB total, ~8 GB free** |
| GPU | RTX 4050, **6 GB VRAM** (spec budgets 8 GB — you have less, so CPU/out-of-core is mandatory, which is the right design anyway) |
| **Disk** | **24 GB free on a 93%-full drive** ⚠️ |

**The disk is the real blocker.** One 7B fp16 checkpoint ≈ 14 GB; you need *pairs* plus repo objects. You cannot hold a realistic 7B fixture locally.

Plan for it:
- Develop and test correctness on **50M–500M** synthetic fixtures (fits easily).
- Prove the out-of-core path separately with a **sparse/streamed multi-GB fixture** — generate a large `.safetensors` on the fly, never materialize two copies.
- Free space, or move the object store to an external drive via a configurable `SYNAPSE_DIR`.
- Never commit fixtures (the PS explicitly says script their generation).

Also: **do not import torch in the daemon hot path.** numpy is enough, and *Daemon peak RSS* is a graded metric — torch costs hundreds of MB of RSS for nothing. Keep torch as a test-only dependency for the `safetensors.torch.load_file()` consistency check.

---

## 1. Architecture decisions to lock before anyone writes code

These are the load-bearing choices. Everything else is plumbing. Judges grade *why*, so each of these needs a written justification in the README.

### 1.1 Byte-exactness comes from storing the header verbatim

A `.safetensors` file is:

```
[8 bytes LE: header_len N][N bytes: JSON header][data buffer]
```

The JSON header's key order, whitespace, and `__metadata__` all affect the bytes. **Never re-serialize the header.** Store the raw header blob as its own content-addressed object and reconstruct as `header_bytes ‖ data_region`. Byte-for-byte exactness then falls out for free, and the FUSE layer becomes a pure offset→bytes mapping.

### 1.2 The residual must be integer-domain, not float-domain

Float subtraction can't round-trip exactly. Map fp16/bf16 bit patterns to a **monotone integer key**, then delta:

```
key(x) = bits ^ 0x8000            if sign bit clear
key(x) = ~bits                    if sign bit set     # (uint16 domain for fp16)
delta  = int32(key(B)) - int32(key(A))
```

This is order-preserving, so *numerically small* changes → *small* integers. Then zigzag + bitpack/varint + zstd. Reconstruction is exact by construction (pure integer arithmetic on raw bit patterns — no float rounding ever happens).

**Benchmark this on day 1** against the alternatives, because the table *is* your README trade-off section and your residual-ratio metric:

| Scheme | Notes |
|---|---|
| (a) raw zstd on B | baseline — what Git LFS effectively does |
| (b) XOR(Â, B) + zstd | the obvious idea; works, but wastes bits |
| (c) monotone-int delta + zigzag + zstd | expected winner |
| (d) each of the above **without** permutation alignment | quantifies what the alignment engine buys you |

### 1.3 Chunk on row boundaries; residuals in target order; base stored raw

**Why chunk at all** — not for dedup. A fine-tuning residual is dense (every weight moves), so block dedup buys little on the residual itself. The two real reasons:

1. **Bounded-work random access.** A zstd frame has no random access — you cannot decompress from an offset without decompressing from the frame start. One frame per tensor means a 4 KB page fault costs a full-tensor decode. That directly wrecks *mmap throughput* (8%) and *daemon peak RSS* (7%).
2. **The networking metric grades blocks explicitly** ("which blocks a peer is missing"). Dedup does pay for genuinely untouched tensors — BN stats, frozen embeddings, partial/LoRA-style tuning.

**The permutation interaction.** Chunking does *not* clash with the aligner — different stages. The aligner streams logical tensors via `safe_open`/`get_slice`, emits π, and π rides in the diff artifact (the PS explicitly permits this). But naive row-chunking *does* clash with reconstruction: with multi-row chunks and arbitrary π, target row `i` comes from source row `π(i)` in chunk `π(i)//rows_per_chunk`, so one target chunk can touch up to `rows_per_chunk` source chunks. One-row chunks would fix it and are not viable — a 4096-wide fp16 row is 8 KB, so 7B ≈ 1.7M objects.

**Resolution — asymmetric storage.** Define the residual in the **target's** index space:

```
R[i] = key(B[i]) - key(A[π(i)])       # i indexes B's rows
```

- **Residual chunks are laid down in B's order** and compressed (~1 MB). Reads of B are sequential in the residual.
- **Base chunks are stored raw and mmappable** — still content-addressed, hashable, dedupable, just not compressed.

The π gather then becomes `memcpy` out of the OS page cache instead of N decompressions. Only residuals pay compression cost, which is where the residual-ratio metric is measured anyway.

**Two hard rules for `FORMAT.md`:**
- Chunk on **row boundaries** (a whole number of output neurons), never fixed byte size — byte chunking destroys the row↔chunk correspondence.
- Tensors below ~1 MB are a single chunk; per-object overhead dominates otherwise.

**In practice π is identity for fine-tuning pairs** — the common case the mount serves under load — so the gather is contiguous. The scattered path is the correctness fallback, not the hot path. This is the second job the identity fast path (§1.4) does: it saves solve time *and* keeps reads sequential.

**Known cost:** delta-chain depth. Reconstructing C→B→A walks the chain. Mitigate with periodic re-basing (a full snapshot every N commits) and put the tradeoff in the README — a judge will ask.

### 1.4 Alignment = per-layer linear assignment, with an identity fast path

Weight matching (Git Re-Basin style): find permutation `P` maximizing `⟨W_A, P W_B⟩` per layer, coordinate-descent across layers because `P_l` couples rows of `W_l` to columns of `W_{l+1}`.

- Solver: `scipy.optimize.linear_sum_assignment` (Hungarian) for small dims; greedy/auction for large ones.
- **Identity fast path:** for real fine-tuning pairs the permutation is almost always identity. Score identity first; if its cost is within ε of the best greedy candidate, skip the O(n³) solve entirely. Huge wall-clock win (that's an 8% metric) and easy to justify.
- **Out-of-core is mandatory:** stream layer pairs with `safe_open(...)` + `get_slice(...)`. Never `load_file()` a whole checkpoint.
- **"Not alignable" detection:** compute relative residual norm before vs. after alignment. If alignment doesn't materially reduce it, report `not alignable` explicitly and fall back to raw storage. The PS calls this out specifically — don't skip it.

### 1.5 Permutation groups come from a topology IR, not from hardcoded layer names

`config.json` shapes will vary. Parse into a general IR: a list of layers with type, in/out dims, and an assignment of each tensor axis to a **permutation group**. Rules:

- `W_l` rows ∈ group g; `W_{l+1}` cols ∈ group g; `bias_l`, BN/LN `gamma/beta/running_mean/running_var` ∈ group g.
- Input and output layers are **pinned** (identity) — you can't permute what the outside world sees.
- **conv → linear flatten** is the classic gotcha: a channel permutation maps to *blocks* of the linear layer's columns. Handle it; it will be in the fixtures.

### 1.6 Storage: immutable content-addressed objects + atomic ref updates

Object kinds: `blob` (chunk), `header`, `tensor-manifest`, `checkpoint-manifest`, `commit`. Address by **BLAKE3** (much faster than SHA-256; wheel available).

Crash safety is nearly free if you follow one rule — **objects are immutable, never written in place**:

1. write to `objects/tmp/<rand>` in the same filesystem
2. `fsync(file)`
3. `rename()` to `objects/<hash-prefix>/<hash>` (atomic on the same fs)
4. refs: same write-temp → fsync → rename, then `fsync` the *directory*

On startup: GC orphaned tmp files, or refuse to proceed. This directly answers "recovery behavior after a simulated crash mid-write."

**Verification argument worth making explicitly:** because a commit stores only *residual* blocks, verifying a commit's lineage hashes only the deltas and their ancestors — not the materialized 14 GB model. That's why `verify` is fast at multi-gigabyte scale. Say this in the README; it's the answer to the 10% verification-time metric.

### 1.7 FUSE: read-only, decode off the event loop

- pyfuse3 + trio, read-only mount. Namespace: `/<branch>/<file>.safetensors` and `/commits/<hash>/...`.
- `read(offset, size)` → touched chunks only → decode → slice. Never materialize the whole file (the PS forbids it).
- **pyfuse3 is a single-threaded trio loop** — push zstd decode and any gather into `trio.to_thread.run_sync()` or you'll serialize all concurrent readers and fail the concurrency part of the POSIX metric.
- LRU cache of decoded chunks with a **hard byte cap** (peak RSS is graded). Make the cap a CLI flag so you can show the tradeoff curve in the presentation.
- `mmap` works through FUSE automatically once `read` and `getattr`(st_size) are correct — it's the kernel page-fault path hitting your `read`. Benchmark from a cold page cache (`echo 3 > /proc/sys/vm/drop_caches`).

### 1.8 Networking: have/want negotiation over any transport

Client sends ref tips → server computes the commit set → sends the object-ID list → client filters against what it already has → requests missing objects in batches → **verifies each object's hash on receipt before writing** → only updates refs once everything is present.

Resumability is a free consequence of content addressing: a partial transfer just means some objects exist. Re-running resumes. Refs move last, so an interrupted sync can never corrupt either peer's history.

---

## 2. Build order

### Phase 0 — Day 0 (today/tomorrow). **Blocks everything. Do it first, together.**

1. **Freeze the on-disk format spec** → `docs/FORMAT.md`. Object kinds, manifest JSON schemas, hash algorithm, ref layout, directory layout. One page. Everyone codes against this.
2. **Freeze the CLI contract** → `docs/CLI.md`. Exact arguments and exit codes for `init commit checkout branch log verify push pull serve merge mount unmount`.
3. **Fixture generator** (`tools/gen_fixtures.py`) — the highest-leverage thing you can build. Emits, from a seed:
   - synthetic MLP and ResNet-style CNN checkpoints in `.safetensors` + matching `config.json`
   - a **permuted** variant (known ground-truth permutation) — this is how you measure permutation-recovery accuracy
   - a **fine-tuned** variant (small noise on all weights) — the realistic delta case
   - a **structurally different** variant — the "not alignable" case
   - sizes from ~5M up to a streamed multi-GB one
4. Repo skeleton, `pyproject.toml`, `README.md` stub, `pytest` harness, `git init` + first commit (the PS grades *full commit history* — start committing today).

### Phase 1 — Days 1–3. Three tracks in parallel.

**Track A — Storage & CLI** (owner + 1 Y26)
`init`, object store (put/get/has, atomic writes), chunker, commit DAG, refs/branches, `commit`, `checkout`, `log`, `verify`, `branch`. Crash-injection test (`SIGKILL` mid-commit → repo still verifies or refuses cleanly).

**Track B — Alignment & Codec** (owner + 1 Y26) — *the critical path*
Topology IR + config parser → permutation-group resolver → linear assignment solver + identity fast path → out-of-core streaming → codec benchmark (§1.2) → residual encoder/decoder → not-alignable detection.
**Deliverable by end of Day 3: exact round-trip.** `reconstruct(A, diff) == B` byte-for-byte, verified with `hashlib` on the whole file.

**Track C — FUSE** (owner + 1 Y26)
Read-only mount over a *stubbed* store first (hardcoded bytes) so you're not blocked on Track A. `getattr/lookup/opendir/readdir/open/read/release`. Then swap in the real reconstructor. Then: concurrency, thread offload, LRU cap, mmap test with `safetensors.torch.load_file()`.

> Track C can start immediately — `main.py` already has the pyfuse3 hello-world skeleton to build from.

### Phase 2 — Days 4–6.

- **Merge + branching** (Track A) — three-way / fast-forward over the commit DAG.
- **Networking** (Track A owner + one from C once FUSE stabilizes) — `serve`, `push`, `pull`, have/want, resume-after-interrupt test.
- **Scale-up** (Track B) — prove alignment works out-of-core on the large fixture within the RAM budget. Measure and record peak RSS.
- **Consistency test** — the PS's explicit requirement: `checkout` output and mount read must be byte-identical. Make this a CI test, not a manual check.

### Phase 3 — Days 7–8.

- **Benchmark harness** producing every graded number: residual ratio, alignment wall-clock, permutation-recovery accuracy, mmap throughput (cold cache), daemon peak RSS, verify time. Automate it — you'll rerun it many times.
- **Tamper-detection tests**: flip a byte in a stored object, inject a foreign block, truncate an object → each must be *detected and rejected*, and the test should say which attack it simulates.
- **README + hosted docs.** Architecture, alignment algorithm, storage format, and a **trade-offs section** built from the §1.2 benchmark table.
- **Presentation** (10 min, ≥2 presenters).

### Phase 4 — Day 9 (2 Sep, before 22:00).

Clean-env build test (fresh venv, follow your own README verbatim — this catches the "works on my machine" failure), ZIP matching the repo including `.git`, submit early.

---

## 3. Ownership (8 people)

| # | Track | Owns | Must be able to defend in Q&A |
|---|---|---|---|
| 1 | A | Object store, CAS, atomic writes, crash safety | Why rename() is atomic; why content-addressing gives resumability |
| 2 | A | Commit DAG, branch, merge, log | DAG walk, three-way merge, fast-forward |
| 3 | A | CLI surface, `verify` | Why lineage verify is fast without materializing the model |
| 4 | B | Topology IR, permutation groups | conv→linear flatten; why input/output layers are pinned |
| 5 | B | Assignment solver, out-of-core streaming | Why LAP; identity fast path; why streaming avoids OOM |
| 6 | B | Residual codec + benchmarks | The monotone-int mapping, bit by bit |
| 7 | C | FUSE ops, mmap, concurrency | Why decode is offloaded; how mmap reaches your `read` |
| 8 | C→A | Chunk cache / RSS, then networking | LRU cap tradeoff; have/want protocol |

> **The Y26 rule is a hard constraint.** Three freshers get a *dedicated* Q&A session on implementation specifics, and coached-but-didn't-build is explicitly penalized. Give each Y26 a module they own end-to-end — not "help with" — and have them present that module.

---

## 4. Which Claude model for which work

Switch with `/model` in Claude Code. Rough division:

**Opus 5 — the default; use it for anything where being wrong is expensive.**
- §1 architecture decisions and the format spec
- The alignment algorithm and the permutation-group resolver (real math, easy to get subtly wrong)
- Byte-exactness reasoning: safetensors header handling, offset arithmetic, dtype/endianness edge cases
- Crash-safety and concurrency: fsync ordering, the trio event loop, races between readers
- FUSE performance debugging — the "why is mmap slow / why is RSS climbing" class of problem
- Merge semantics over the commit DAG
- First drafts of the README trade-off arguments (**then rewrite them yourself** — see below)

Use `/fast` on Opus 5 when you're iterating tightly and want lower latency; it's the same model, just faster output. If your client exposes a reasoning-effort setting, keep it high for the items above and drop it for routine work.

**Sonnet 5 — bulk implementation of things that are already well-specified.**
- CLI plumbing (argparse, subcommands, exit codes) once `docs/CLI.md` is frozen
- The network protocol implementation once the have/want design is settled
- Test suites, the benchmark harness, fixture generator
- Straightforward refactors and porting a design you already agreed on

Near-Opus quality on coding, meaningfully cheaper/faster. This should be most of your Phase 1–2 typing.

**Haiku 4.5 — mechanical work.**
- Greps across the repo, "where is X defined", log-format cleanups
- Docstrings, type annotations, small renames
- Parsing benchmark output into tables

**Fable 5 — only if you hit a genuinely hard wall** (e.g. alignment is wrong and nobody can see why after a real attempt). It costs more; it's not the daily driver.

### The important caveat

The PS explicitly says: *"if developers are not able to answer any question on a feature during the presentation, the implementation of that feature shall be considered null and void"* and *"if we find that the explanations are AI-generated and participants are not able to answer questions, heavy penalties will be applied."*

So bias toward using Claude as an **explainer and reviewer**, not a generator:

- Ask it to *explain* linear assignment, FUSE internals, or Merkle DAG design — then write the code yourself.
- Have the module owner write the first version, then ask Claude to review it and argue against it.
- **Have Claude generate adversarial Q&A drills** per module ("ask me 15 hard implementation questions about my residual codec, then grade my answers"). This is the highest-value use of the subscription for this particular competition, and it maps directly onto the two Q&A sessions.
- Write the README trade-offs section in your own words. Judge-facing prose that reads as AI-generated is a scored penalty here.

---

## 5. Scoring-weight sanity check

| Area | Weight | Where it's won |
|---|---|---|
| Alignment & compression | 25% | §1.2 codec + §1.4 identity fast path + out-of-core |
| Filesystem access & memory | 25% | §1.3 chunk-axis choice + thread offload + capped LRU |
| Cryptographic integrity | 20% | §1.6 immutable objects + delta-only verification |
| Networking & CLI | 15% | §1.8 have/want + crash-recovery behavior |
| Docs & presentation | 15% | README trade-offs + Q&A preparedness |

Alignment and filesystem are half the score and they're also the two hardest tracks — that's why they start on Day 1 in parallel and why Track B is the critical path.
