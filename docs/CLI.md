# SynapseFS — CLI Contract

Status: **draft — needs sign-off from all tracks before Phase 1 code lands.**
`OPEN QUESTION` markers are unresolved decisions.

Binary name: `synapsefs`. Every command required by the PS is here:
`init` `commit` `checkout` `branch` `log` `verify` `push` `pull` `serve` `merge`
`mount` `unmount`.

---

## 1. Global conventions

### 1.1 Global flags

Accepted before or after the subcommand.

| Flag | Default | Meaning |
|---|---|---|
| `-C, --repo <path>` | `.` | Operate on the repo at `<path>` (searches upward for `.synapse/`) |
| `--json` | off | Emit machine-readable JSON on stdout instead of human text |
| `-q, --quiet` | off | Suppress progress; errors still go to stderr |
| `-v, --verbose` | off | Per-tensor detail |
| `--no-color` | auto | Disable ANSI (auto-disabled when stdout is not a TTY) |
| `--version` | | Print version, exit 0 |

### 1.2 Streams

- **stdout** — results only. Under `--json`, exactly one JSON document, nothing else.
- **stderr** — progress, warnings, errors. Always.

Progress must never touch stdout; the benchmark harness parses stdout.

### 1.3 Exit codes

| Code | Name | Meaning |
|---|---|---|
| 0 | OK | Success |
| 1 | ERROR | Generic runtime failure |
| 2 | USAGE | Bad arguments, unknown flag, missing operand |
| 3 | NOREPO | Not a SynapseFS repository |
| 4 | **INTEGRITY** | Verification failed — hash mismatch, corrupt object, tamper detected |
| 5 | NOTALIGNABLE | Checkpoints not meaningfully alignable (only with `--strict`; see §3.2) |
| 6 | CONFLICT | Merge conflict requiring human resolution |
| 7 | NETWORK | Transport failure, peer unreachable, protocol error |
| 8 | MOUNT | FUSE mount/unmount failure |

Code **4 is reserved for integrity failures only.** Graders will script against it; do
not reuse it for I/O errors.

### 1.4 Object references

| Form | Example |
|---|---|
| Branch name | `main` |
| Full commit hash | `9f2c1a...` (64 hex) |
| Abbreviated hash | `9f2c1a` (≥ 6 chars, must be unambiguous) |
| `HEAD` | current commit |
| `HEAD~N` | Nth first-parent ancestor |

---

## 2. `init`

```
synapsefs init [<path>] [--branch <name>]
```

Creates `<path>/.synapse/` with `objects/`, `objects/pack/`, `refs/heads/`, `HEAD`.
`<path>` defaults to `.`. `--branch` defaults to `main`. No commit is created.

Fails with **2** if the directory already contains `.synapse/`.

```
$ synapsefs init myrepo
Initialized empty SynapseFS repository in /home/u/myrepo/.synapse (branch: main)
```

---

## 3. `commit`

```
synapsefs commit <checkpoint.safetensors> -m <message>
                 [--config <config.json>] [--base <ref>]
                 [--no-align] [--chunk-size <bytes>] [--strict]
                 [--anchor-policy flat|adaptive]
```

Ingests a checkpoint, aligns it against the base, stores the residual, writes a commit,
advances the current branch.

| Argument | Notes |
|---|---|
| `<checkpoint.safetensors>` | Required. Must exist and parse. |
| `-m, --message <msg>` | Required. |
| `--config <config.json>` | Topology. Defaults to `config.json` beside the checkpoint. Required for the first commit; reused from the base commit afterward if omitted. |
| `--base <ref>` | Base to diff against. Default: current `HEAD`. Ignored on the root commit. |
| `--no-align` | Skip permutation matching; assume identity. Useful for benchmarking (d) in the codec table. |
| `--chunk-size <bytes>` | Override the default chunk size. |
| `--strict` | Exit **5** instead of **0** when a tensor is not alignable. |
| `--anchor-policy flat\|adaptive` | How each tensor picks its diff base. **`flat`** (default): every tensor diffs against the same commit-level anchor, which resets for the whole checkpoint every `REBASE_INTERVAL` commits. **`adaptive`**: each tensor is judged on its own measured drift and may ride a distant anchor, or re-anchor on its own, independently of the others. See ARCHITECTURE.md §4.3.2 — measured 0.79pp smaller on a 24-epoch 90M CNN, at the cost of a second encode pass. |

### 3.1 Output

```
$ synapsefs commit model_e3.safetensors -m "epoch 3"
Aligning against 9f2c1a (16 permutation groups)
  identity permutation detected — fast path
Encoding 64 tensors, 3.2 GiB
  residual: 41.7 MiB (1.29% of original)
  new chunks: 412   deduped: 1088
[main 4d8e2f] epoch 3
```

`--json`:

```json
{
  "commit": "4d8e2f...",
  "branch": "main",
  "base": "9f2c1a...",
  "original_bytes": 3435973836,
  "residual_bytes": 43724800,
  "residual_ratio": 0.0127,
  "tensors": 64,
  "chunks_new": 412,
  "chunks_deduped": 1088,
  "alignment": {
    "groups": 16,
    "identity": true,
    "wall_clock_s": 0.84,
    "not_alignable": []
  }
}
```

`residual_ratio` is the graded metric — emit it here so the harness never has to
compute it.

### 3.2 Not-alignable reporting

Per the PS, non-alignable pairs must be reported explicitly, never silently degraded.
On stderr:

```
warning: 'layer7.weight' not meaningfully alignable (residual norm 0.98 of baseline)
         stored in full (raw-zstd), no delta applied
```

`not_alignable` in the JSON lists the tensor names. Exit is **0** by default (the
commit did succeed) and **5** under `--strict`.

---

## 4. `checkout`

```
synapsefs checkout <branch|commit> [--out <path>] [--no-materialize]
```

Pre-2.23 Git semantics, as the PS specifies — one command, two jobs:

| Argument | Effect |
|---|---|
| `<branch>` | Switch active branch: `HEAD` → `ref: refs/heads/<branch>` |
| `<commit>` | Detached `HEAD` → raw hash |

In both cases the checkpoint is **materialized to the working tree** (the repo root,
under the filename recorded in the commit) — this is the PS's "restores a specific
checkpoint version to the working state."

| Flag | Meaning |
|---|---|
| `--out <path>` | Write the reconstructed checkpoint here instead of the working tree |
| `--no-materialize` | Move `HEAD` only; write no file |

```
$ synapsefs checkout main
Switched to branch 'main'
Materialized model.safetensors (3.2 GiB) in 4.1s

$ synapsefs checkout 9f2c1a --out /tmp/old.safetensors
HEAD is now at 9f2c1a (detached) — epoch 2
Wrote /tmp/old.safetensors (3.2 GiB)
```

**The output of `checkout --out` must be byte-identical to reading the same commit
through the mount.** That is the PS's Consistency Requirement and it is a CI test.

---

## 5. `branch`

```
synapsefs branch                          # list
synapsefs branch <name> [<start-point>]   # create at <start-point>, default HEAD
synapsefs branch -d <name>                # delete
synapsefs branch -m <old> <new>           # rename
```

Creating does **not** switch — use `checkout <name>`, per pre-2.23 semantics.

```
$ synapsefs branch
* main      4d8e2f  epoch 3
  experiment 7a1c90  lr sweep
```

Deleting the current branch fails with **2**.

---

## 6. `log`

```
synapsefs log [<ref>] [-n <count>] [--graph] [--oneline] [--no-size] [--json]
```

Walks first-parent by default from `<ref>` (default `HEAD`); `--graph` shows all parents.

```
$ synapsefs log --oneline -n 3
4d8e2f  epoch 3        2026-08-25T12:00:00Z   41.7 MiB
9f2c1a  epoch 2        2026-08-25T09:14:00Z   39.2 MiB
0a1b99  initial        2026-08-24T22:03:00Z   3.2 GiB (full)
```

The size column is **derived, not recorded** — it is recomputed from the
commit's tensor-manifests and the pack indexes each time (see FORMAT.md §4.6
for why it is not a field on the commit). That costs one JSON read per tensor
per commit listed, which is why `--no-size` exists for deep histories.

---

## 6A. `restore`

```
synapsefs restore [<ref>] [--out <path>] [--compare <checkpoint.safetensors>]
                  [--all] [--strict]
```

Not required by the PS. It exists because `checkout` couples reconstruction to
moving `HEAD`, and two things want the first without the second:

- Pulling epoch 3's weights out to a scratch path to load in torch, leaving the
  repo on whatever commit it was already on.
- **Proving the codec is lossless**, which is what `--compare` is for. It
  streams the commit and a reference file past each other and reports, per
  tensor, how many elements differ and by how many **ULPs** — the integer
  distance between the two values' bit patterns, via the same monotone key the
  residual encoder uses (FORMAT.md §8).

The ULP column is the point. A tolerance answers "is this close enough"; this
codec's claim is stronger — the residual is exact integer arithmetic, so the
answer must be *bit*-equal, and a single differing mantissa bit has to show up
rather than round away. For a correct reconstruction every number in the table
is zero.

```
$ synapsefs restore HEAD --out /tmp/e6.safetensors --compare epoch06.safetensors
Wrote /tmp/e6.safetensors (6.1 MiB, 38 tensors)
Comparing 22adae against epoch06.safetensors
  (no differing tensors)
  38/38 tensors bit-identical
  whole-file comparison: byte-identical
```

Two checks run, not one. The per-tensor table compares the **live commit**
against the reference, so it stays meaningful without `--out` and isolates the
codec from the writer. `whole-file comparison` appears only when `--out` was
given and compares the two files outright, headers included — that is the
property §4 actually requires, and the tensor table cannot see a header
difference.

`--strict` turns a mismatch into exit **4**; without it `restore` is a pure
report, since diffing two genuinely different checkpoints is a legitimate use.

---

## 7. `verify`

```
synapsefs verify [<ref>] [--shallow | --fast | --deep] [--content] [--packs]
                 [--all] [--json]
```

Walks and cryptographically verifies lineage. **Independent of `checkout` and
`mount`** so integrity can be graded standalone — it talks to the object store
and pack set directly, and works on a repo whose FUSE mount is broken or absent.

Walks **every parent**, not first-parent only (PS 2f/2g): a merge's second
parent is reachable history whose corruption is just as fatal. A visited set
keeps a diamond from re-verifying its shared ancestry.

### 7.1 Where trust comes from

PS 2c fixes the root: *"Trust is rooted at a locally accepted commit/ref ID"*,
and a peer presenting an entirely different but self-consistent history is
explicitly out of scope. So the ref is the axiom and everything else is derived
from it:

```
ref  (trusted by assumption)
 └─ commit hash             → re-hash the commit's bytes
     └─ checkpoint_manifest → re-hash
         ├─ header_object   → re-hash
         └─ tensor_manifest → re-hash
             └─ chunks[].object → decompress the payload, re-hash
```

Every comparison is against a hash that came from the object's **parent**. That
single property is what the command is for, and the tempting shortcuts break it:

- Verifying a pack against its own trailer proves the pack is internally
  consistent — which an attacker who rewrote it has already ensured.
- Verifying a chunk against the checksum in the pack index proves the index and
  the pack agree — which the same attacker has also ensured.

Both are worth doing, because rot is the likelier failure and they are cheap.
Neither can detect tampering, because the reference value lives in a file the
attacker controls as fully as the payload.

### 7.2 Tiers

| Tier | Adds | Detects |
|---|---|---|
| `--shallow` | Loose objects re-hashed against the hash their parent named; every chunk reference probed for existence | Structural corruption, broken links |
| `--fast` | + each chunk's stored payload vs the index's 8-byte checksum. No decompression | **Bit-rot only** |
| *(default)* `--deep` | + decompress each chunk, re-hash against `chunks[].object` | **Malicious block injection** |
| `--content` | + reconstruct each tensor, check its manifest `content_hash` | A permutation applied in the wrong order |

**`--deep` is the default**, departing from this document's earlier table which
defaulted to the checksum tier. PS 2b asks specifically that malicious block
injection be rejected, and the checksum tier structurally cannot do it. Measured
on a 25-commit / 152 MiB history the difference is 0.19 s → 0.54 s; that is not
a price worth paying to ship a default that detects no tampering.

`--content` is separate rather than folded into `--deep` because it is the only
check requiring *reconstruction* — a residual chunk must be applied to its base.
Deep hashes the decompressed stream instead, so it needs no base and stays
O(stored bytes). Costs ~4× deep.

`--packs` re-hashes each pack file against its own trailer. Off by default: at
the deep tier every *referenced* byte is already checked against a stronger,
ref-anchored hash, so this only covers framing and unreferenced regions — at the
cost of doubling read volume.

`--all` verifies every branch rather than `<ref>`'s ancestry.

### 7.3 Output

```
$ synapsefs verify
Verifying lineage for 'HEAD' (25 commits) [content]
  commits 25   manifests 950   chunks 950   objects 1002
OK  -- 25 commits, 950 chunks, 129.1 MiB verified in 0.54s (239.0 MiB/s)

$ synapsefs verify --deep
FAIL -- 1 integrity failure(s) in 6 commit(s)
  chunk-content-mismatch: 38d3dea1e1c83835
       expected 38d3dea1e1c83835...  got 7a74c3754ec310a3...
       in pack 2569713655e43323....pack
       referenced by tensor-manifest ccc7f5a7 ('blocks.0.bn1.bias', rows 0-95)
       decompressed bytes do not match the hash the manifest names -- this block was substituted
```

Exit **0** clean, **4** on any mismatch. `--json` includes `"ok": bool` and a
`"failures"` array of `{kind, object, pack, expected, actual, referenced_by,
detail}`.

`chunks` counts references walked; `chunks_distinct` counts what was actually
read. The gap is the memoisation win — dedup plus §4.5 manifest reuse mean one
chunk serves many commits, and it is most of why verification stays fast as
history deepens (950 references → 810 distinct at 25 commits).

Verification **never repairs**. `PackSet` rebuilds a missing index on open by
default; `verify` opens with that disabled, because silently repairing the
artifact under inspection and then reporting OK would make the command
worthless.

Failures are capped at 100 (reported as `truncated`). A corrupt pack yields one
per chunk, and `ok` is already false after the first.

---

## 8. `merge`

```
synapsefs merge <branch> [-m <message>] [--ff-only] [--no-ff] [--average]
```

Three-way merge of two branches. **Implemented.**

Tensors are classified by their tensor-manifest `content_hash` — not by
manifest hash. Two branches can hold byte-identical weights whose manifests
differ because they were diffed against different bases, and comparing
manifests would report a conflict on a tensor nobody touched (FORMAT.md §2.1).

| base vs ours vs theirs | resolution |
|---|---|
| ours == theirs | unchanged |
| ours == base, theirs differs | take theirs |
| theirs == base, ours differs | take ours |
| **all three differ** | **conflict** |
| present on one side only | take that side |

A conflict stops the merge at **exit 6** unless `--average` is given.

### 8.1 `--average`, and why it is opt-in

`--average` resolves conflicting tensors by **averaging them after alignment**:
the incoming side is permuted into the current branch's basis, then the two are
averaged elementwise. Integer buffers (BatchNorm's `num_batches_tracked`) are
not averaged — the mean of two counters is not a counter — and keep ours.

It is behind a flag, and prints its caveats every time, because averaging is
not a neutral operation on weights:

- Two independently-initialised models occupy different basins of the loss
  landscape. Their elementwise mean is near-chance **unless** one is permuted
  into the other's basis first. That is the Git Re-Basin result and the reason
  `align/` exists.
- Even aligned, wide networks merge well and narrow ones retain a real loss
  barrier. Empirical, not guaranteed.
- **SynapseFS cannot tell you the merged model is any good.** It guarantees the
  merge is deterministic, hash-verified and byte-reproducible. Whether the
  result is a *useful* model needs evaluation on data, which a storage system
  does not have.

```
$ synapsefs merge alt
error: 12 tensor(s) changed on both sides and cannot be merged automatically: ...
       re-run with --average to resolve them by averaging after alignment

$ synapsefs merge alt --average -m "merge alt into main"
Merging into 'main' (base bb950e, theirs ec7f1d)
  averaged        12 tensor(s)
  aligned theirs into our basis: 4 groups, identity=False
[main 4bde50] merge alt into main
```

### 8.2 Mechanics

`MergedCheckpoint` implements the same `names()` / `spec()` / `rows()` surface
as `SafetensorsFile` and `CommitCheckpoint`, so `materialize` writes the result
with no new writer and it is byte-exact by the same path everything else uses.

A merge commit is **always stored full**: it matches neither parent, so neither
is a useful base for it. It carries both parents, and `verify` walks both
(PS 2f/2g).

Fast-forward when the current tip is an ancestor of the incoming branch;
`--no-ff` forces a merge commit; `--ff-only` refuses anything else with exit 2.
"Already up to date" covers both `theirs == head` and *theirs is an ancestor of
head* — the state after any previous merge.

---

## 9. `serve`

```
synapsefs serve [--host <addr>] [--port <n>] [--read-only]
```

Starts the sync listener so another instance can `push`/`pull` against this repo.
Defaults: `--host 127.0.0.1`, `--port 9418`. Runs in the foreground; `SIGINT`/`SIGTERM`
shut down cleanly.

```
$ synapsefs serve --host 0.0.0.0 --port 9418
SynapseFS serving /home/u/repo on 0.0.0.0:9418 (read-write)
```

**The PS asks for this invocation to be documented in the README by name** — copy the
address/port/config table there verbatim.

---

## 10. `push` / `pull`

```
synapsefs push <url> [<branch>] [--dry-run]
synapsefs pull <url> [<branch>] [--dry-run]
```

`<url>` is `synapse://host:port`. `<branch>` defaults to the current branch.
`--dry-run` performs negotiation and reports what *would* transfer, transferring nothing
— use it to demonstrate the differential-transfer metric.

```
$ synapsefs push synapse://10.0.0.2:9418 main
Negotiating with 10.0.0.2:9418
  local  main 4d8e2f    remote main 9f2c1a
  objects to send: 418 of 1506  (1088 already present)
Sending transfer pack (41.7 MiB)... done in 2.3s
Remote 'main' updated to 4d8e2f
```

The "N of M (K already present)" line is the differential-transfer evidence. Emit it
always, not only under `--verbose`.

**Resume:** killing either side mid-transfer must leave both recoverable. Re-running the
same command must not re-send already-received objects, and must report the reduced count.

Exit **7** on transport failure, **4** if a received object fails its hash check.

---

## 11. `mount` / `unmount`

```
synapsefs mount <mountpoint> [--ref <branch|commit>] [--foreground]
                [--cache-size <bytes>] [--allow-other] [--debug-fuse]
synapsefs unmount <mountpoint>
```

Read-only POSIX mount. Daemonizes unless `--foreground`.

| Flag | Default | Notes |
|---|---|---|
| `--ref` | all branches | Restrict the namespace to one ref |
| `--cache-size` | `OPEN QUESTION` | Hard cap on the decoded-chunk cache. Drives the peak-RSS metric — must be tunable for the benchmark curve. |
| `--allow-other` | off | Requires `user_allow_other` in `/etc/fuse.conf` |
| `--debug-fuse` | off | FUSE protocol tracing |

Namespace:

```
<mountpoint>/
├── <branch>/<file>.safetensors
└── commits/<hash>/<file>.safetensors
```

```
$ synapsefs mount /mnt/syn
Mounted /home/u/repo at /mnt/syn (read-only, cache 512 MiB)

$ python -c "from safetensors.torch import load_file; load_file('/mnt/syn/main/model.safetensors')"
$ synapsefs unmount /mnt/syn
Unmounted /mnt/syn
```

`unmount` falls back to `fusermount3 -u` if the daemon is unresponsive. Exit **8** on
failure.

---

## 12. Debug commands (not required by the PS)

Keep these — they make the demo and the Q&A much easier — but do not spend time
polishing them.

```
synapsefs cat-object <hash> [--type]      # dump/inspect any object
synapsefs stat <ref>                      # tensor count, sizes, chunk histogram
synapsefs fsck                            # orphan scan, pack/index consistency
synapsefs bench <fixture-dir>             # run the graded benchmark suite
```

---

## 13. Open questions

- [ ] Default `--cache-size` for `mount` (needs the RSS/throughput curve).
- [ ] Merge conflict semantics (§8).
- [ ] Whether `verify --deep` becomes the default (§7).
- [ ] Working-tree filename: recorded per commit, or fixed at `init`?
- [ ] Does `commit` accept a directory of shards (`model-00001-of-00003.safetensors`),
      or single files only? PS fixtures are ≤ 7B, so single-file is probably safe —
      confirm before assuming.
