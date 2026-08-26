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
synapsefs log [<ref>] [-n <count>] [--graph] [--oneline] [--json]
```

Walks first-parent by default from `<ref>` (default `HEAD`); `--graph` shows all parents.

```
$ synapsefs log --oneline -n 3
4d8e2f  epoch 3        2026-08-25T12:00:00Z   41.7 MiB
9f2c1a  epoch 2        2026-08-25T09:14:00Z   39.2 MiB
0a1b99  initial        2026-08-24T22:03:00Z   3.2 GiB (full)
```

---

## 7. `verify`

```
synapsefs verify [<ref>] [--shallow | --deep] [--json]
```

Walks and cryptographically verifies lineage. **Independent of `checkout` and `mount`**
so integrity can be graded standalone — do not make it depend on either.

| Tier | Checks | Catches |
|---|---|---|
| `--shallow` | DAG structure: commit → checkpoint-manifest → tensor-manifests, re-hashed | Structural corruption, broken links |
| *(default)* | Above + per-chunk stored-payload checksum from the pack index. No decompression. | Bit-rot, truncation, damaged pack |
| `--deep` | Above + decompress each chunk, check its full content hash | **Malicious block injection** |

Be accurate about this in output and in the README: the default tier **cannot** detect a
crafted payload carrying a matching stored checksum. Only `--deep` can.

```
$ synapsefs verify
Verifying lineage for 'main' (3 commits)
  commits 3/3   manifests 192/192   chunks 1500/1500
OK  — 3 commits, 1500 chunks, 127.4 MiB verified in 0.31s

$ synapsefs verify --deep
FAIL — chunk aa01f3… in pack-7c2e… : content hash mismatch
       expected aa01f3…  got 3e91b7…
       referenced by tensor-manifest 44de01… ('layer3.weight', rows 512-1023)
```

Exit **0** clean, **4** on any mismatch. `--json` includes `"ok": bool` and a
`"failures"` array with `{object, pack, expected, actual, referenced_by}`.

`OPEN QUESTION` — whether `--deep` becomes the default once measured.

---

## 8. `merge`

```
synapsefs merge <branch> [-m <message>] [--ff-only] [--no-ff]
```

Reconciles two commit DAGs: fast-forward when the target is an ancestor, otherwise a
three-way merge against the common ancestor, producing a commit with two parents.

```
$ synapsefs merge experiment
Fast-forward: main 4d8e2f -> 7a1c90

$ synapsefs merge experiment -m "merge lr sweep"
Merge base: 9f2c1a
[main 8b3d11] merge lr sweep  (parents: 4d8e2f, 7a1c90)
```

Exit **6** on conflict, listing conflicting tensor names on stderr.

`OPEN QUESTION` — conflict semantics when both branches modify the same tensor.
Candidates: (a) always conflict, require `--ours`/`--theirs`; (b) auto-resolve when one
side is unchanged from the merge base. **(b) is the useful default; (a) is the honest
fallback.** Decide before Phase 2.

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
