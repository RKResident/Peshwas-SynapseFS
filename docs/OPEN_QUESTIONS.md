# Open questions

Every unresolved decision in SynapseFS, in one place. Nothing here is a task —
tasks live in PLAN.md §2. These are **decisions**, each of which blocks someone.

Rules for this file:

- One line per question, plus *why it matters* and *what would settle it*.
- When a question is answered, delete it from here and write the answer into the
  doc named in "Settled by" — do not leave an answered question sitting here
  marked resolved. This file should only ever shrink or gain new questions.
- If a question turns out to be blocking someone *today*, move it to §1.

Last reviewed: 2026-08-25.

---

## 1. Blocking now — needed before the codec/pack work starts

**1.1 Chunk granularity: row-aligned or fixed-byte?**
Whether a chunk boundary is forced to land on a tensor-row boundary, or chunks
are a flat N bytes over the tensor's raw buffer.
*Why it matters:* it is the single decision the codec API, the pack layout, the
dedup rate, and the mmap read path all sit on top of. Row-aligned makes a future
row permutation reorder whole chunks; fixed-byte gets better dedup on unaligned
tensors and is simpler. Cannot be deferred — the codec's function signature
depends on it.
*Settles:* FORMAT.md §4, FileFormat.md §7.
*Blocks:* codec team, pack team.

**1.2 Default chunk size / rows-per-chunk.**
*Why it matters:* the graded `residual_ratio` metric moves with it, and so does
index size. Also the reason `commit --chunk-size` currently has `default=None`.
*Settles:* FORMAT.md §13, `cli/commands/commit.py`.
*What would settle it:* the codec benchmark (PLAN.md §1.2) over the tiny+small
fixtures, sweeping sizes and plotting residual_ratio vs. index size.

**1.3 Root-checkpoint encoding: `raw` or `raw-zstd`?**
The first commit has no base, so its tensors are stored whole.
*Why it matters:* `raw` is directly mmappable, which is the entire performance
argument for the FUSE read path; `raw-zstd` saves disk, and there is only 24 GB
free locally. These pull in opposite directions.
*Settles:* FORMAT.md §7.
*Blocks:* mmap/FUSE team — they cannot write the fast path until this is fixed.

**1.4 Topology-IR schema.**
*Why it matters:* must be frozen before the permutation-group resolver is
written. Note it is only needed at *commit* time, not at reconstruction time
(FORMAT.md §7.2), which makes it less load-bearing than it first appeared.
*Settles:* a new doc, `docs/TopologyIR.md`.
*Blocks:* alignment team.

---

## 2. Storage and CLI layer

**2.1 stdout/stderr split for multi-line command output.**
CLI.md §1.2 says stdout is results only and stderr is progress, and the
benchmark harness parses stdout. But CLI.md §3.1's worked example for `commit`
shows five lines of which only `[main 4d8e2f] epoch 3` is a result — the rest
(`Aligning against…`, `Encoding…`, `residual:…`, `new chunks:…`) are progress.
`cli/output.py::emit` currently has no stderr channel, so `format_human` sends
all five to stdout.
*Why it matters:* affects every command still to be written, and `-q`
("suppress progress") is meaningless until it is settled.
*Options:* (a) commands write progress to stderr themselves as work proceeds —
matches `-q`'s wording and gives live output; (b) `emit` grows a second string.
*Settles:* CLI.md §1.2, `cli/output.py`.

**2.2 Do commit objects ever get packed?**
`Repo._lookup_hash` resolves an abbreviated hash by globbing
`objects/<hh>/` — loose objects only. If commits can end up inside a packfile,
hash resolution silently stops finding them.
*Why it matters:* breaks `checkout <hash>` and `log` after the first repack.
*Leaning:* commits stay loose forever — they are small and few. If that is the
answer, say so explicitly in FORMAT.md so nobody "optimizes" it later.
*Settles:* FORMAT.md §5.

**2.3 `gc_tmp_dir` still runs on every `ObjectStore` construction.**
**Partially settled.** Option (a) is done: `gc_tmp_dir` now skips files younger
than `TMP_MIN_AGE_SECONDS` (15 min), so a concurrent `ObjectStore` open no
longer destroys an in-flight commit. Packfiles made this urgent — a pack stages
for the length of a whole checkpoint encode, not microseconds.
*What remains:* option (b), moving the GC to an explicit `Repo.open()` startup
step so it runs once per process rather than per `ObjectStore`. Then the age
gate becomes belt-and-braces instead of the load-bearing guard it is today.
*Why it still matters:* the age gate is a heuristic; a genuinely long write
(a 7B checkpoint on slow disk) could in principle outlive it.
*Settles:* `store/repo.py`, `store/objectstore.py`.

**2.4 Does `message` belong in `commit --json`?**
CLI.md §3.1's human example embeds the commit message; its `--json` block does
not list a `message` key. `format_human` currently reads it defensively.
*Settles:* CLI.md §3.1.

**2.5 `HEAD~N` ancestry refs.**
CLI.md §1.4 specifies them; `Repo.resolve_ref` does not implement them.
*Why it matters:* needs a commit parent graph, which does not exist yet. Not
blocking — listed so it is not forgotten when the graph lands.
*Settles:* `store/repo.py`.

---

## 3. Deferred — answer with measurements, not opinions

**3.1 Pack dictionary on/off, and dictionary size.**
Measure on real fixtures. `zstandard.train_dictionary` is verified working and
deterministic; gave ~5% on synthetic samples, which is not evidence.
*Settles:* FORMAT.md §13.

**3.2 Dynamic re-basing trigger.**
The topology is settled — commits form a **star**: every residual diffs
directly against its group's full checkpoint, and every 4th commit starts a
new one (FORMAT.md §12A). Reconstruction is two decodes regardless of history
length. What remains is replacing the fixed count with a response to the data:
start a new hub when the group's residuals stop being cheap.
*Why it matters:* N now bounds how far a group drifts from its hub. Fixed at 4
it re-bases too often on a stable run (paying a full checkpoint for nothing)
and too rarely across a sharp change (paying inflated residuals); both cost
graded storage or wall-clock.
*What would settle it:* the same measurements as 3.3 — the statistic is
probably residual-ratio degradation or the per-commit `raw-zstd` fallback
rate, both of which the encoder already computes.
*Settles:* FORMAT.md §12A.

**3.3 Not-alignable threshold.**
Measure against the non-alignable fixture. Drives exit code 5 / `--strict`.
*Settles:* FORMAT.md §13.

**3.4 Repack trigger, and bloom filters if pack count exceeds ~32.**
*Settles:* FORMAT.md §13.

**3.5 Should `verify --deep` be the default?**
*Settles:* CLI.md §8, FORMAT.md §12.
