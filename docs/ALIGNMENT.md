# `synapsefs/align/` — permutation alignment, file by file

Read `SYSTEMS_PRIMER.md` §6 first for the caching ideas. The machine-learning
background is in `PRIMER.md` §3.

**The problem.** A neural network has **permutation symmetry**: renumber a
layer's hidden units, apply the same renumbering to the next layer's input
columns, and you get a bitwise-different checkpoint computing the *identical*
function. Two checkpoints that are "the same model" can therefore share no
bytes at all.

This module recovers that renumbering, so the codec can diff against a
*reordered* base instead of a scrambled one. It is **data-free** — weights
only, no forward passes.

```
config_parser.py   checkpoint -> a chain of layers
  group.py         chain -> PermutationGroups (union-find)
    objective.py   groups -> a cost matrix per group
      lap.py       cost matrix -> one permutation (Hungarian)
        coordinate_descent.py   sweep until nothing changes
          solver.py             fan out per-group answers to per-tensor
            residual.py         is this permutation worth applying?
```

`reader.py` is the input side, `IR.py` the shared vocabulary, `axes.py` the
geometry, `Error.py` the exceptions, `report.py` the CLI rendering.

---

## `Error.py` — 53 lines

Every exception subclasses `SynapseError` so the CLI maps them to exit codes.

| class | exit | meaning |
|---|---|---|
| `AlignError` | — | base |
| `TopologyError` | 2 | the graph is inconsistent (sizes disagree, a block does not divide) |
| `MalformedCheckpoint` | 2 | the file is not readable as safetensors |
| `UnsupportedArchitecture` | 2 | attention, embeddings — out of scope |
| `NotAlignable` | 5 | solved, but the result is not worth storing |

Exit **4** is never raised here. It is reserved exclusively for verification
failures on repo-internal data.

---

## `IR.py` — 195 lines

The frozen vocabulary everything else codes against.

### `LayerKind`
`LINEAR`, `CONV`, `NORM`, `EMBEDDING`, `ATTENTION`, `OTHER`.

### `TensorRef` (frozen)
`name`, `shape`, `dtype`, plus `rows` (`shape[0]`), `cols` (product of the
rest) and `ndim`. **Everything works in a logical 2-D view**: a conv kernel
`[out, in, kh, kw]` is `rows = out`, `cols = in*kh*kw`. That single convention
removes conv-versus-linear branching from the whole module.

### `ColMember`
A tensor whose *column* axis a group permutes, with its `col_block_size`.

### `LayerNode`
One layer: `kind`, `weight`, optional `bias`, `in_group`, `out_group`, and conv
geometry.

### `PermutationGroup`
**The central object.** One permutable axis, shared by everything that touches
it: `id`, `size`, `pinned`, `row_members`, `col_members`.

The subtlety that makes this work: a group spans **two roles**. A hidden group
is the ROW axis of its own weight and bias *and* the COLUMN axis of the next
layer's weight. Permuting it moves both, which is exactly why the function is
preserved.

### `Topology`
Tensors, nodes, groups, `input_groups`, `output_groups`, `unassigned`. Helpers:
`solvable_groups()` (unpinned and non-empty), `pin()`, `group_for_tensor(name,
axis)`, `col_block_size(gid, name)`.

### `validate(topo) -> list[str]`
Returns complaints rather than raising, so a caller can print all of them.
Checks sizes against members, that no tensor's row axis is claimed twice, that
block sizes divide, and that boundary groups are pinned. **Worth running after
any parser change** — most topology bugs are caught here rather than by a wrong
answer later.

---

## `axes.py` — 93 lines

Rows move one at a time. Columns sometimes move in **contiguous blocks**: after
a conv-to-linear flatten, output channel `c` owns columns
`[c*block, (c+1)*block)`. One rule covers every case:

```
block = consumer.cols // producer.size

  linear -> linear   in_features   / in_features  -> 1
  conv   -> conv     in_ch*kh*kw   / in_ch        -> kh*kw
  conv   -> linear   in_ch*H*W     / in_ch        -> H*W
```

- `row_axis_length`, `col_axis_length` — the logical 2-D view.
- `permutes_columns(kind)` — false for `NORM`; a norm is per-channel and has no
  column axis.
- `assert_block_divides`, `col_block_size_for` — **raise rather than defaulting
  to 1.** A silent 1 produces a valid-looking permutation that scrambles a
  flattened feature map.
- `producer_size(node)` — `out_channels` if known, else `weight.rows`.
- `col_block_size_from_nodes` — derives the block and, for a conv,
  cross-checks it against `kh*kw`.
- `spatial_cells`, `block_columns`, `block_count` — helpers.

---

## `config_parser.py` — 308 lines

Checkpoint -> `Topology`. **Shapes are normative; `config.json` only
corroborates.** Everything needed is already in the safetensors header, and
config key names vary by architecture. What config *is* good for is disagreeing
— which means the wrong config was passed.

> The parser must never read `ground_truth.json`. That is the answer key, for
> scoring only.

- `natural_key(name)` — sorts `features.2` before `features.10`. Alphabetical
  does not.
- `Layer` — a parse-time record: `stem`, `kind`, `weight`, `bias`, `buffers`;
  `out_size` is the weight's row count.
- `split_suffix`, `classify` — `.weight`/`.gamma` versus `.bias`/`.beta`;
  4-D is `CONV`, 2-D `LINEAR`, 1-D `NORM`. `num_batches_tracked` is ignored;
  `UNSUPPORTED_HINTS` (`attn`, `embed`, `lora`, …) raise
  `UnsupportedArchitecture` rather than producing a plausible wrong graph.
- `collect(tensors)` — group tensors by stem into `Layer`s.
- `backbone(layers)` — the non-norm layers, which are the chain.
- `check_chain(layers)` — **the trap.** safetensors sorts keys alphabetically,
  so `features.10` precedes `features.2` *and* `classifier.*` precedes
  `features.*`. Natural sort fixes the first, not the second. So the chain is
  type-checked: each layer's column count must be a whole multiple of the
  previous layer's output size. If not, **raise** rather than wire a
  plausible-looking graph that aligns nothing.
- `infer_order(layers)` — recovers the true order from shape divisibility when
  names cannot supply it, via `can_follow(consumer, producer)`.
- `host_of(norm, layers)` — norms attach **by width, not position**.
  `bn1, bn2, conv1, conv2` sorts both norms ahead of every conv, so position
  would put them on the input axis. A norm's width equals its producer's output
  width, so its host is the nearest layer — backward first, then forward —
  producing exactly that many channels. Its parameters *and* `running_mean` /
  `running_var` join that host's group. Those buffers are per output channel
  and **must** move with the permutation; leaving them behind is invisible
  under identity and wrong under every real permutation.
- `to_nodes(layers)` — `Layer` -> `LayerNode` with seed group ids.
- `corroborate(config, layers)` — checks declared `widths`/`hidden_sizes`/
  `channels` are a subset of the widths actually present.
- `parse(tensors, config)` — the entry point: collect, order, check, build,
  validate.
- `from_checkpoint(reader, config)`, `describe(topo)` — convenience and
  debugging.

---

## `group.py` — 175 lines

Layer graph -> `PermutationGroup`s.

The rule: **a node's `out_group` owns its weight's ROW axis; its `in_group`
owns the COLUMN axis.** That is an axis-level statement, not a data-flow one.

### `UnionFind`
`add`, `find` (with path compression), `union`, `members`. Axes that must share
one permutation — a residual stream, a skip connection — are merged **here**,
rather than requiring the parser to guess the right name up front.

### `_resolve`, `_producers`, `_axis_size`
Resolve every seed id to its union-find root; find the unique producer of each
group (raising if two producers disagree on size); determine a group's size
from its producer, or from the consumer's `in_channels` at an input boundary.

### `build_groups(nodes, tensors, shared, pin)`
Creates a group per axis, adds row members (weight, bias, and a norm's
parameters and buffers) and column members with their block sizes, then detects
boundaries **structurally**: an axis produced but never consumed is an output;
consumed but never produced is an input. Both are **pinned to identity** —
permuting the input axis would reorder pixels, permuting the output would
relabel classes. Anything unclaimed lands in `topo.unassigned` and is stored
without a permutation.

### `merged_view`
Debug view of which seed ids merged.

---

## `objective.py` — 300 lines

Builds the cost matrix. **The most important file to understand.**

### The cost
For one group, an `[n, n]` matrix where `C[i, j]` scores target unit `i`
against base unit `j`:

```
C = sum over row members   W_target @ W_base.T
  + sum over col members   W_target.T @ W_base      (blockwise)
```

Both terms are needed, and the reason is precise: **rows alone tie whenever two
units have identical incoming weights; columns alone tie whenever two units are
read identically downstream.** A unit is identified by what feeds it *and* what
consumes it.

### Direction — the convention that must not be broken
> `p[i]` is the **BASE** index that **TARGET** index `i` was diffed against.

`C` is target-major, so scipy's `col_ind` **is** `p` — no inversion anywhere.
Encode and decode apply the *same* gather, `base[p]`. Invert it and you get a
valid bijection of the right length that reconstructs scrambled weights, which
no structural check catches — only `content_hash` does.

**Identity is `None`, never `arange(n)`.** A `None` perm skips the gather
entirely and is written as `null` in the manifest.

### `as_matrix(arr)`
Folds any array to the logical 2-D `[rows, cols]` and widens to float32.

### `DEFAULT_CACHE_BYTES = 1 GiB`, `MatrixPair`
Adapts a pair of readers (or dicts) and **caches the widened matrices** under a
byte cap, LRU.

Widening fp16/bf16 to float32 is the single most expensive thing the solver
does — **51.6%** of an alignment, against 23% for the matmuls and 2.3% for the
Hungarian solve. It was also nearly all wasted: one sweep performed **694
conversions of 144 distinct (tensor, side) pairs**, because each group re-reads
its members, a tensor in two groups is converted twice, and the norm and
residual passes convert everything again.

The cache was previously **off** by default, with a note to enable it "only for
small models" — right about 7B, wrong about everything actually run, and it
left a 4.8x redundancy on the dominant cost. A **cap** rather than a switch
means a 92M model caches everything and a 7B model keeps what fits.

- `_remember(key, mat)` — evicts LRU until it fits; refuses items larger than
  the whole budget (they would evict everything, then be evicted).
- `_fetch(side, src, name)` — hit moves to the end, miss widens and stores.
- `base`, `target`, `scale(name)` — `scale` is `1 / (||target|| * ||base||)`,
  cached forever, since Frobenius norms are permutation-invariant.

### Per-tensor normalisation — why `scale` exists
Each member's contribution is divided by `||W_target|| * ||W_base||`. Without
it the objective is decided by whichever member holds the largest numbers.

A **1-D** member — bias, norm scale, running statistic — contributes
`outer(t, a)`, a **rank-1** matrix whose argmax is the same column for every
row. It carries almost no correspondence information. A 2-D member contributes
a full-rank matrix that does. On `g.features.28` of the 92M benchmark between
consecutive epochs, where identity is provably correct:

| member | shape | ‖contribution‖ | `argmax == i` |
|---|---|---|---|
| `features.28.weight` | (1792, 12096) | 5.75e+02 | **100.0%** |
| `features.29.running_var` | (1792, 1) | 2.17e+06 | 0.1% |

`running_var` holds variances, so its rank-1 term outweighed the kernel by
**3783x** and the solver optimised it instead: 668 of 1792 units moved away
from identity on a pair never permuted. Normalising makes members *comparable*
rather than letting the biggest win — the rank-1 terms are near-flat among
similar-magnitude units, so the full-rank term breaks the ties.

Effect: a fine-tune commit went 4 m 59.7 s -> 23.9 s, and the fine-tune fast
path began firing at all.

### `apply_row_perm`, `apply_col_perm(mat, p, block)`
Reindex rows, or column *blocks*. `None` is identity.

### `group_cost(topo, gid, src, perms)`
Assembles `C`. For each row member, applies the *column* owner's current
permutation to the base first; for each column member, the row owner's. That is
the coupling: every group's cost depends on its neighbours' current answers.

### `assignment_value`, `member_names`, `objective_value`
`objective_value` is the global weight-matching objective with the same
per-tensor scaling, so coordinate descent's monotonicity guarantee holds.

---

## `lap.py` — 136 lines

The assignment step, and the permutation algebra around it.

- `solve(cost, maximize=True)` — `scipy.optimize.linear_sum_assignment`.
  Returns **`None` for identity**, which is what makes convergence-at-sweep-1
  the fast path. Raises `NotAlignable` on non-finite entries.
- `is_identity`, `as_array`, `same` — `None` and `arange(n)` compare equal.
- `invert` — **for tests only.** Needing it on the main path means the
  convention was broken upstream.
- `compose(first, second)` — defined by what a gather does, so the direction
  cannot be misremembered: `A[compose(f, s)] == A[f][s]`.
- `is_permutation` — validates a bijection read from disk.
- `agreement(p, truth, n)` — recovery accuracy, for benchmarks.
- `pack` / `unpack` — packed int32 LE, no header. `pack` refuses a
  non-bijection; `unpack` rejects one.

---

## `coordinate_descent.py` — 138 lines

Algorithm 1. Every group's cost depends on its neighbours and theirs on it, so
**no ordering solves each group once with correct inputs.** Instead: start at
identity, solve one group against whatever the others currently claim, sweep
until a full pass changes nothing. `DEFAULT_MAX_SWEEPS = 25`.

### `DescentResult`
`perms`, `sweeps`, `converged`, `identity`, `changed_per_sweep`,
`solved_per_sweep`, `objective`, `unsolved`, plus `summary()`.

### `neighbours(topo)`
Groups sharing a tensor — moving one changes the other's cost matrix.

### `descend(...)`
Two properties fall out:

**The fine-tune fast path is not a special case.** The paper initialises
`P <- I`; if sweep 1 changes nothing, the checkpoints were never permuted. That
is the ordinary termination condition firing early.

**Monotonicity.** Solving one group maximises exactly the objective terms
involving it, so the global objective cannot decrease.

A dirty set re-solves only groups adjacent to something that moved — an
optimisation, not a different algorithm, and the tests assert it agrees with
the naive version group for group.

> **A sweep is not free.** "Identity detected — fast path" means it converged on
> sweep *1*; that sweep still built every cost matrix and ran every solve. Late
> epochs converge in 1 sweep (~6 s); early ones take 5 (~35 s) and *still* land
> on identity, because early weights move enough to flip a group before it
> settles back.

---

## `residual.py` — 233 lines

Did alignment help, and is a delta worth storing at all?

### `norm(a)`
L2 accumulated in float64 via `einsum(..., dtype=np.float64)` **whatever the
input dtype** — a float32 dot over ten million elements drifts enough to move a
threshold decision.

### `relative_residual(target, base)`
`||target - base|| / ||target||`.

Relative to the **target's own magnitude**, not to the unaligned residual.
Scoring post against pre would give every fine-tune exactly 1.00 and flag the
easiest case in the system as not-alignable. Against `||B||`, a fine-tune
scores ~0.01 and unrelated tensors ~1.41 (√2 — independent samples have twice
the variance).

It used to widen both operands to float64 before subtracting. That bought
nothing — the accumulator is what needs float64, and `norm` already handles it
— and cost two 231 MiB temporaries per call, 144 calls per alignment: **40% of
the runtime**. Removing it took an alignment from 15.2 s to 6.0 s, and changed
the answer by 2e-13 against a threshold of 0.9.

### `improvement`, `is_alignable(post, threshold=0.9)`
Break-even is **1.0**: there the delta is as large as the tensor and storing raw
wins outright. 0.9 leaves margin for framing overhead.

### `Assessment`
`pre`, `post`, `alignable`, `numel`. `helped` is `improvement > 1e-6`.

`numel` exists because `pre`/`post` are *relative* and not comparable between
tensors — see below.

### `group_bit_delta(assessments)` and `group_helped(...)`
Estimated change in **stored bits**, weighted by element count:
`sum(numel * log(post/pre))`, negative meaning smaller.

This must be per **group**, not per tensor, because a permutation *is* per
group. Accepting it for some members and rejecting it for others produces an
incoherent checkpoint: the stored permutation no longer describes a
correspondence. Reconstruction still works — each tensor gathers with whatever
it stored — so nothing catches it except looking.

Getting the weight right is the whole difficulty:

- by **count**, ten BatchNorm buffers outvote the kernel they belong to;
- by **norm**, magnitude decides — `running_var` summed to 199 while a
  21.7-million-element kernel summed to 11.8, so six statistics buffers
  outvoted 50 MiB of weights whose residual had **tripled**, growing a commit
  from 145.8 MB to 153.2 MB;
- by **element count**, the tensors that actually occupy the commit decide.

### `Summary`, `summarize`, `warning_lines`
> `summarize` averages over tensors **unweighted**, so many small well-aligned
> buffers dominate. On the MNIST pair it reports 44.7% removed where the
> element-weighted figure is 13.1%. Use the weighted number for storage claims.

---

## `solver.py` — 300 lines

The front door: two checkpoints in, one `TensorAlignment` per tensor out.
Alignment is solved per **group**; the codec works per **tensor**; this fans
one out to the other.

- `TensorAlignment` — `pi_row`, `pi_col`, `col_block_size`, `alignable`,
  `residual_pre`, `residual_post`.
- `AlignmentResult` — per-tensor alignments plus `not_alignable`,
  `unsolved_groups`, `rejected_groups`, `disabled`, `sweeps`, `converged`,
  `wall_clock_s`, `assessments`.
  - `solved` — false when a group was left at identity because it *could not*
    be solved. Never conflate that with identity being the right answer: one is
    the fast path, the other is alignment silently doing nothing.
  - `disabled` — true under `--no-align`. Identity was **assumed**, not found,
    and the two must not print the same sentence.
- `_open`, `_shapes` — accept paths, readers or dicts.
- `_unusable(topo, base, target)` — groups whose members the base cannot
  supply. Solving them would crash, so they are skipped and reported.
- `_aligned_base(base, a)` — row gather then column gather, the same order the
  codec uses.
- `_members(topo, gids)`, `_measure(...)` — the residual pass.
- `_reject_unhelpful(topo, perms, assessments)` — drops each group's
  permutation that did not pay for itself, via `group_helped`. Attribution is
  joint (a tensor between two permuted groups reflects both), so the caller
  re-measures the affected tensors rather than trusting stale scores.
- `plan(topo, perms, names)` — group answers -> per-tensor row/column
  assignments.
- `align_checkpoints(base, target, topo, ...)` — solve, fan out, measure,
  reject, re-measure. `measure=False` under `--no-align` skips the residual
  pass entirely: 42 s of CPU on a 176 MiB model for a number nothing reads.

---

## `reader.py` — 335 lines

`SafetensorsReader` — mmap-backed, `MAX_HEADER = 100 MiB`. `TensorEntry` gives
`rows`, `cols`, `count`, `nbytes`, `ref`. `tensor()` is a zero-copy view;
`matrix()` is the logical 2-D view; **`as_float()`** widens to float32 — bf16 by
shifting into the high half of a float32, exactly the bits the truncation that
produced it discarded, because numpy has no bf16.

`LayerWindow` is a two-layer sliding cache for streaming a layer pair at a time
without materialising a checkpoint. Not on the current path — `MatrixPair`'s
bounded cache serves the same purpose — but it is the mechanism for
out-of-core alignment on a model too large to widen.

---

## `report.py` — 159 lines

CLI rendering: `header_line`, `status_lines`, `warnings_for`, `detail_lines`,
`render`, `emit`, `alignment_json`. Colour is auto-disabled when stdout is not
a TTY.

`status_lines` distinguishes three states that look alike and are not:
alignment **disabled** (`--no-align`), identity **found**, and groups that
could not be solved.

---

## Things that will bite you

1. **Permutation direction.** `p[i]` is the BASE index for TARGET `i`. Gather
   `base[p]`, never the inverse. Only `content_hash` catches this.
2. **A sweep is not free.** "Identity detected" still costs a full pass.
3. **Never read `ground_truth.json`.** `config.json` is the only metadata the
   aligner may read.
4. **`summarize` is unweighted.** Quote the element-weighted residual for
   anything about storage.
5. **Exit 4 is not yours.** Verification failures only.
6. **BatchNorm buffers must move with their group.** Invisible under identity,
   wrong under every real permutation.

## Known remaining work

- **Early-out on the objective.** Comparing the solved objective against
  identity's would collapse the 5-sweep early-epoch case to 1 sweep — the
  largest remaining win, and unbuilt.
- **GPU cost matrices.** Measured 3.84x on the cost build (`ARCHITECTURE.md`
  §4.6.2), with two caveats: ship fp16 and widen on the device, and leave the
  Hungarian solve on the CPU. torch must stay an optional import.
- **The quadratic gather.** Under a real row permutation,
  `_gather_from_manifest` decodes every base chunk for every delta chunk. It
  does not bite today because every group lands on identity — it will the first
  time a genuinely permuted checkpoint is committed.
