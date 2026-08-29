# SynapseFS — Start Here

**Read this before `ARCHITECTURE.md`.** This explains what the project is, why
it works the way it does, and every term the other documents assume you know.
No prior knowledge of the codebase needed.

---

## 1. The problem

When you train a neural network, you save the weights every so often — an
**epoch checkpoint**. After a week of training you have 40 of them. Each is a
few gigabytes. Together they are half a terabyte, and every one of them is
*almost identical* to the one before it.

You would like to do with them what you do with source code: keep a history,
branch, go back to "the model from Tuesday", and be sure nothing has been
corrupted or tampered with in between.

So: **Git, but for model weights.** That is the whole project.

### 1.1 Why you cannot just use Git

Try `git add model.safetensors` on a 3 GB file 40 times and you get a 120 GB
repository. Git *does* have delta compression — it stores similar objects as
differences — but it fails badly here for two reasons.

**First, Git's diffing is byte-oriented and weights are floating-point.**
Suppose a weight changes from `0.31459` to `0.31460`. Those two numbers share
no meaningful byte pattern — the underlying bytes might be `3E A1 04 7B` and
`3E A1 04 9C`. Git's delta algorithm looks for *repeated byte sequences*, which
works beautifully for text (a line you didn't edit is a long identical run) and
falls apart for float arrays, where every single number nudged slightly means
no long runs survive anywhere.

**Second, and stranger: two models that behave identically can look completely
different.** More on this in §3 — it is the part of the problem that makes this
an interesting project rather than a plumbing exercise.

### 1.2 What we have to deliver

The competition problem statement (the "PS") asks for five things:

| # | module | weight | what it means |
|---|---|---|---|
| 1 | Alignment & compression | 25% | store checkpoints small, handling permuted models |
| 2 | Filesystem access | 25% | expose stored checkpoints as real files a program can open |
| 3 | Cryptographic integrity | 20% | prove nothing has been corrupted or tampered with |
| 4 | Networking & CLI | 15% | the commands, plus push/pull between machines |
| 5 | Documentation | 15% | explain all of it |

And one hard rule underneath all of it:

> **A checkpoint you get back must be byte-for-byte identical to the one you
> put in.** Not "numerically close". Identical.

That rule is why so much of the design looks paranoid. Every shortcut that
would be fine for a lossy system is closed off.

---

## 2. Background you need

Skip any section you already know.

### 2.1 What a checkpoint actually is

A model is a big pile of **tensors** — multi-dimensional arrays of numbers. A
convolution layer has a weight tensor shaped `[out_channels, in_channels,
height, width]`; a linear layer has `[out_features, in_features]`.

We store them in **safetensors** format, which is refreshingly simple:

```
[ 8 bytes: how long the header is ]
[ the header: a JSON object                                   ]
[ all the tensor numbers, back to back, no separators         ]
```

The JSON header maps each tensor's name to its type, its shape, and **where its
bytes start and end** in the data region:

```json
{"layer1.weight": {"dtype": "F16", "shape": [96, 3, 3, 3],
                   "data_offsets": [0, 5184]}}
```

So reading tensor `layer1.weight` means: parse the header, look up
`data_offsets`, read bytes 0–5184. That is the entire format.

### 2.2 Floating-point numbers, and `dtype`

A `dtype` says how each number is stored. We care about three:

| dtype | bits per number | notes |
|---|---|---|
| `F16` (fp16) | 16 | half precision — the common choice for saved models |
| `BF16` | 16 | "brain float" — same size, different split between exponent and mantissa |
| `I64` | 64 | a 64-bit integer, not a float |

Why is an *integer* type in a list of float formats? Because when you convert a
model to fp16 with `model.half()`, PyTorch converts the floats and **leaves
BatchNorm's `num_batches_tracked` counter as a 64-bit integer**. So every real
fp16 checkpoint is secretly a mixed-type file. A codec that assumes "one type,
16 bits" breaks on the first real model it sees. This is the kind of detail
that only shows up when you try it.

### 2.3 Hashing, and "content addressing"

A **hash function** takes any amount of data and produces a short fixed-size
fingerprint — we use BLAKE3, which produces 32 bytes, written as 64 hex
characters:

```
blake3("hello") = 5c17ec5f8b1a1b2c…
```

Two properties matter:

1. The same input always gives the same fingerprint.
2. You cannot find a *different* input with the same fingerprint. (Not
   "it's unlikely" — it is computationally out of reach.)

**Content addressing** means: store a piece of data in a file *named after its
own hash*. Data whose hash is `abc123…` lives at `objects/ab/c1/23…`.

This one trick gives you three things for free:

- **Deduplication.** Store the same bytes twice and the second write finds the
  file already there. Nothing is stored twice, ever, automatically.
- **Tamper detection.** Re-hash a file and compare to its name. If they
  disagree, someone changed it.
- **Chains of trust.** If object A contains the hash of object B, then
  verifying A also pins down B — B cannot be swapped out without A changing.
  Build a chain of those and one trusted starting hash verifies everything
  beneath it. That is §5.

### 2.4 Commits and DAGs

Same idea as Git. A **commit** is a small record saying "here is a snapshot,
and here is the commit that came before me". Commits point *backwards* at their
**parents**.

Because a commit can have two parents (a merge), and two commits can share one
parent (a branch), the shape is not a line but a **DAG** — Directed Acyclic
Graph. "Directed" = the links have a direction. "Acyclic" = you can never loop
back to where you started, since a commit's hash depends on its parents' hashes
and you cannot make that circular.

A **branch** is just a name pointing at one commit. **HEAD** is a pointer
saying which commit you are currently on.

---

## 3. The interesting problem: permutation

Here is the thing that makes checkpoint versioning genuinely different from
file versioning.

Take a trained network. Pick a hidden layer. **Renumber its neurons** — swap
neuron 5 and neuron 12 — and correspondingly swap the weights feeding into them
and the weights reading out of them.

The network now computes **exactly the same function**. Same outputs, same
accuracy, same everything. But every weight matrix has had its rows or columns
reordered, so a byte-level comparison says *100% of the file changed*.

```
Model A:  neurons  [n0 n1 n2 n3]      Model B:  neurons  [n2 n0 n3 n1]
          weights   w0 w1 w2 w3                 weights   w2 w0 w3 w1

          identical behaviour   ·   zero bytes in common
```

This happens for real. Train the same architecture twice from different random
seeds and you get two models that may be functionally close but whose neurons
are in unrelated orders. Merge two teammates' fine-tunes and you hit it
immediately.

**So before we can diff two checkpoints, we have to figure out which neuron in
model A corresponds to which neuron in model B.** That is the *alignment*
problem, and it is module 1 of the PS.

### 3.1 How alignment works, roughly

Two neurons "correspond" if they have similar incoming weights *and* are read
similarly by the next layer. So for each layer we build a score table:

> `score[i][j]` = how well does target neuron *i* match base neuron *j*?

Then we need to pick one match per neuron, no two neurons sharing a partner,
maximising the total score. That is a classic problem called the **linear
assignment problem** (LAP), and `scipy` solves it in one call.

The complication: each layer's answer depends on its neighbours' answers, which
depend on it. There is no order that gets everyone right first time. So we do
**coordinate descent** — a fancy name for a simple loop:

```
start with everyone unpermuted
repeat:
    for each layer: solve its assignment, assuming the others are correct
until a full pass changes nothing
```

A pleasant consequence: if the two checkpoints were *never* permuted — which is
exactly the case when they are consecutive epochs of one training run — the
first pass changes nothing and we stop immediately. The common case is the fast
case, without any special handling.

The output is a **permutation** for each layer: an array where `p[i] = j` means
"target neuron *i* corresponds to base neuron *j*".

---

## 4. Making a checkpoint small

Now that tensors line up, we can store differences instead of copies.

### 4.1 Why you cannot just subtract floats

The obvious idea is `difference = new - old` in floating point. This is wrong,
and the reason is worth understanding because the fix is the cleverest part of
the codebase.

Floating-point subtraction **loses information**. `(a - b) + b` does not
reliably give you back `a` — the result gets rounded. For a system whose whole
promise is *byte-for-byte identical*, "almost `a`" is a failure.

### 4.2 The fix: treat the bits as integers

A float in memory is just a pattern of bits. If we reinterpret those bits as an
*integer* and subtract **those**, the subtraction is exact and perfectly
reversible — integer arithmetic never rounds. Numbers wrap around instead of
losing precision, and wrapping is undone by adding back.

That's the whole trick. `delta = new_bits - old_bits`, and later
`new_bits = old_bits + delta`. Nothing is approximated at any point.

### 4.3 What those differences look like

This matters, because the next step is designed around it.

A weight is 16 bits: two bytes. When a weight drifts a little, the difference
is a small number, so it only occupies the **low** byte. What ends up in the
**high** byte is just the sign:

- drifted **up** → the difference is a small positive number → high byte `00`
- drifted **down** → the difference is a small negative number, which in binary
  means all the high bits are set → high byte `ff`

About two thirds of weights drift by less than a low byte can hold, so their
high byte is exactly `00` or `ff` — two values.

And it gets better: neighbouring weights in a layer tend to drift *the same
direction* during training. So the high bytes don't just take two values, they
come in long **runs** of `00 00 00 …` or `ff ff ff …`.

That's a lot of structure for a compressor to exploit. There's just one problem
with how it's laid out.

### 4.4 A worked example

Two fp16 weights, one from each checkpoint:

```
   value       raw bits       difference   high byte
     1.0           3c00
                                  65535        ff       (i.e. -1)
     1.0009766     3c01
```

One step of drift becomes `-1`, stored as `65535` because unsigned numbers wrap.
Do this for every weight and you get a long stream of mostly-tiny numbers, each
carrying its drift direction in its high byte.

### 4.5 Byte shuffling

The problem is the **order** the bytes sit in. Each weight contributes a low
byte (noise) and a high byte (the useful run), and they alternate:

```
lo hi lo hi lo hi lo hi …
```

Every run of high bytes is chopped up by noise sitting between them, so the
compressor never sees a run at all.

**Shuffle** transposes them — all the low bytes together, then all the high
bytes together:

```
before:  A0 B0 A1 B1 A2 B2
after:   A0 A1 A2 B0 B1 B2
```

Now the runs are contiguous and the compressor can see them. Measured on our
real checkpoints: **78% → 72%** of original size, for six lines of code.

#### Two steps this replaced

The codec used to have two more transforms before compressing. Both were
measured and removed, and the reason is the same in each case: **they helped
before the shuffle existed, and hurt once it was there.**

- **zigzag** mapped small negative numbers onto small positive ones. But it
  works by replacing the *sign* in the high byte with a *magnitude*, and
  magnitude varies weight to weight where sign does not — so it shattered the
  very runs shuffle depends on. Cost 0.6–0.8%.
- the **monotone key** was a bit fix-up that made negative floats sort in the
  right order (they're stored "sign and magnitude", so their raw bits run
  backwards). It genuinely helps for the 3–5% of weights that cross zero, but
  costs more than it saves on everything else. Net −0.2%.

Several other ideas were tried and measured worse: splitting into *bit* planes
instead of byte planes, separating the float's sign/exponent/mantissa fields,
and reordering the tensor diagonally. One rule explains all of them:

> **The rearrangement has to be a permutation of whole bytes.** Split finer than
> a byte and each output byte ends up mixing several different weights, so
> repeated weights stop producing repeated bytes and the compressor loses its
> matches. Pad the pieces out to whole bytes instead and you've doubled the
> data, which costs more than the tidier layout saves.

### 4.6 Then compress

Finally we run **zstd** (a standard compressor) at level 1 over the result. The end
product is a **residual**: a compressed blob that, combined with the older
checkpoint, reproduces the newer one exactly.

### 4.7 Chunks

We don't do this a whole tensor at a time. Each tensor is cut into **chunks** —
a chunk is a contiguous group of rows, about 4 MB by default.

Chunks are the unit of everything:

- **deduplication** — a chunk that didn't change is stored once and referenced
  twice
- **partial reads** — to read rows 100–200 you fetch only the chunks covering
  them, not the whole tensor
- **memory** — nothing ever loads a whole 3 GB model into RAM

---

## 5. Not storing the same thing twice, and knowing nothing was tampered with

Every piece of data gets hashed and stored under its hash (§2.3). A commit is
then a small tree of hashes:

```
commit  ──►  checkpoint-manifest  ──►  the original file header (stored verbatim)
                     │
                     └──►  one tensor-manifest per tensor
                                  │
                                  └──►  the hashes of that tensor's chunks
```

A **manifest** is just an index — a small JSON file listing what something is
made of. A tensor-manifest says "this tensor is F16, shape [96, 3, 3, 3], made
of these 3 chunks, diffed against that other tensor-manifest".

### 5.1 The chain of trust

Notice what that structure gives you. If you trust one commit hash, then:

- you can re-hash the commit and confirm it wasn't changed
- the commit names the checkpoint-manifest's hash, so you can check that
- which names each tensor-manifest's hash, so you can check those
- which name each chunk's hash, so you can check every byte of data

**One trusted hash at the top verifies everything underneath.** That is what
"cryptographic integrity" means here, and it is 20% of the grade.

The critical rule, easy to get wrong: **always check a thing against the hash
its parent gave you** — never against a hash stored next to the thing itself.
If an attacker replaces a chunk, they will happily also update any checksum
stored beside it. They cannot update the commit hash, because that would change
the commit, which you trust.

### 5.2 Storing every 4th checkpoint whole

If commit 40 is a difference against commit 39, which is a difference against
38, and so on, reading commit 40 means undoing 40 differences. Slow, and one
corrupt link breaks everything after it.

So every 4th commit stores its checkpoint **in full**, and the three in between
each diff **directly against that full one** — not against each other.

```
chain:  A ◄── B ◄── C ◄── D          reading D = undo 3 differences
star:   A ◄── B                      reading D = undo 1 difference
        A ◄────── C
        A ◄────────── D
```

We call this a **star** (one hub, spokes off it) rather than a chain. It costs
about 9% more space and reads about 2× faster.

---

## 6. Giving the files back

Two ways to get a checkpoint out.

**`checkout`** writes a real file to disk. It reads the stored header verbatim,
then rebuilds each tensor chunk by chunk and streams it out.

**The mount** (module 2, not yet built) is the more interesting one. The PS
wants a **virtual filesystem**: a directory that *looks* like it contains
`model.safetensors`, so that `torch.load_file("mount/model.safetensors")` just
works — but where no such file exists on disk. When a program reads bytes from
it, we reconstruct exactly those bytes on the spot.

The usual tool for this is **FUSE** ("Filesystem in Userspace"), a Linux
mechanism that lets an ordinary program answer filesystem requests. Your
program registers "when someone reads this file, call me", and the kernel does.

The point is that a 3 GB checkpoint can be *used* without ever existing as 3 GB
on disk.

---

## 7. How the pieces fit

```
   your checkpoint file
            │
            ▼
     [ align ]        which neuron matches which?          (module 1)
            │
            ▼
     [ codec ]        subtract · byte shuffle · zstd   (module 1)
            │
            ▼
   [ chunk store ]    save each chunk under its hash
            │
            ▼
     [ graph ]        write the manifests and the commit
            │
            ├──► [ verify ]      re-hash everything          (module 3)
            ├──► [ checkout ]    write a real file back
            └──► [ FUSE ]        serve it as a virtual file  (module 2)
```

Each layer knows only about the one below it. The codec never touches the
filesystem; the chunk store never knows what a tensor is.

### 7.1 What a commit does, start to finish

1. Open the new checkpoint, read its header.
2. Find what to diff against — the nearest "full" commit (§5.2).
3. Align: work out the neuron correspondence (§3).
4. For each tensor, for each chunk: subtract, byte shuffle, compress.
5. Hash each result; if that hash already exists on disk, skip writing it.
6. Write a tensor-manifest per tensor, then a checkpoint-manifest, then a
   commit.
7. **Last of all**, move the branch pointer to the new commit.

Step 7 being last is deliberate. If the machine dies at any earlier point, you
have some unreferenced files lying around that nobody points at — harmless. If
the pointer moved first, you would have a branch pointing at a commit whose
data was never finished — broken. **The pointer always moves last.**

---

## 8. Glossary

Terms `ARCHITECTURE.md` uses without defining.

| term | meaning |
|---|---|
| **tensor** | a multi-dimensional array of numbers; one weight matrix |
| **checkpoint** | one saved model — a file full of tensors |
| **safetensors** | the file format: 8-byte length, JSON header, raw numbers |
| **dtype** | how each number is stored (`F16`, `BF16`, `I64`) |
| **chunk** | a contiguous group of rows from one tensor; ~4 MB; the unit of storage |
| **residual / delta** | the compressed difference between two versions of a chunk |
| **base** | the older thing a residual is measured against |
| **manifest** | a small JSON index listing what something is made of |
| **object** | any content-addressed file: a commit, a manifest, a header, a chunk |
| **content-addressed** | stored in a file named after its own hash |
| **BLAKE3** | our hash function; 32 bytes, written as 64 hex characters |
| **dedup** | storing identical data once, which content addressing gives for free |
| **commit** | a record: this snapshot, and its parent(s) |
| **DAG** | the branching, merging shape the commits form |
| **HEAD** | pointer to the commit you are currently on |
| **branch** | a name pointing at a commit |
| **ref** | any name that resolves to a commit (`HEAD`, a branch, a hash) |
| **star topology** | every 4th commit stored in full; the others diff against it |
| **permutation** | a reordering of a layer's neurons; `p[i] = j` means target *i* ↔ base *j* |
| **identity permutation** | "no reordering"; written as `null`, not `[0,1,2,…]` |
| **permutation group** | one set of neurons that must be reordered together |
| **LAP** | linear assignment problem — pick the best one-to-one matching |
| **coordinate descent** | solve one layer at a time, repeat until nothing changes |
| **byte shuffle** | regrouping bytes so similar ones sit together (§4.5) |
| **zstd** | the compressor we run last |
| **ULP** | "unit in the last place" — the gap between adjacent representable floats; the natural unit for "how far apart are these two numbers really" |
| **mmap** | mapping a file into memory so you can read parts of it without loading it all |
| **FUSE** | Linux mechanism for writing a filesystem as an ordinary program |
| **atomic write** | write to a temp file, then rename; a reader sees the old file or the new one, never half of one |
| **fsync** | force data to actually reach the disk rather than sitting in a cache |
| **trust chain / anchored** | checking each hash against the value its *parent* recorded |
| **byte-exact** | the reconstructed file is identical to the original, byte for byte |

---

## 9. Where things stand

| module | grade | state |
|---|---|---|
| 1 · Alignment & compression | 25% | compression **done**; alignment **done and connected** |
| 2 · Filesystem (FUSE) | 25% | **not started** |
| 3 · Cryptographic integrity | 20% | **done** |
| 4 · Networking & CLI | 15% | CLI done except `merge`; `push`/`pull`/`serve` live outside this tree and are not merged in yet |
| 5 · Documentation | 15% | this, `ARCHITECTURE.md`, `FORMAT.md`, `CLI.md`; **no README yet** |

Some real numbers from the current code, on a 25-epoch training run of a 3.2M
parameter CNN:

- a checkpoint stores at about **74–85%** of its original size
- reconstruction is **byte-identical**, verified on every commit
- full integrity verification of the 25-commit history takes **0.54 seconds**
- deliberately corrupting one stored block is detected, and the report names
  the exact tensor and rows affected

That 76–84% is honest but unexciting, and worth understanding: consecutive
epochs of Adam training differ in *almost every weight* by a small amount, so
there is little to deduplicate. Measured, the differences carry about 10.9 bits
of entropy per weight against a 16-bit format — so ~68% is the theoretical
floor and we achieve 72%. The compression is close to optimal; the data is
simply not very compressible. Much bigger wins come from checkpointing more
often, or from alignment: a permuted model that would otherwise store at ~88%
now stores at **0.16%**, because the aligner recovers the reordering and finds
the weights are the same ones.

---

## 10. Next

Read **`ARCHITECTURE.md`**. It covers the same system at implementation depth:
exact byte layouts, the algorithms in full, what every source file does, and a
list of traps that look like working code.

`FORMAT.md` and `CLI.md` are reference material — look things up in them rather
than reading them front to back.
