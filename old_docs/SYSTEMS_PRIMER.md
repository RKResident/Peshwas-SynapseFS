# Systems primer

For the parts of SynapseFS that are systems work rather than machine learning:
the virtual filesystem, and the caches. Assumes you can read Python and know
what a checkpoint is, and assumes nothing about kernels, syscalls or caching.

Read this before `FUSE.md` or `ALIGNMENT.md`. Those two go function by function
and take these ideas for granted.

---

## 1. What happens when a program opens a file

```python
f = open("/home/me/model.safetensors")
data = f.read(1024)
```

Neither line is a function call in the ordinary sense. Both are **syscalls** —
requests handed to the operating system kernel, which is the only thing allowed
to touch a disk.

The kernel keeps a layer called the **VFS** (virtual filesystem switch) whose
job is to answer "which filesystem owns this path?" and forward the request. On
ext4 the answer is "the ext4 driver, which reads blocks off an SSD."

The important thing is that the program never learns where the bytes came from.
It asks for 1024 bytes at an offset; something produces them. Whether they were
on a disk, in RAM, on another machine, or *computed on the spot* is invisible.

That last possibility is what SynapseFS uses.

## 2. FUSE: a filesystem written by you, in userspace

**FUSE** is Filesystem in USErspace. It's a kernel module that forwards
filesystem requests to an ordinary program instead of handling them itself.

```
  cat mount/model.safetensors
        |
        v
  kernel VFS  --"who owns /mount?"-->  the FUSE driver
        |
        v
  our Python process:  "read 128 KiB at offset 4096"
        |
        v
  we decompress a chunk, add a residual, hand back the bytes
```

Your program registers a set of callbacks — `lookup`, `getattr`, `open`,
`read`, `readdir` — and the kernel calls them. Answer honestly and the directory
behaves like a real one to every program on the machine: `cat`, `cp`, Python,
PyTorch, all of it.

**Why this matters here.** A checkpoint can be *used* without ever existing.
`torch.load_file("mount/model.safetensors")` works, and there is no 3 GB file
anywhere — we reconstruct exactly the bytes it asks for, when it asks.

The cost is that every read is a round trip into our process and back. FUSE is
never as fast as a real filesystem; the question is only whether it is fast
enough.

## 3. Inodes: the kernel does not use paths

The first genuine surprise.

You think in paths — `/mount/main/model.safetensors`. The kernel does not. It
uses **inodes**, which are just integers naming a file or directory.

So the kernel never asks us to "read /mount/main/model.safetensors". It walks
the path one component at a time, asking us to translate each:

```
lookup(parent=1, name="main")              -> "that is inode 3"
lookup(parent=3, name="model.safetensors") -> "that is inode 4"
getattr(inode=4)                           -> size, permissions, timestamps
```

Inode 1 is always the root of your filesystem. Everything else you allocate,
and you must remember what each number means — so a FUSE program always has a
table mapping integers to "what does this one refer to".

## 4. File handles: the kernel does not re-ask either

When a program opens a file you return a **file handle**, another integer.
Every later `read` says "read from handle 107", not "from inode 4".

Handles exist because the same file can be open several times at once with
different state. They are also where you put per-open setup, so it happens once
rather than on every read.

## 5. Lookup counts and `forget`

Subtle, and easy to get wrong by omission.

Every time you answer a `lookup`, the kernel **caches** the answer — it
remembers "inode 4 is that file" so it need not ask again. It is now holding a
reference into *your* memory.

So you cannot free inode 4 whenever you like. FUSE handles this with a counter:
each `lookup` reply increments it, and when the kernel is done it calls
`forget(inode, n)` meaning "I have dropped n references."

Skip `forget` and nothing breaks — entries just stay valid forever. It is a
slow leak, not a crash, which is exactly why it survives casual testing. Ours
was missing until it was looked for.

## 6. Caching

A cache is a dictionary in front of something expensive. You look up a **key**;
a **hit** returns the stored value, a **miss** does the expensive thing and
stores the result.

Two decisions define a cache, and both can be got wrong quietly.

### 6.1 The key

The key is what you look things up by, and it decides whether the cache ever
helps at all.

Our chunk cache was keyed on `(commit, tensor, start_row, end_row)` — the
**request**. Reasonable-looking. But the kernel hands out 128 KiB reads while
the compression unit, a **chunk**, is 4 MiB. So every read asked for a slightly
different row range, produced a brand-new key, missed, and decompressed the
whole 4 MiB chunk to return 3% of it:

| | object fetches | bytes read | time |
|---|---|---|---|
| one whole-file read | 369 | 277.7 MiB | 2.4 s |
| 128 KiB reads, request-keyed | 3,189 | **8,761.8 MiB** | 297.8 s |
| 128 KiB reads, chunk-keyed | 369 | 277.7 MiB | 1.5 s |

**8.7 GB of disk reads to serve a 177 MB file.** The cache was pure overhead:
it stored everything and returned nothing.

The fix is to key on the **chunk**, the thing that was actually decoded. Then
the first read of a chunk pays for it and the next thirty-one are slices of a
hit.

> **A cache key must describe the unit of work you performed, not the unit of
> work you were asked for.**

That is the single most transferable idea in this document, and it is why the
alignment solver's cache in §6.4 looks so different from this one.

### 6.2 Eviction, and why caches are bounded

An unbounded cache is a memory leak with good intentions. If the data is bigger
than RAM, something must be thrown out.

**LRU** — least recently used — evicts whatever has gone longest untouched, on
the bet that recently used things get used again. It is not optimal, it is
cheap and usually right.

Ours are bounded **by bytes, not by entry count**, because the entries are
wildly different sizes: one 4 MiB chunk against a 200-byte one. "Keep 100
entries" says nothing about memory; "keep 512 MiB" is a promise you can check:

```
  cache   32 MiB -> idle 57 MiB, peak 168 MiB
  cache  128 MiB -> idle 57 MiB, peak 266 MiB
  cache  512 MiB -> idle 59 MiB, peak 292 MiB
```

Reading a 176 MiB checkpoint never approaches its size. That is the property
the whole project is for: use a model bigger than your RAM.

### 6.3 Read amplification

The ratio between bytes you moved and bytes you were asked for. 8,761 / 177 =
**31.6x** above.

It appears whenever the unit you *store* is bigger than the unit you *serve*.
Compression forces exactly that: you cannot decompress half a chunk, so a
1-byte read costs a whole chunk. The cure is never "smaller chunks" — that
costs compression ratio — it is caching the chunk so the cost is paid once.

### 6.4 The same idea, somewhere that looks unrelated

The alignment solver had no filesystem in it and the identical bug.

It reads tensors as 16-bit floats and must widen them to 32-bit to do linear
algebra. That conversion — `.astype(np.float32)` — is the single most expensive
thing it does. And it was redoing it constantly: **694 conversions of 144
distinct tensors** in one pass, because each permutation group re-read its own
members, a tensor belonging to two groups was converted twice, and the
measurement pass converted everything again.

Same disease, different organ: expensive work repeated because nothing
remembered it. Same cure — a byte-bounded LRU. Not "always cache", which would
be a 56 GB dictionary on a 7B model, but a **cap**: a small model fits entirely,
a huge one keeps what it can and degrades to re-reading the rest.

And the same lesson about measuring rather than assuming, because caching alone
only bought 1.36x. Re-profiling found the real cost hiding behind it: a
one-line residual calculation that allocated two 231 MiB temporaries per call,
144 times per alignment. Removing that took 15.2 s to 6.0 s.

> Profile, fix the top item, **profile again**. The second bottleneck is
> invisible until the first is gone.

## 7. Memory: RSS, and why streaming matters

**RSS** (resident set size) is how much physical RAM a process actually holds.
`VmHWM` in `/proc/<pid>/status` is its high-water mark — the peak, which is
what determines whether you get killed by the OOM killer.

A program that loads a 3 GB checkpoint has 3 GB of RSS. A program that
**streams** it — a chunk at a time, discarding each — holds only a chunk. The
FUSE daemon streams, which is why 176 MiB flows through it at ~292 MiB peak
rather than needing 176 MiB resident plus overhead.

The enemy of streaming is a temporary. Every `x.astype(...)`, every slice that
copies, every intermediate list allocates a full-size buffer, and Python's
convenience makes them easy to write without noticing. The 231 MiB temporaries
in §6.4 were one line of very reasonable-looking code.

## 8. Threads, and the one Python detail that matters

A CPU has many cores; a normal Python program uses one.

Python has the **GIL** (global interpreter lock): only one thread may execute
Python bytecode at a time. So threads usually do *not* speed up CPU-bound
Python — they take turns.

The escape hatch: a thread calling into a C library that does not touch Python
objects can **release the GIL** while that C code runs. `zstandard` does this
during decompression. So our chunk decoding is one of the rare cases where
plain threads would give real multi-core speedup — the work happens in C,
outside the interpreter.

Two ceilings to expect if you try it. **Amdahl's law**: the serial parts do not
shrink, so making 70% of a job infinitely fast still leaves 30%. And **memory
bandwidth**: adding two 4 MiB arrays is data movement, not computation, and all
cores share one path to RAM. Realistic expectation is 3-5x, not 24x.

There is also a trap specific to recursive work. If eight worker threads each
take a task, and each task needs a sub-task that must also be done by a worker,
all eight can end up waiting for a worker that will never be free. That is a
**deadlock**, and it does not appear in small tests because small inputs never
fan out. The fix is not a bigger pool; it is making nesting impossible — only
the outermost call is allowed to use the pool, and anything a worker needs it
does itself.

## 9. Atomic writes

`rename()` on POSIX is atomic: after a crash the destination is either the old
file or the new one, never half of both. Writing directly to the destination
has no such guarantee.

So every writer here — the object store, the network receiver — writes to
`<path>.tmp` and renames. An interrupted transfer leaves a complete object or
none, never a truncated one that hashes to nothing and poisons the store.

## Where to go next

- **`FUSE.md`** — every file and function in `synapsefs/fuse/`.
- **`ALIGNMENT.md`** — every file and function in `synapsefs/align/`.
- **`ARCHITECTURE.md` §4.4.1** — the two bugs above with full measurements.
- **`PRIMER.md`** — the compression and versioning side, no systems assumed.
