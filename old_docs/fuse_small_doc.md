# The SynapseFS mount

A read-only FUSE filesystem that presents every commit in a repository as a
`.safetensors` file, reconstructed byte-range by byte-range as it is read.
Nothing is written to disk when you mount, and nothing is assembled ahead of
time — a request for bytes `[o, o+n)` decodes only the chunks those bytes
actually land in.

Built on `pyfuse3` + `trio`. `synapsefs/fuse/` is four files: `daemon.py`
(process lifecycle), `fs.py` (the FUSE operations), `reconstruct.py` (offset →
tensor rows), and `cache.py` (the decoded-chunk cache).

---

## Namespace

```
<mountpoint>/
├── commits/
│   └── <64-hex commit hash>/model.safetensors
└── <branch>/model.safetensors          # e.g. main/
```

`commits/` lists every commit reachable from a branch tip, not just the tips —
history is walked through `parents`. Branch directories are aliases for
whatever their ref currently points at.

Inodes are allocated lazily on `lookup` and memoised by `(parent, name)`.
Without that map, `_alloc_inode` scans every allocated inode per lookup, which
is quadratic in the number of commits a mount has touched.

`ls -l commits/` is deliberately cheap: `getattr` needs a size and nothing
else, so file sizes come from a separate `_size_cache` rather than building a
`VirtualSafetensorsFile` (and its segment table) for every commit in the
listing.

---

## How a read is served

A `.safetensors` file is a length-prefixed JSON header followed by tensor data
laid out contiguously. On the way in, the header is stored **verbatim** as an
object; on the way out it is served as bytes. So the mount does not need to
re-serialise anything to be byte-identical — it replays the original header and
reconstructs the data region underneath it.

`VirtualSafetensorsFile` parses that header once per commit into a segment
table sorted by absolute file offset. Each segment records a tensor's byte span
and its row geometry. A read then walks:

```
byte offset  ->  which tensor segment
             ->  which rows of that tensor
             ->  which chunks hold those rows
             ->  decode, slice, copy into the output buffer
```

The last step is where the storage format shows through. Tensors are stored as
row-range chunks; a residual chunk is a delta against a chunk of a *base*
commit, so decoding one chunk can require decoding another. That recursion is
`graph.py`'s job, and the mount just asks for rows.

**Chunks are the unit of decoding, and the cache is keyed on the chunk, not on
the request.** This is the single most important decision in the read path.
FUSE hands out 128 KiB reads; chunks are 1 MiB. Keying the cache on the
requested range looks equivalent and is not — a sequential reader whose block
is smaller than a chunk produces a fresh key every call, never hits, and
re-decodes the entire chunk to return a fraction of it. Measured on the 92M
benchmark that was **8,762 MiB of object reads to serve a 177 MiB file**, a
31.6× amplification. Snapping requests to chunk boundaries makes the first
block of a chunk pay for the decode and the rest be slices of a cache hit.

---

## The two-path read

`fs.read` splits on whether the data is already decoded:

- **hit** — every chunk the request touches is in the cache. Served *inline* on
  the trio event loop; it is a `memoryview` slice, microseconds, non-blocking.
- **miss** — needs a decode. Dispatched to a worker thread via
  `trio.to_thread.run_sync`, because decoding on the event loop would stall
  every other request for its duration.

The split exists because dispatch is not free. On a mount whose `read()`
returned a preallocated buffer — no decode at all, isolating the transport —
the same workload measured **504 MiB/s through `to_thread` against 1959 MiB/s
inline**. That ~4× is irrelevant against a 4.5 ms decode (~4% of a miss) and
catastrophic against a hit, where the work is nothing. On a sequential mmap
load, ~85% of requests take the inline path.

Set `SYNAPSEFS_INLINE_HITS=0` to force everything through the thread pool.

### Worker threads

`trio`'s default thread limiter is 40. Each in-flight decode holds a
decompressed stream, an unshuffled array, and the same again for the delta
base, so 40 concurrent decodes is a large transient footprint for no gain — the
numpy half of the decode holds the GIL anyway. Capped at 8 via
`SYNAPSEFS_READ_THREADS`.

### Single-flight

`ChunkCache.get_or_compute` guarantees a chunk is decoded **at most once**
across threads. Plain check-miss-decode-put has a window between the miss and
the put where the value is being produced but is not yet visible; kernel
readahead fires several requests into the same chunk at once, worker threads
pick them up together, and every one of them misses and decodes the same bytes
to the same answer.

Measured, one reader, cache far larger than the working set so eviction could
not be the cause:

```
SYNAPSEFS_READ_THREADS=1    1.00x the minimum object bytes
SYNAPSEFS_READ_THREADS=8    1.48x        <- before single-flight
SYNAPSEFS_READ_THREADS=8    1.00x        <- after
```

The first caller to miss owns the decode; the rest wait on a `_Flight` and
receive the same array object. It costs peak RSS as well as time — duplicate
decodes each allocate their own buffers, where sharing one result costs a
reference.

---

## Memory

Peak daemon RSS on the 90M benchmark runs **120–164 MB** across every workload
shape. Three things set it, and the chunk cache is the smallest of them:

**Floor: ~44 MiB before a single byte is served.** 11.5 MiB interpreter,
+15 numpy, +15 trio/pyfuse3. Unavoidable without leaving Python.

**The chunk cache**, `--cache-size`, default 32 MiB. Size it by the number of
**distinct** checkpoints read concurrently — roughly 64 MiB each — not by the
number of readers. Eight processes loading the *same* checkpoint share one
working set; eight loading eight *different* commits share nothing and will
thrash a small cache. `mount --help` carries the measured table.

**Allocator high-water from decode churn.** Decoded chunks are large,
short-lived allocations, and glibc grows a per-thread arena for each rather
than returning memory to the OS. `daemon.py` calls `mallopt(M_ARENA_MAX, 2)`
before any worker thread exists — arenas are created lazily on first allocation
from a new thread, so it has to happen early.

This is why amplification is the shared root cause of both throughput *and*
memory: doing the same decode twice costs twice the CPU and twice the
allocation pressure.

---

## Chunk size

1 MiB (`DEFAULT_CHUNK_SIZE_BYTES`), and this is a read-path parameter, not a
storage one. The chunk is the decode unit, so a page fault from a memory-mapped
reader costs one whole chunk plus its delta base. At 4 MiB that is a 2048:1
mismatch against a 4 KiB fault.

Swept on epochs 1–4 of the 90M benchmark, four concurrent readers of four
distinct commits:

```
chunk    store    mmap MB/s   peak RSS   amplification
4 MiB   584 MiB       35.66     293.6MB      47.0x
1 MiB   589 MiB       63.65     173.8MB      10.7x
512KiB  594 MiB       63.40     148.7MB       3.1x
```

**+78% throughput and −41% RSS for +0.9% stored bytes.** The compression cost
everyone expects to pay here is nearly absent — the zstd window was never the
binding constraint at these sizes. Smaller chunks also removed the fault storm:
4 KiB single-page requests went from **42% to 1%** of all FUSE requests.

Chunk bounds are per-manifest, so changing this needs no migration — old
commits stay readable and new ones use the new size.

---

## Where the time goes

Codec, single-threaded, one delta commit (176.3 MiB delivered):

```
zstd decompress    0.332 s   83%
unshuffle (Cython) 0.045 s
delta add          0.021 s
                   -------
                   0.399 s   =  442 MiB/s ceiling
```

Actual in-process reconstruction reaches **366 MiB/s, 83% of that** — the
Python glue costs 17%. zstd is the codec now; the Cython kernel took unshuffle
from 32.6% of the read path to ~5%.

Transport ceiling, measured with decode stubbed out: **1959 MiB/s** for
pyfuse3 served inline, against **2204 MiB/s** for a hand-written C libfuse3
filesystem doing the same thing. pyfuse3 is 89% of C, so the transport is not
the constraint and a rewrite in C++ would buy ~12% there.

Cold, on the 1 MiB repo, against an analytic minimum of 283.5 MiB of objects:

```
workload                        MiB/s   daemon read   amp    peak RSS
read() sequential, 1 reader     148.6      283.6 MiB  1.00x    120 MB
mmap load_file, 1 reader        191.6      338.3 MiB  1.19x    152 MB
mmap, 8 readers, same file      183.5      339.6 MiB  1.20x    150 MB
mmap, 8 readers, 8 distinct      67.3    13874.8 MiB  6.12x    164 MB
```

`1.00x` means every object was read exactly once — nothing decoded twice. The
last row is the one shape that still thrashes: eight independent working sets
against a 32 MiB cache.

---

## Behaviour worth knowing

**Files are opened with `keep_cache=True` and `direct_io=False`.** The kernel
page-caches what the daemon serves, so a re-read of the same file never reaches
the daemon at all — warm reads measure the page cache, not this filesystem.
Any benchmark that does not drop caches (or remount) is measuring the wrong
thing; a warm re-read clocks ~5900 MiB/s with the daemon asleep.

**A single reader cannot saturate the daemon.** FUSE bdi readahead is 128 KiB,
so one mmap reader keeps ~1–2 requests in flight and most of the eight decode
threads idle. Concurrency raises aggregate throughput several-fold; per-load
latency does not improve. Watch `Slowest Worker`, not aggregate MB/s, if you
care about how long one load takes.

**`max_read` cannot be raised.** libfuse rejects it under pyfuse3
(`init() and fuse_session_new() requested different maximum read size`), and
pyfuse3 3.5.0 exposes no API for it. The kernel batches to 256 KiB regardless.

**Concurrent readers are safe.** Eight processes reading the same file return
byte-identical data; single-flight makes them share one decode rather than race
on it.

---

## Tuning summary

| knob | default | when to change |
|---|---|---|
| `--cache-size` | 32 MiB | ~64 MiB × distinct checkpoints read at once |
| `SYNAPSEFS_READ_THREADS` | 8 | rarely; fewer is slower, more buys nothing |
| `SYNAPSEFS_INLINE_HITS` | 1 | set 0 only to isolate the dispatch cost |
| `--chunk-size` (at commit) | 1 MiB | smaller trades ~1% size for lower RSS |
| `OMP/OPENBLAS_NUM_THREADS` | unset | honoured if set; alignment caps to 8 otherwise |

---

## Benchmarks

`tools/tools/benchmark/` holds the harnesses:

- **`bench_fuse.py`** — `read()` at random offsets, concurrent processes, with
  bitwise verification against a ground-truth file.
- **`bench_load.py`** — `safetensors.torch.load_file`, i.e. the **mmap** path.
  Reports daemon bytes served against the analytic minimum, so amplification is
  visible rather than hidden inside a throughput number. `--remount` gives a
  cold mount without `sudo`; `--distinct-commits` avoids measuring the page
  cache.
- **`rss_trace.py`** — wraps either one and plots daemon RSS over time.

Both report peak daemon RSS. Prefer cold numbers: the PS specifies benchmarking
from a cold OS page cache, and warm numbers here are the kernel's, not ours.
