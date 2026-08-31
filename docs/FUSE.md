# `synapsefs/fuse/` — the virtual filesystem, file by file

Read `SYSTEMS_PRIMER.md` first: inodes, file handles, `forget`, cache keys and
read amplification are assumed here.

**What this module does.** It makes a directory that looks like it contains
`.safetensors` files, so `torch.load_file("mount/main/model.safetensors")`
works — while no such file exists. Byte ranges are reconstructed on demand from
compressed residuals.

**Measured, live, on the 92M benchmark:** 176 MiB read back byte-identical in
0.86 s (206 MiB/s) via `cp`, 110 MiB/s via `safetensors.load_file` (mmap), with
the daemon peaking at 292 MiB RSS.

```
cli/commands/mount.py        argument parsing
  daemon.py                  process lifecycle: fork, handshake, signals, unmount
    fs.py                    the FUSE callbacks, inode table, namespace
      reconstruct.py         byte offset -> tensor rows
        cache.py             byte-bounded LRU
          graph.CommitCheckpoint.rows()      the actual decode (not in this module)
```

The layering is worth noticing: **this module imports exactly three things** —
`CommitCheckpoint`, `Repo`, and its own cache. It never touches the codec,
never sees an encoding string, never walks the commit graph. That is why it
survived the packfile removal and the chain-to-star migration without edits.

---

## `__init__.py`

Re-exports `ChunkCache`, `SynapseFSOperations`, `VirtualSafetensorsFile`,
`mount_fuse`, `unmount_fuse`. No logic.

---

## `cache.py` — 64 lines

One class. Bounded by **bytes**, not entry count, because entries range from a
200-byte tensor to a 4 MiB chunk and "keep 100 entries" is not a memory bound.

### `ChunkCache.__init__(max_bytes=512 MiB)`
An `OrderedDict` (insertion-ordered, so the front is least-recently-used), a
running byte total, and a `threading.Lock`. The lock is for safety, not speed:
FUSE reads are dispatched to worker threads, so two can be inside the cache at
once.

### `get(key)`
Miss returns `None`. Hit does `move_to_end` — that is the "recently used" half
of LRU, and forgetting it silently turns the cache into FIFO.

### `put(key, val, nbytes)`
Three cases in order, and each exists for a reason:

1. **Replacing an existing key** subtracts the old size first, or the byte
   total drifts upward forever and the cache slowly starves itself.
2. **An item larger than the whole budget is not stored.** Without this it
   would evict everything else to make room, then be evicted itself — pure loss.
3. **Evict from the front** until it fits.

### `clear()`, `__len__()`
Teardown and introspection.

### What the caller must get right
`ChunkCache` does not know what a chunk is; it stores whatever it is handed
under whatever key. Choosing that key correctly is `reconstruct.py`'s job, and
getting it wrong cost 31.6x — see `_get_rows`.

---

## `reconstruct.py` — 197 lines

Turns "bytes `[offset, offset+size)` of a `.safetensors` file" into "rows of
these tensors". The arithmetic layer; no FUSE here, which is why it is unit
testable without mounting anything.

### `TensorSegment` (frozen dataclass)
One tensor's geometry: `name`, `dtype`, `shape`, `width`, `num_rows`,
`row_elems`, `row_nbytes`, and `abs_begin` / `abs_end` — its absolute byte
range **in the reconstructed file**. Absolute, so `read` intersects ranges
directly instead of tracking a running offset.

### `VirtualSafetensorsFile.__init__(checkpoint, cache=None)`
Holds the `CommitCheckpoint`, the cache, a per-tensor chunk-span cache, and the
stored header. Calls `_parse_header` once — a mounted file is immutable, so the
layout is computed once and reused for every read.

### `_parse_header()`
A safetensors file is:

```
[8 bytes: header length][JSON header][tensor data, back to back]
                                     ^ data_start
```

Reads the length prefix, parses the JSON, and builds one `TensorSegment` per
tensor, skipping `__metadata__`. Element width is *derived*
(`byte_span / element_count`) rather than mapped from the dtype name, so a
dtype this code has never heard of still works.

Segments are **sorted by `abs_begin`**, which lets `read` stop early, and
`total_size` is the largest `abs_end` — the file's size, and what `getattr`
reports.

### `read(offset, size) -> bytes`
The heart of the module.

1. Clamp to `total_size`, allocate the output buffer.
2. **Header region.** If the range touches `[0, data_start)`, copy those bytes
   **verbatim** from the stored header object. This is why reconstruction is
   byte-exact: the header is replayed, never re-serialised, so key order and
   whitespace survive.
3. **Each overlapping segment.** Intersect the request with the segment, then
   convert bytes to rows:
   ```python
   start_row = rel_start // seg.row_nbytes
   end_row   = ceil(rel_end / seg.row_nbytes)
   ```
   Rows are the smallest decodable unit — you cannot decompress half a row —
   so a 1-byte read still costs a row.
4. Fetch those rows, slice the exact bytes out, copy into place.

Segments sorted by offset means the loop `break`s once past the request, so a
read never inspects tensors it does not touch.

### `_spans(name)`
Chunk row boundaries for a tensor, from `CommitCheckpoint.chunk_spans()`,
memoised per tensor. Boundaries are fixed for a commit, so this is read once.

### `_chunk_rows(name, lo, hi)`
Decodes **one whole chunk** and caches it under `(commit, name, lo, hi)` — the
chunk's own span. Cache miss calls `checkpoint.rows()`, which is where
decompression and residual addition happen.

### `_get_rows(name, start_row, end_row)` — the 31.6x bug
Snaps the request **outward to chunk boundaries**, fetches each covering chunk
via `_chunk_rows`, concatenates, and slices back down.

The snapping is the entire point. The original keyed the cache on the requested
row range:

```python
key = (commit, name, start_row, end_row)     # the REQUEST
```

The kernel issues 128 KiB reads; chunks are 4 MiB. Every read produced a fresh
key, missed, and decoded the whole chunk to return 3% of it — **8,761 MiB of
object reads to serve a 177 MiB file, 124x slower** than reading it in one call.

Keyed on the chunk, the first block of a chunk pays for the decode and the next
thirty-one are slices of a hit: 369 fetches instead of 3,189.

> A cache key must describe the unit of work you performed, not the unit you
> were asked for.

---

## `fs.py` — 519 lines

The `pyfuse3.Operations` subclass. Callbacks the kernel invokes, plus the inode
table and namespace.

### The namespace
```
/                          root (inode 1)
/commits/                  every commit reachable from a branch tip (inode 2)
/commits/<hash>/model.safetensors
/<branch>/                 one per branch
/<branch>/model.safetensors
```

### `InodeInfo`
What an inode number means: `name`, `parent_inode`, `is_dir`, and optionally
`commit_hash`, `branch_name`, a built `vfile`, and `lookup_count`.

### `SynapseFSOperations.__init__(repo, ref_filter=None, cache_size_bytes=512 MiB)`
Builds the chunk cache and four tables:

- `_inodes` — inode -> `InodeInfo`
- `_by_name` — `(parent, name)` -> inode. **Without this, `_alloc_inode`
  scanned every allocated inode on every lookup**, which is quadratic in
  commits touched.
- `_fh_to_vfile` — open file handles
- `_vfile_cache` / `_size_cache` — per commit

Inodes 1 and 2 are created here and never freed.

### `close()`
Clears the cache. Chunk files are opened per read, so there are no long-lived
handles.

### `_get_vfile(hash)` / `_resolve_commit(hash)`
Build (and memoise) a `VirtualSafetensorsFile` for a commit. `_resolve_commit`
is the exception-swallowing wrapper, so a damaged commit yields `ENOENT`
instead of crashing the daemon.

### `_get_branch_commit(branch)`, `_list_branches()`
Read `refs/heads/<branch>`; list branches, honouring `--ref`.

### `_list_commits()`
Every commit **reachable** from a branch tip, via `graph.ancestors`, not just
the tips. `lookup` always resolved any hash, so history was reachable by typing
a path but invisible to `ls` — the listing disagreed with what `cd` accepted.
Walks all parents, not first-parent, so a merge's second parent is included.

### `_commit_size(hash)`
File size for `stat`, cached as an **int**. Building a whole
`VirtualSafetensorsFile` for a size would parse the header and retain a segment
table per commit, so `ls -l /commits/` would do that for all 25.

### `_alloc_inode(...)`
Returns the existing inode for `(parent, name)` or allocates the next integer.
`_by_name` makes the lookup O(1).

### `_looked_up(inode)`
`_get_entry_attrs` **plus** incrementing `lookup_count`. Every reply from
`lookup` goes through this, because the kernel counts those replies and
`forget` decrements them — the two must balance or inodes leak.

### `_get_entry_attrs(inode)`
Fills a `pyfuse3.EntryAttributes`: mode (`0o555` dirs, `0o444` files — read-only
in the mode bits, not only by policy), nlink, size, uid/gid, timestamps frozen
at mount time, and 1 s attribute/entry timeouts so the kernel caches metadata.

### `async lookup(parent_inode, name, ctx)`
Path resolution, one component at a time. Three cases: under root (`commits`,
or a branch name); under `commits/` (resolve a hash or abbreviation via
`repo.resolve_ref`); inside a branch or commit directory (a `.safetensors`
name). Anything else raises `ENOENT`.

### `async getattr(inode, ctx)`
`stat`. Does **not** count as a lookup.

### `async forget(inode_list)`
Decrements `lookup_count` by the kernel's count and drops the inode at zero,
except inodes 1 and 2, which the kernel may reference at any time without a
fresh lookup. Missing this leaks one entry per path ever looked up.

### `async opendir` / `readdir(fh, start_id, token)`
`readdir` builds `.`, `..` and the children, then emits from `start_id` — the
kernel may resume a listing, so entries need stable ids across calls.

### `async open(inode, flags, ctx)`
Rejects `O_WRONLY`, `O_RDWR`, `O_CREAT`, `O_TRUNC` with `EACCES`, then returns
a `FileInfo` with `keep_cache=True` (content is immutable, so let the kernel
page-cache it) and `direct_io=False`.

### `async read(fh, off, size)`
```python
return await trio.to_thread.run_sync(vfile.read, off, size)
```
pyfuse3 runs on a trio event loop — one thread juggling many requests. Decoding
is pure CPU; doing it on the loop thread would stall every other request. This
hop is the only reason the mount stays responsive under concurrent readers.

### `async release` / `releasedir` / `statfs`
Drop the handle; no-op; plausible fabricated `statvfs` numbers so `df` works.

---

## `daemon.py` — 263 lines

Process lifecycle. No FUSE logic.

### `_mount_id` / `_mount_record_path` / `record_mount` / `remove_mount_record` / `find_mount_pid`
A mount is recorded at `.synapse/mounts/<blake3(mountpoint)[:16]>.json` holding
the pid, so `unmount` can find the daemon. Keyed by the **resolved** path, so
two spellings of one mountpoint agree.

### `run_fuse_loop(ops, mountpoint, options)`
`pyfuse3.init`, `trio.run(pyfuse3.main)`, and unmount in a `finally`.

### `mount_fuse(...)`
Foreground mode runs the loop directly. Background mode does the **double fork**
so the daemon detaches from the shell — and then the parent cannot see whether
mounting worked.

So the parent opens a **pipe**. The child writes `b"OK"` after `pyfuse3.init`
succeeds, or the error text if it fails; the parent blocks reading it. You get a
real error at the prompt instead of exit 0 and a broken mountpoint.

The second fork prevents the daemon reacquiring a controlling terminal.
`SIGINT`/`SIGTERM` unmount cleanly and remove the record.

### `unmount_fuse(repo, mountpoint)`
`SIGTERM` the recorded pid and wait up to 2 s; fall back to `fusermount3 -u`,
`fusermount -u`, `umount`; then confirm with `os.path.ismount`. Removes the
record either way, so a crashed daemon does not leave one behind forever.

---

## Testing it

`tests/test_fuse_vfs.py` exercises `VirtualSafetensorsFile` directly — no
mount, no kernel, fast. That covers the arithmetic and nothing else.

**Both bugs found in this module were invisible to it**, because its fixtures
are small enough that a tensor has one chunk, so nothing fans out and no read
is ever smaller than a chunk. The live test is the one that matters:

```bash
synapsefs -C <repo> mount /tmp/mnt
cp /tmp/mnt/main/model.safetensors /tmp/out.safetensors
cmp /tmp/out.safetensors <original>          # must be byte-identical
synapsefs -C <repo> unmount /tmp/mnt
```

Do that for a **residual** commit as well as a full one — only the residual
path exercises the recursion to the anchor.

## Known remaining work

- **Parallel chunk decode.** `_rows_from_manifest` decodes chunk after chunk on
  one core. `zstandard` releases the GIL, so plain threads would give real
  parallelism; expect 3-5x, and see `SYSTEMS_PRIMER.md` §8 for the deadlock the
  recursion invites.
- **Readahead.** After the cache fix, 31 of every 32 reads are pure hits with
  no decode. Only the misses are slow, so speculatively decoding the next chunk
  on a miss is what would lift the 110 MiB/s mmap figure.
- **A C++ port is not the answer to speed** — see `ARCHITECTURE.md` §4.4.2. The
  read path is already ~65% compiled code; the argument for C++ is memory.
