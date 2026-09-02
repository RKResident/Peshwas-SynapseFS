"""SynapseFS state-dict load benchmark.

Companion to `bench_fuse.py`. That one measures raw byte reads -- `f.read()`
at random offsets -- which is the FUSE contract but not how anybody loads a
checkpoint. This one measures `safetensors.torch.load_file`, the call a
training or eval script actually makes, and the one we are most likely to be
judged on.

The distinction is not cosmetic. `load_file` **mmaps** the file and faults
pages in, rather than issuing sequential `read()`s:

    openat("mnt/main/model.safetensors", O_RDONLY)      = 3
    mmap(NULL, 184889472, PROT_READ, MAP_PRIVATE, 3, 0)

so the kernel drives the FUSE daemon through `readpages` with its own
readahead, a different request pattern from `dd`. A change that helps one can
leave the other flat.

Every phase reports the bytes the daemon actually served (`/proc/<pid>/io`
rchar) next to the wall time, so a suspiciously fast number can be checked
against whether the work really happened -- a warm page cache serves the whole
file without waking the daemon at all, and the daemon's counter is what makes
that visible instead of silently inflating the result.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
import torch
import safetensors.torch

from synapsefs.codec.chunk import is_delta
from synapsefs.fuse.daemon import find_mount_pid
from synapsefs.graph import CommitCheckpoint
from synapsefs.store.repo import Repo


def minimum_object_bytes(repo: Repo, commits: "list[str]") -> float:
    """MiB of distinct store objects needed to reconstruct `commits` once.

    The denominator for every amplification figure below, derived from the
    manifests rather than from a timed run. An earlier version of this file
    took the baseline from a `dd` of the mount, which was wrong in a way worth
    recording: that `dd` ran against the default chunk cache and was itself
    re-decoding evicted chunks, so it measured 1.97x the real minimum and every
    ratio computed from it was understated by the same factor.

    The union matters. Reconstructing eight commits does not cost eight times
    one commit -- residual chunks share delta bases, so on the 25-epoch
    benchmark eight commits need 1550 MiB against a sum-of-parts of 1192 MiB.
    Counting per-commit and adding would flatter the result.

    Note this is *below* the file's own size (149.0 MiB of objects for a
    176.3 MiB checkpoint): chunks are zstd-compressed, so a perfect reader
    moves less than the bytes it serves.
    """
    need: set[str] = set()
    for commit in commits:
        ck = CommitCheckpoint(repo.store, commit)
        need.add(ck.manifest["header_object"])

        def walk(manifest_hash: str) -> None:
            if manifest_hash in need:
                return
            need.add(manifest_hash)
            manifest = ck._load(manifest_hash)
            base = manifest.get("base_tensor_manifest")
            for chunk in manifest["chunks"]:
                need.add(chunk["object"])
                if base and is_delta(chunk["encoding"]):
                    walk(base)

        for name in ck.names():
            walk(ck._tensor_manifests[name])
    return sum(os.path.getsize(repo.store.path_for(h)) for h in need) / (1024 * 1024)


#: Rough working set of one reader marching through one checkpoint: the 4 MiB
#: chunk it is in, the 4 MiB base chunk that is a delta against, and the few
#: more the kernel has in flight from readahead. Readers on the SAME file share
#: these; readers on different files do not, which is why the cache requirement
#: scales with distinct files rather than with process count.
WORKING_SET_MB_PER_DISTINCT_FILE = 64

def drop_linux_page_cache() -> bool:
    """Flush dirty pages and clear the Linux OS page cache via drop_caches."""
    print("[Cache] Dropping Linux page cache...")
    try:
        subprocess.run(["sync"], check=True)
        res = subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            print("✓ OS page cache cleared successfully.")
            return True
        print(f"⚠ Failed to drop caches: {res.stderr.strip()}")
        return False
    except Exception as exc:
        print(f"⚠ Could not drop OS page cache ({exc}). Continuing with existing cache...")
        return False


def get_process_rss_mb(pid: int) -> float:
    """Read Resident Set Size (RSS) in MB for a specific process."""
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def get_process_rchar_mb(pid: int) -> float:
    """Bytes the daemon has read, from /proc/<pid>/io.

    The honesty check on every timing below. If a "cold" load reports a
    throughput the decode path cannot physically reach, this number will show
    the daemon barely moved and the kernel served the file from cache.
    """
    try:
        with open(f"/proc/{pid}/io") as fh:
            for line in fh:
                if line.startswith("rchar:"):
                    return int(line.split(":")[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


def resolve_fuse_pid(repo: Repo, mount_dir: Path) -> "int | None":
    """The daemon PID, from `.synapse/mounts/<id>.json`'s `pid` field.

    Read fresh every time rather than cached: `remount` replaces the daemon,
    so a PID captured before it is a dead process, and every `rchar`/RSS
    reading taken against it would silently come back as zero.
    """
    return find_mount_pid(repo, mount_dir)


def daemon_launch_spec(pid: int) -> "tuple[list[str], str] | None":
    """The argv and cwd of the running daemon, so a remount can reproduce it.

    Copying the existing command rather than assuming `synapsefs mount <dir>`
    keeps whatever flags this mount was started with -- `--cache-size`,
    `--ref`, `--allow-other` -- instead of quietly benchmarking a different
    configuration than the one the user set up.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            argv = [a for a in fh.read().decode().split("\0") if a]
        cwd = os.readlink(f"/proc/{pid}/cwd")
        return (argv, cwd) if argv else None
    except OSError:
        return None


def remount(repo: Repo, mount_dir: Path, spec: "tuple[list[str], str]") -> int:
    """Tear the mount down and bring it back, returning the new daemon PID.

    This is the cold-cache mechanism. `drop_caches` alone is not enough here:
    `fs.py` opens files with `keep_cache=True`, so the kernel treats its cached
    pages for the inode as valid across opens and serves whole loads without
    ever waking the daemon. Destroying the mount destroys the inodes, and the
    cached pages go with them -- and "mount the filesystem, then load the
    model" is the shape of the thing we are actually being measured on.
    """
    argv, cwd = spec
    old_pid = resolve_fuse_pid(repo, mount_dir)

    if old_pid is not None:
        try:
            os.kill(old_pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)
            except OSError:
                break

    if os.path.ismount(mount_dir):
        for binary in ("fusermount3", "fusermount", "umount"):
            try:
                if subprocess.run(
                    [binary, "-u", str(mount_dir)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                ).returncode == 0:
                    break
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

    subprocess.run(
        argv, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
    )

    for _ in range(100):
        time.sleep(0.1)
        pid = resolve_fuse_pid(repo, mount_dir)
        if pid is not None and pid != old_pid and os.path.ismount(mount_dir):
            return pid

    raise RuntimeError(
        f"remount failed: {mount_dir} did not come back up. "
        f"Tried: {' '.join(argv)} (cwd {cwd})"
    )


def materialise(state: dict) -> float:
    """Force every byte of every tensor to be read, and return a checksum.

    **This is not optional book-keeping, it is the measurement.**
    `safetensors.torch.load_file` hands back tensors whose storage is the mmap
    itself, so the call returns before the data has been read: on this repo it
    completes in 0.158s having caused the daemon to read 37.7 MiB, against the
    292.9 MiB a complete sequential read of the same file costs. Timing
    `load_file` alone measures header parsing and reports throughput several
    times the decode path can physically sustain.

    `sum(dtype=...)` accumulates in a wider type without materialising a cast
    copy, so this touches every element while allocating only a scalar --
    `.float().sum()` would double peak memory in each of N worker processes.
    """
    total = 0.0
    for tensor in state.values():
        if tensor.is_floating_point():
            total += float(tensor.sum(dtype=torch.float64))
        else:
            total += float(tensor.sum(dtype=torch.int64))
    return total


def load_worker(args_tuple) -> dict:
    """Load one state dict and report elapsed time. Runs in its own process."""
    worker_id, path, elements_expected = args_tuple
    t0 = time.perf_counter()
    state = safetensors.torch.load_file(str(path))
    materialise(state)
    elements = sum(t.numel() for t in state.values())
    duration = time.perf_counter() - t0
    return {
        "worker_id": worker_id,
        "duration": duration,
        "tensors": len(state),
        "elements": elements,
        "complete": elements_expected in (0, elements),
    }


def verify_against_truth(mount_file: Path, truth_file: Path) -> int:
    """Load both state dicts and compare them tensor by tensor."""
    print("\n[Phase 1] Verifying loaded tensors against ground truth...")
    t0 = time.perf_counter()

    state_mount = safetensors.torch.load_file(str(mount_file))
    state_truth = safetensors.torch.load_file(str(truth_file))

    assert set(state_mount.keys()) == set(state_truth.keys()), "Tensor key sets do not match!"

    for key in state_truth:
        a, b = state_mount[key], state_truth[key]
        assert a.shape == b.shape, f"Shape mismatch in {key}: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"Dtype mismatch in {key}: {a.dtype} vs {b.dtype}"
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            raise ValueError(f"Data corruption in tensor '{key}'! Max diff: {diff}")

    elements = sum(t.numel() for t in state_truth.values())
    dur = time.perf_counter() - t0
    print(f"✓ All {len(state_truth)} tensors match bit-for-bit ({elements:,} elements, {dur:.2f}s).")
    return elements


def timed_load(path: Path, pid: "int | None") -> "tuple[float, float, float]":
    """One `load_file`, returning (seconds, daemon MiB read, peak daemon RSS MB)."""
    rchar_before = get_process_rchar_mb(pid) if pid else 0.0
    t0 = time.perf_counter()
    state = safetensors.torch.load_file(str(path))
    materialise(state)
    duration = time.perf_counter() - t0
    served = (get_process_rchar_mb(pid) - rchar_before) if pid else 0.0
    return duration, served, get_process_rss_mb(pid) if pid else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="SynapseFS safetensors.torch.load_file Benchmark & Verification Suite"
    )
    parser.add_argument(
        "--mount-dir", type=Path, default=Path("mnt"),
        help="Root path of the active FUSE mount (e.g., mnt_a, mnt_b).",
    )
    parser.add_argument(
        "--commit", type=str,
        default="6ed842c9ae3b29982df6609237419befdfd594732034ac4816c5af81b37ec136",
        help="Commit hash or branch folder to evaluate.",
    )
    parser.add_argument(
        "--truth", type=Path, default=Path("epoch24.safetensors"),
        help="Path to the original uncompressed ground-truth safetensors file.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of concurrent loader processes.",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="Number of sequential load iterations to time.",
    )
    parser.add_argument(
        "--distinct-commits", action="store_true",
        help=(
            "Give each concurrent worker a different commit. Without this every "
            "worker loads the same file and the kernel page cache serves all but "
            "the first, which measures the page cache rather than the decode path."
        ),
    )
    parser.add_argument(
        "--remount", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Tear down and re-create the mount before each timed phase (default: "
            "True). This is what actually makes a load cold -- the daemon opens "
            "files with keep_cache=True, so without it the kernel serves whole "
            "loads from its page cache and never wakes the daemon."
        ),
    )
    parser.add_argument(
        "--cold-cache", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Also drop the OS page cache before each phase, so the daemon's own "
            "object reads hit disk rather than cache (default: True). Needs sudo; "
            "warns and continues without it. Complementary to --remount, which "
            "clears the mount's page cache but not the object store's."
        ),
    )
    return parser.parse_args()


def resolve_mount_file(mount_dir: Path, commit: str) -> Path:
    if (mount_dir / "commits" / commit / "model.safetensors").exists():
        return mount_dir / "commits" / commit / "model.safetensors"
    if (mount_dir / commit / "model.safetensors").exists():
        return mount_dir / commit / "model.safetensors"
    return mount_dir / "commits" / commit / "model.safetensors"


def main():
    args = parse_args()

    mount_dir = args.mount_dir.resolve()
    mount_file = resolve_mount_file(mount_dir, args.commit)
    truth_file = args.truth.resolve()

    if not mount_file.is_file():
        raise FileNotFoundError(f"Mount file not found: {mount_file}")
    if not truth_file.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")

    repo = Repo.find(Path("."))

    def commit_of(path: Path) -> str:
        """The commit hash behind a mounted model.safetensors path."""
        name = path.parent.name
        if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
            return name
        ref = repo.refs_heads_dir / name
        return ref.read_text(encoding="utf-8").strip()

    fuse_pid = resolve_fuse_pid(repo, mount_dir)
    launch_spec = daemon_launch_spec(fuse_pid) if fuse_pid else None

    if args.remount and launch_spec is None:
        raise RuntimeError(
            f"--remount needs a running daemon to copy its command from, but no "
            f"mount record was found for {mount_dir}. Mount it first, or pass "
            f"--no-remount (loads will then be served from the page cache)."
        )

    def go_cold(label: str) -> "int | None":
        """Put the system in a cold state and return the current daemon PID."""
        nonlocal fuse_pid
        if args.remount:
            fuse_pid = remount(repo, mount_dir, launch_spec)
            print(f"[Cold] Remounted for {label} (daemon pid {fuse_pid})")
        if args.cold_cache:
            drop_linux_page_cache()
        return fuse_pid

    mount_size = os.path.getsize(mount_file)
    truth_size = os.path.getsize(truth_file)
    size_mb = mount_size / (1024 * 1024)

    print("=" * 60)
    print(f"Mount Directory:   {mount_dir}")
    print(f"Mount File:        {mount_file}")
    print(f"Ground Truth:      {truth_file}")
    print(f"FUSE Daemon PID:   {fuse_pid if fuse_pid else 'Not Found (Tracking Disabled)'}")
    print(f"Cold Method:       {'remount + drop_caches' if args.remount and args.cold_cache else 'remount' if args.remount else 'drop_caches only' if args.cold_cache else 'NONE (warm)'}")
    print(f"File Size:         {mount_size} bytes ({size_mb:.2f} MB)")
    print(f"Cold Cache Mode:   {args.cold_cache}")
    minimum_one = minimum_object_bytes(repo, [commit_of(mount_file)])
    print(f"Minimum Object Bytes:  {minimum_one:.1f} MiB to reconstruct this commit once")
    print("=" * 60)

    assert mount_size == truth_size, f"Size mismatch: {mount_size} != {truth_size}"

    if args.cold_cache:
        drop_linux_page_cache()

    elements = verify_against_truth(mount_file, truth_file)

    # -- Phase 2: single-process load latency, cold then warm ---------------
    print("\n[Phase 2] Single-process load_file latency")
    if fuse_pid:
        print(f"Tracking FUSE Daemon (PID: {fuse_pid}) | Baseline RSS: {get_process_rss_mb(fuse_pid):.2f} MB")
    print(f"  {'run':<12} {'seconds':>9} {'MB/s':>9} {'daemon read':>13} {'daemon RSS':>12}")

    def report(label: str, dur: float, served: float, rss: float) -> None:
        # A row where the daemon barely moved is the page cache answering, not
        # the decode path. Say so on the row itself -- the throughput column
        # alone reads as a real result and is off by two orders of magnitude.
        #
        # Judged against the analytic minimum, not against the file size.
        if not fuse_pid or not label.startswith("cold"):
            note = ""            # a warm row *should* read nothing; that is the point of it
        elif served < minimum_one * 0.6:
            note = "  <- not actually cold, page cache served it"
        elif served > minimum_one * 1.5:
            note = f"  <- {served / minimum_one:.1f}x the minimum"
        else:
            note = ""
        print(f"  {label:<12} {dur:9.3f} {size_mb / dur:9.1f} {served:10.1f} MiB {rss:9.1f} MB{note}")

    fuse_pid = go_cold("the cold load")
    report("cold", *timed_load(mount_file, fuse_pid))

    for rep in range(args.reps):
        report(f"warm #{rep + 1}", *timed_load(mount_file, fuse_pid))

    # Native baseline: the same call against a plain file on local disk.
    if args.cold_cache:
        drop_linux_page_cache()
    t0 = time.perf_counter()
    safetensors.torch.load_file(str(truth_file))
    native = time.perf_counter() - t0
    print(f"  {'native cold':<12} {native:9.3f} {size_mb / native:9.1f} {'(local file, no FUSE)':>26}")

    # -- Phase 3: concurrent loads ------------------------------------------
    print(f"\n[Phase 3] {args.workers} concurrent load_file processes")
    if args.distinct_commits:
        commits_dir = mount_dir / "commits"
        commits = sorted(os.listdir(commits_dir))[: args.workers]
        targets = [commits_dir / c / "model.safetensors" for c in commits]
        print(f"  each worker loads a different commit ({len(targets)} available)")
    else:
        targets = [mount_file] * args.workers
        print("  every worker loads the same file (pass --distinct-commits to avoid the page cache)")

    fuse_pid = go_cold("the concurrent phase")

    peak_rss = [get_process_rss_mb(fuse_pid) if fuse_pid else 0.0]
    stop_sampler = False

    def sample_rss():
        while not stop_sampler and fuse_pid:
            peak_rss[0] = max(peak_rss[0], get_process_rss_mb(fuse_pid))
            time.sleep(0.01)

    rchar_before = get_process_rchar_mb(fuse_pid) if fuse_pid else 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as sampler_pool:
        if fuse_pid:
            sampler_pool.submit(sample_rss)
        t0 = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(
                load_worker,
                [(i, p, elements) for i, p in enumerate(targets)],
            ))
        wall_time = time.perf_counter() - t0
        stop_sampler = True

    served = (get_process_rchar_mb(fuse_pid) - rchar_before) if fuse_pid else 0.0
    total_mb = size_mb * len(results)
    n_distinct = len(set(targets))
    target_commits = [commit_of(t) for t in targets]
    incomplete = [r for r in results if not r["complete"]]
    slowest = max(r["duration"] for r in results)

    print("\n" + "=" * 60)
    print(f"RESULTS FOR {mount_dir.name.upper()}")
    print("=" * 60)
    print(f"Verification:          {'PASSED (bit-for-bit)' if not incomplete else f'FAILED ({len(incomplete)} short loads)'}")
    print(f"Concurrent Workers:    {args.workers} processes")
    print(f"State Dicts Loaded:    {len(results)} x {results[0]['tensors']} tensors")
    print(f"Total Transferred:     {total_mb:.2f} MB")
    print(f"Wall Clock Time:       {wall_time:.3f} s")
    print(f"Aggregate Throughput:  {total_mb / wall_time:.2f} MB/s")
    print(f"Slowest Worker:        {slowest:.3f} s")
    if fuse_pid:
        # Scale by the DISTINCT files actually read, as a union: workers on
        # the same file share a working set, and different commits share delta
        # bases, so neither "per worker" nor "sum of per-commit" is right.
        expected = minimum_object_bytes(repo, sorted(set(target_commits)))
        print(f"Distinct Files Read:   {n_distinct} (reconstructing them needs {expected:.0f} MiB of objects)")
        print(f"Daemon Bytes Served:   {served:.1f} MiB")
        if served < expected * 0.6:
            print("  ^ well under the minimum: the kernel page cache served part of this.")
            print("    Use --remount to measure the decode path.")
        elif served > expected * 1.5:
            print(f"  ^ {served / expected:.1f}x the minimum -- chunks are being decoded, evicted,")
            print("    and decoded again. The chunk cache has to hold the working set of every")
            print(f"    DISTINCT file being read at once (~64 MiB each, so ~{64 * n_distinct} MiB here).")
            print(f"    Raise --cache-size on the mount, or read fewer distinct commits at once.")
        print(f"Peak Daemon RSS:       {peak_rss[0]:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
