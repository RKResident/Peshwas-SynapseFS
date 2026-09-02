import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
import random
import subprocess
import time
import psutil
import torch
import safetensors.torch
from synapsefs.fuse.daemon import find_mount_pid
from synapsefs.store.repo import Repo


def drop_linux_page_cache() -> bool:
    """Flush dirty pages and clear the Linux OS page cache via drop_caches."""
    print("[Cache] Dropping Linux page cache...")
    try:
        # Sync unwritten file system buffers first
        subprocess.run(["sync"], check=True)
        # Write 3 to /proc/sys/vm/drop_caches via sudo
        res = subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            print("✓ OS page cache cleared successfully.")
            return True
        else:
            print(f"⚠ Failed to drop caches: {res.stderr.strip()}")
            return False
    except Exception as exc:
        print(f"⚠ Could not drop OS page cache ({exc}). Continuing with existing cache...")
        return False


def get_process_rss_mb(pid: int) -> float:
    """Read Resident Set Size (RSS) in MB for a specific process."""
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def read_and_verify_worker(
    worker_id: int,
    mount_path: Path,
    truth_path: Path,
    file_size: int,
    read_size: int,
    reps: int,
) -> dict:
    """Worker task executing concurrent uniformly random reads and byte-level diffing."""
    bytes_read = 0
    mismatches = 0
    t_start = time.perf_counter()

    rng = random.Random(1337 + worker_id)
    max_offset = max(0, file_size - read_size)

    with open(mount_path, "rb") as f_mount, open(truth_path, "rb") as f_truth:
        for _ in range(reps):
            offset = rng.randint(0, max_offset)

            f_mount.seek(offset)
            buf_mount = f_mount.read(read_size)
            bytes_read += len(buf_mount)

            f_truth.seek(offset)
            buf_truth = f_truth.read(read_size)

            if buf_mount != buf_truth:
                mismatches += 1

    duration = time.perf_counter() - t_start
    return {
        "worker_id": worker_id,
        "bytes_read": bytes_read,
        "mismatches": mismatches,
        "duration": duration,
    }


def full_tensor_sanity_check(mount_path: Path, truth_path: Path):
    """Deep verification: loads entire state_dict and compares values."""
    print("\n[Phase 1] Running high-level tensor numerical comparison...")
    state_truth = safetensors.torch.load_file(str(truth_path))
    t0 = time.perf_counter()

    state_mount = safetensors.torch.load_file(str(mount_path))


    assert set(state_mount.keys()) == set(state_truth.keys()), "Tensor key sets do not match!"

    for key in state_truth.keys():
        t_mount = state_mount[key]
        t_truth = state_truth[key]

        assert t_mount.shape == t_truth.shape, f"Shape mismatch in {key}: {t_mount.shape} vs {t_truth.shape}"
        assert t_mount.dtype == t_truth.dtype, f"Dtype mismatch in {key}: {t_mount.dtype} vs {t_truth.dtype}"

        if not torch.equal(t_mount, t_truth):
            diff = (t_mount - t_truth).abs().max().item()
            raise ValueError(f"Data corruption detected in tensor '{key}'! Max diff: {diff}")

    dur = time.perf_counter() - t0
    print(f"✓ All {len(state_truth)} tensors match bit-for-bit with ground truth ({dur:.2f}s).")


def resolve_fuse_pid(repo: Repo, mount_dir: Path) -> int | None:
    """Identify the exact FUSE daemon PID for the specific mount directory."""
    pid = find_mount_pid(repo, mount_dir)
    if pid is not None:
        return pid

    # Fallback 1: fuser on the target mountpoint
    try:
        res = subprocess.run(
            ["fuser", str(mount_dir.resolve())],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        pids = [int(p) for p in res.stdout.decode().split() if p.isdigit() and int(p) != os.getpid()]
        if pids:
            return pids[0]
    except Exception:
        pass

    # Fallback 2: psutil search matching mount directory path
    mount_str = str(mount_dir.resolve())
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "synapsefs" in cmd and mount_str in cmd and proc.info["pid"] != os.getpid():
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="SynapseFS FUSE Benchmark & Verification Suite")
    parser.add_argument(
        "--mount-dir",
        type=Path,
        default=Path("mnt"),
        help="Root path of the active FUSE mount (e.g., mnt_a, mnt_b).",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="6ed842c9ae3b29982df6609237419befdfd594732034ac4816c5af81b37ec136",
        help="Commit hash or branch folder to evaluate.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("epoch24.safetensors"),
        help="Path to the original uncompressed ground-truth safetensors file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent reader processes.",
    )
    parser.add_argument(
        "--read-size-mb",
        type=int,
        default=4,
        help="Read block size in megabytes.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=20,
        help="Number of read iterations per worker.",
    )
    # Boolean flag: default True, pass --no-cold-cache to disable
    parser.add_argument(
        "--cold-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop OS page cache prior to running benchmarks (default: True).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mount_dir = args.mount_dir.resolve()
    if (mount_dir / "commits" / args.commit / "model.safetensors").exists():
        mount_file = mount_dir / "commits" / args.commit / "model.safetensors"
    elif (mount_dir / args.commit / "model.safetensors").exists():
        mount_file = mount_dir / args.commit / "model.safetensors"
    else:
        mount_file = mount_dir / "commits" / args.commit / "model.safetensors"

    truth_file = args.truth.resolve()
    read_size_bytes = args.read_size_mb * 1024 * 1024

    if not mount_file.is_file():
        raise FileNotFoundError(f"Mount file not found: {mount_file}")
    if not truth_file.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")

    # Optional cold cache clearance before anything touches disk
    if args.cold_cache:
        drop_linux_page_cache()

    repo = Repo.find(Path("."))
    fuse_pid = resolve_fuse_pid(repo, mount_dir)

    mount_size = os.path.getsize(mount_file)
    truth_size = os.path.getsize(truth_file)

    print("=" * 50)
    print(f"Mount Directory:   {mount_dir}")
    print(f"Mount File:        {mount_file}")
    print(f"FUSE Daemon PID:   {fuse_pid if fuse_pid else 'Not Found (Tracking Disabled)'}")
    print(f"Mount File Size:   {mount_size} bytes ({mount_size / (1024*1024):.2f} MB)")
    print(f"Truth File Size:   {truth_size} bytes ({truth_size / (1024*1024):.2f} MB)")
    print(f"Cold Cache Mode:   {args.cold_cache}")
    print("=" * 50)

    assert mount_size == truth_size, f"Size mismatch: {mount_size} != {truth_size}"

    # Sample RSS across the WHOLE run, not just phase 2.
    #
    # This used to start after phase 1, and phase 1 is where the peak actually
    # is: it loads both state dicts, so the daemon decodes the entire
    # checkpoint there. Measured on the 92M benchmark, 8 workers -- this
    # reported "210.32 MB (Delta: +0.00 MB)" against a true peak of 266.5 MB
    # sampled externally. A +0.00 delta reads as "this workload costs nothing",
    # which is exactly backwards.
    idle_rss = get_process_rss_mb(fuse_pid) if fuse_pid else 0.0
    peak_rss = [idle_rss]
    stop_sampler = False

    def sample_rss():
        while not stop_sampler and fuse_pid:
            rss = get_process_rss_mb(fuse_pid)
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            time.sleep(0.01)

    sampler_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if fuse_pid:
        sampler_pool.submit(sample_rss)

    try:
        # Phase 1: High-level verification
        full_tensor_sanity_check(mount_file, truth_file)

        # If running in cold-cache mode, clear page cache again between Phase 1
        # and Phase 2 so the concurrent read throughput starts strictly cold
        if args.cold_cache:
            drop_linux_page_cache()

        # Phase 2: Concurrent Benchmark
        print("\n[Phase 2] Starting Concurrency, RSS, and Throughput Evaluation")
        if fuse_pid:
            print(f"Tracking FUSE Daemon (PID: {fuse_pid}) | Idle RSS: {idle_rss:.2f} MB")

        t0 = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    read_and_verify_worker,
                    wid,
                    mount_file,
                    truth_file,
                    mount_size,
                    read_size_bytes,
                    args.reps,
                )
                for wid in range(args.workers)
            ]
            results = [f.result() for f in futures]

        wall_time = time.perf_counter() - t0
    finally:
        stop_sampler = True
        sampler_pool.shutdown(wait=True)

    total_bytes = sum(r["bytes_read"] for r in results)
    total_mismatches = sum(r["mismatches"] for r in results)
    total_mb = total_bytes / (1024 * 1024)
    throughput = total_mb / wall_time

    print("\n" + "=" * 50)
    print(f"RESULTS FOR {mount_dir.name.upper()}")
    print("=" * 50)
    print(f"Bitwise Verification:  {'PASSED (0 mismatches)' if total_mismatches == 0 else f'FAILED ({total_mismatches} mismatches)'}")
    print(f"Concurrent Workers:    {args.workers} processes")
    print(f"Total Transferred:     {total_mb:.2f} MB")
    print(f"Wall Clock Time:       {wall_time:.3f} s")
    print(f"Read Throughput:       {throughput:.2f} MB/s")
    if fuse_pid:
        print(f"Idle Daemon RSS:       {idle_rss:.2f} MB")
        print(f"Peak Daemon RSS:       {peak_rss[0]:.2f} MB "
              f"(Delta: +{peak_rss[0] - idle_rss:.2f} MB, whole run incl. phase 1)")
    print("=" * 50)


if __name__ == "__main__":
    main()