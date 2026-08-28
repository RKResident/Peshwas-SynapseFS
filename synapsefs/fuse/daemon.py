"""FUSE mount and daemon lifecycle management.

Handles foreground/background mount execution, trio event loop startup,
clean signal shutdown, and unmount fallback to fusermount3 (CLI.md ~11).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Optional, Set, Union

import blake3
import pyfuse3
import trio

from synapsefs.errors import MountError, UsageError
from synapsefs.fuse.fs import SynapseFSOperations
from synapsefs.store.repo import Repo


def _mount_id(mountpoint: Path) -> str:
    """Deterministic identifier for a resolved mountpoint."""
    return blake3.blake3(str(mountpoint.resolve()).encode("utf-8")).hexdigest()[:16]


def _mount_record_path(repo: Repo, mountpoint: Path) -> Path:
    mounts_dir = repo.synapse_dir / "mounts"
    mounts_dir.mkdir(parents=True, exist_ok=True)
    return mounts_dir / f"{_mount_id(mountpoint)}.json"


def record_mount(repo: Repo, mountpoint: Path, pid: int) -> None:
    path = _mount_record_path(repo, mountpoint)
    data = {
        "pid": pid,
        "mountpoint": str(mountpoint.resolve()),
        "repo": str(repo.root.resolve()),
        "time": time.time(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def remove_mount_record(repo: Repo, mountpoint: Path) -> None:
    path = _mount_record_path(repo, mountpoint)
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def find_mount_pid(repo: Repo, mountpoint: Path) -> Optional[int]:
    path = _mount_record_path(repo, mountpoint)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("pid")
        except Exception:
            return None
    return None


def run_fuse_loop(
    ops: SynapseFSOperations,
    mountpoint: Path,
    fuse_options: Set[str],
) -> None:
    """Initialize pyfuse3 and run the main trio event loop."""
    pyfuse3.init(ops, str(mountpoint), options=frozenset(fuse_options))
    try:
        trio.run(pyfuse3.main)
    finally:
        try:
            pyfuse3.close(unmount=True)
        except Exception:
            pass
        ops.close()


def mount_fuse(
    repo: Repo,
    mountpoint: Union[str, Path],
    *,
    ref: Optional[str] = None,
    foreground: bool = False,
    cache_size: int = 512 * 1024 * 1024,
    allow_other: bool = False,
    debug_fuse: bool = False,
) -> dict:
    """Mount a SynapseFS repository at `mountpoint`."""
    mount_path = Path(mountpoint).resolve()
    if not mount_path.exists():
        raise UsageError(f"mountpoint does not exist: {mount_path}")
    if not mount_path.is_dir():
        raise UsageError(f"mountpoint is not a directory: {mount_path}")

    # Build fuse options
    fuse_opts: Set[str] = {"default_permissions"}
    if allow_other:
        fuse_opts.add("allow_other")
    if debug_fuse:
        fuse_opts.add("debug")

    ops = SynapseFSOperations(
        repo=repo,
        ref_filter=ref,
        cache_size_bytes=cache_size,
    )

    if foreground:
        record_mount(repo, mount_path, os.getpid())
        try:
            run_fuse_loop(ops, mount_path, fuse_opts)
        finally:
            remove_mount_record(repo, mount_path)
        return {
            "repo": str(repo.root.resolve()),
            "mountpoint": str(mount_path),
            "cache_size": cache_size,
        }

    # Daemon mode: double-fork with pipe synchronization
    r_fd, w_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # Parent: wait for child to initialize or fail
        os.close(w_fd)
        with os.fdopen(r_fd, "rb") as r_pipe:
            status = r_pipe.read()
        if status != b"OK":
            err_msg = status.decode("utf-8", errors="replace") if status else "unknown error"
            raise MountError(f"failed to start FUSE daemon: {err_msg}")
        return {
            "repo": str(repo.root.resolve()),
            "mountpoint": str(mount_path),
            "cache_size": cache_size,
        }

    # Child (intermediate)
    os.close(r_fd)
    os.setsid()

    # Second fork to detach from session leader
    pid2 = os.fork()
    if pid2 > 0:
        # Intermediate process exits
        sys.exit(0)

    # Daemon child process
    # Redirect stdin/stdout/stderr
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    if not debug_fuse:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    os.close(devnull)

    record_mount(repo, mount_path, os.getpid())

    # Set up signal handling
    def _sig_handler(signum, frame):
        try:
            pyfuse3.close(unmount=True)
        except Exception:
            pass
        remove_mount_record(repo, mount_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        pyfuse3.init(ops, str(mount_path), options=frozenset(fuse_opts))
        # Notify parent that mount is ready
        os.write(w_fd, b"OK")
        os.close(w_fd)
    except Exception as exc:
        try:
            os.write(w_fd, str(exc).encode("utf-8"))
            os.close(w_fd)
        except Exception:
            pass
        remove_mount_record(repo, mount_path)
        sys.exit(1)

    try:
        trio.run(pyfuse3.main)
    finally:
        try:
            pyfuse3.close(unmount=True)
        except Exception:
            pass
        ops.close()
        remove_mount_record(repo, mount_path)
        sys.exit(0)


def unmount_fuse(
    repo: Repo,
    mountpoint: Union[str, Path],
) -> dict:
    """Unmount a mounted SynapseFS filesystem."""
    mount_path = Path(mountpoint).resolve()
    pid = find_mount_pid(repo, mount_path)

    unmounted = False

    # 1. Try stopping the daemon process gracefully
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 2 seconds for clean exit
            for _ in range(20):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    unmounted = True
                    break
        except OSError:
            # Process already dead
            pass

    # 2. Fall back to fusermount3 -u (or fusermount -u) if still mounted
    if not unmounted:
        cmd = None
        for binary in ("fusermount3", "fusermount", "umount"):
            try:
                res = subprocess.run(
                    [binary, "-u", str(mount_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                )
                if res.returncode == 0:
                    unmounted = True
                    break
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

    remove_mount_record(repo, mount_path)

    if not unmounted:
        # Check if the path is still mounted
        try:
            if not os.path.ismount(str(mount_path)):
                unmounted = True
        except Exception:
            pass

    if not unmounted:
        raise MountError(f"could not unmount {mount_path}")

    return {
        "mountpoint": str(mount_path),
    }
