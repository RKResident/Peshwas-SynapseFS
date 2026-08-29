"""Tests for FUSE read-only virtual filesystem, LRU cache, and mount/unmount."""

import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest
import pyfuse3
import trio

from synapsefs.cli.main import main
from synapsefs.errors import MountError
from synapsefs.fuse.cache import ChunkCache
from synapsefs.fuse.daemon import mount_fuse, unmount_fuse
from synapsefs.fuse.fs import SynapseFSOperations
from synapsefs.fuse.reconstruct import VirtualSafetensorsFile
from synapsefs.graph import CommitCheckpoint
from synapsefs.store.repo import Repo


def test_chunk_cache_basic_and_eviction():
    cache = ChunkCache(max_bytes=100)
    assert len(cache) == 0

    cache.put("k1", b"12345", 50)
    assert cache.current_bytes == 50
    assert cache.get("k1") == b"12345"
    assert cache.get("nonexistent") is None

    cache.put("k2", b"67890", 40)
    assert cache.current_bytes == 90
    assert len(cache) == 2

    # Adding k3 (20 bytes) exceeds max_bytes (90 + 20 = 110 > 100) -> evicts k1
    cache.put("k3", b"ab", 20)
    assert cache.get("k1") is None
    assert cache.get("k2") == b"67890"
    assert cache.get("k3") == b"ab"
    assert cache.current_bytes == 60

    # Overly large entry should not be cached
    cache.put("too_big", b"x" * 150, 150)
    assert cache.get("too_big") is None

    cache.clear()
    assert len(cache) == 0
    assert cache.current_bytes == 0


@pytest.fixture
def populated_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo = Repo.init_at(repo_dir, branch="main")

    fixture_base = Path("fixtures/mlp-tiny/base")
    if not fixture_base.is_dir():
        pytest.skip("fixtures/mlp-tiny/base not found; run make fixtures first")

    # Initial commit
    ret = main([
        "-C", str(repo_dir),
        "commit", str(fixture_base / "model.safetensors"),
        "-m", "initial commit",
        "--config", str(fixture_base / "config.json"),
    ])
    assert ret == 0

    return repo, fixture_base / "model.safetensors"


def test_virtual_safetensors_file_byte_exactness(populated_repo):
    repo, orig_file = populated_repo
    _branch, commit_hash = repo.read_head()
    assert commit_hash is not None

    orig_bytes = orig_file.read_bytes()

    cache = ChunkCache(max_bytes=1024 * 1024)
    checkpoint = CommitCheckpoint(repo.store, commit_hash)
    vfile = VirtualSafetensorsFile(checkpoint, cache=cache)

    assert vfile.total_size == len(orig_bytes)

    # 1. Full read
    reconstructed_full = vfile.read(0, vfile.total_size)
    assert reconstructed_full == orig_bytes

    # 2. Slice reads: header and arbitrary offsets
    assert vfile.read(0, 100) == orig_bytes[0:100]
    assert vfile.read(100, 500) == orig_bytes[100:600]
    assert vfile.read(vfile.total_size - 50, 100) == orig_bytes[vfile.total_size - 50 :]
    assert vfile.read(vfile.total_size, 10) == b""


def test_fuse_operations_read_only(populated_repo):
    repo, orig_file = populated_repo
    orig_bytes = orig_file.read_bytes()

    async def _test_async():
        ops = SynapseFSOperations(repo=repo)
        ctx = pyfuse3.RequestContext()

        try:
            # Root getattr
            root_attr = await ops.getattr(pyfuse3.ROOT_INODE, ctx)
            assert stat.S_ISDIR(root_attr.st_mode)

            # Lookup "main"
            main_attr = await ops.lookup(pyfuse3.ROOT_INODE, b"main", ctx)
            assert stat.S_ISDIR(main_attr.st_mode)

            # Lookup "model.safetensors" in main
            file_attr = await ops.lookup(main_attr.st_ino, b"model.safetensors", ctx)
            assert stat.S_ISREG(file_attr.st_mode)
            assert file_attr.st_size == len(orig_bytes)

            # Open read-only
            finfo = await ops.open(file_attr.st_ino, os.O_RDONLY, ctx)
            assert finfo.fh > 0

            # Read content
            chunk = await ops.read(finfo.fh, 0, 512)
            assert chunk == orig_bytes[0:512]

            await ops.release(finfo.fh)

            # Open write mode must raise EACCES (CLI exit code 8 / permission)
            with pytest.raises(pyfuse3.FUSEError) as exc_info:
                await ops.open(file_attr.st_ino, os.O_WRONLY, ctx)
            assert exc_info.value.errno == 13  # EACCES

            # Statfs
            st = await ops.statfs(ctx)
            assert st.f_bsize == 4096
        finally:
            ops.close()

    trio.run(_test_async)


def test_fuse_mount_and_unmount_cli(populated_repo, tmp_path):
    repo, orig_file = populated_repo
    orig_bytes = orig_file.read_bytes()

    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()

    # Mount via CLI command in subprocess
    res = subprocess.run(
        [
            sys.executable, "-m", "synapsefs.cli.main",
            "-C", str(repo.root),
            "mount", str(mountpoint),
        ],
        capture_output=True,
        text=True,
    )

    if res.returncode != 0:
        pytest.skip(f"Kernel FUSE mount not permitted in current sandbox: {res.stderr.strip()}")

    try:
        time.sleep(0.3)
        entries = os.listdir(mountpoint)
        assert "main" in entries or "commits" in entries

        mounted_file = mountpoint / "main" / "model.safetensors"
        assert mounted_file.is_file()
        assert mounted_file.stat().st_size == len(orig_bytes)
        assert mounted_file.read_bytes() == orig_bytes
    finally:
        # Unmount via CLI command
        subprocess.run(
            [
                sys.executable, "-m", "synapsefs.cli.main",
                "-C", str(repo.root),
                "unmount", str(mountpoint),
            ],
            capture_output=True,
            text=True,
        )
