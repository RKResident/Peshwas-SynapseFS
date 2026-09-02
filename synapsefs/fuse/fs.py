"""FUSE read-only virtual filesystem implementation using pyfuse3 and trio.

Maps commits and branches to virtual .safetensors files that decode tensor rows
on demand without disk pre-materialization (docs/CLI.md ~11, docs/TeamInstructions.md ~B).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import stat
import time
from typing import Dict, List, Optional, Tuple, Union

import pyfuse3
import trio

from synapsefs.fuse.cache import ChunkCache
from synapsefs.fuse.reconstruct import VirtualSafetensorsFile
from synapsefs.graph import CommitCheckpoint, ancestors
from synapsefs.store.repo import Repo


class InodeInfo:
    """Metadata and resolution context for an allocated virtual inode."""

    def __init__(
        self,
        inode: int,
        name: str,
        parent_inode: int,
        is_dir: bool,
        commit_hash: Optional[str] = None,
        branch_name: Optional[str] = None,
        vfile: Optional[VirtualSafetensorsFile] = None,
    ) -> None:
        self.inode = inode
        self.name = name
        self.parent_inode = parent_inode
        self.is_dir = is_dir
        self.commit_hash = commit_hash
        self.branch_name = branch_name
        self.vfile = vfile
        self.lookup_count = 0


class SynapseFSOperations(pyfuse3.Operations):
    """Read-only FUSE filesystem implementation for SynapseFS."""

    def __init__(
        self,
        repo: Repo,
        ref_filter: Optional[str] = None,
        cache_size_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.repo = repo
        self.ref_filter = ref_filter
        self.cache = ChunkCache(cache_size_bytes)
        # Every read is offloaded to a trio worker thread, and each in-flight
        # decode holds a working set several times the chunk size (compressed
        # payload, decompressed stream, unshuffled array -- and the same again
        # for each level of the delta chain). Trio's default limiter is 40,
        # which multiplies that transient by 40 and is what actually drives
        # daemon RSS -- not the bounded chunk cache. The numpy half of the
        # decode holds the GIL anyway, so extra threads buy no parallelism.
        self.read_threads = int(os.environ.get("SYNAPSEFS_READ_THREADS", "8"))
        self._read_limiter = trio.CapacityLimiter(self.read_threads)
        # Chunks are loose, content-addressed objects (ARCHITECTURE.md 3.3),
        # so there is no pack set to open, hold open, or close -- the object
        # store is reached through `repo.store` directly.

        self._next_inode = 3
        self._next_fh = 100
        self._inodes: Dict[int, InodeInfo] = {}
        # (parent_inode, name) -> inode. Without it _alloc_inode scans every
        # allocated inode on every lookup and readdir entry, which is quadratic
        # in the number of commits a mount has touched.
        self._by_name: Dict[Tuple[int, str], int] = {}
        self._fh_to_vfile: Dict[int, VirtualSafetensorsFile] = {}
        self._vfile_cache: Dict[str, VirtualSafetensorsFile] = {}
        # `stat` needs a file size and nothing else. Building a whole
        # VirtualSafetensorsFile to get one would parse the header and keep a
        # segment table per commit, so `ls -l commits/` would do that for every
        # commit in the listing. The size alone is cached instead.
        self._size_cache: Dict[str, int] = {}

        self._mount_time_ns = int(time.time() * 1e9)

        # Inode 1: Root
        self._inodes[pyfuse3.ROOT_INODE] = InodeInfo(
            inode=pyfuse3.ROOT_INODE,
            name="",
            parent_inode=pyfuse3.ROOT_INODE,
            is_dir=True,
        )

        # Inode 2: commits/
        self._commits_inode = 2
        self._inodes[self._commits_inode] = InodeInfo(
            inode=self._commits_inode,
            name="commits",
            parent_inode=pyfuse3.ROOT_INODE,
            is_dir=True,
        )

    def close(self) -> None:
        """Release cached state. Chunk files are opened per read, so there are
        no long-lived handles to release."""
        self.cache.clear()

    # -- Internal resolution helpers ---------------------------------------

    def _get_vfile(self, commit_hash: str) -> VirtualSafetensorsFile:
        """Get or lazily construct a VirtualSafetensorsFile for a commit."""
        if commit_hash in self._vfile_cache:
            return self._vfile_cache[commit_hash]

        checkpoint = CommitCheckpoint(self.repo.store, commit_hash)
        vfile = VirtualSafetensorsFile(checkpoint, cache=self.cache)
        self._vfile_cache[commit_hash] = vfile
        return vfile

    def _resolve_commit(self, commit_hash: str) -> Optional[VirtualSafetensorsFile]:
        try:
            return self._get_vfile(commit_hash)
        except Exception:
            return None

    def _get_branch_commit(self, branch: str) -> Optional[str]:
        branch_path = self.repo.refs_heads_dir / branch
        if branch_path.is_file():
            return branch_path.read_text(encoding="utf-8").strip()
        return None

    def _list_branches(self) -> List[str]:
        if not self.repo.refs_heads_dir.is_dir():
            return []
        branches = [
            p.name
            for p in self.repo.refs_heads_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]
        if self.ref_filter:
            # If ref_filter is a branch name, keep only that branch
            branches = [b for b in branches if b == self.ref_filter]
        return sorted(branches)

    def _list_commits(self) -> List[str]:
        """Every commit reachable from a branch tip, not just the tips.

        `lookup` has always resolved any commit hash, so the history was
        reachable by typing a path; it just was not listed, which made `ls`
        disagree with what `cd` would accept. Walking all parents rather than
        first-parent matters once merges exist: a merge's second parent is
        genuine history and would otherwise be invisible.
        """
        commits = set()
        for b in self._list_branches():
            tip = self._get_branch_commit(b)
            if not tip:
                continue
            commits.add(tip)
            try:
                commits.update(ancestors(self.repo.store, tip))
            except Exception:
                pass
        if self.ref_filter and len(self.ref_filter) >= 6:
            try:
                resolved = self.repo.resolve_ref(self.ref_filter)
                if resolved:
                    commits.add(resolved)
            except Exception:
                pass
        return sorted(commits)

    def _commit_size(self, commit_hash: str) -> int:
        """Byte size of a commit's virtual file, without building a vfile."""
        size = self._size_cache.get(commit_hash)
        if size is None:
            vfile = self._vfile_cache.get(commit_hash)
            if vfile is not None:
                size = vfile.total_size
            else:
                try:
                    size = CommitCheckpoint(self.repo.store, commit_hash).total_size
                except Exception:
                    size = 0
            self._size_cache[commit_hash] = size
        return size

    def _alloc_inode(
        self,
        name: str,
        parent_inode: int,
        is_dir: bool,
        commit_hash: Optional[str] = None,
        branch_name: Optional[str] = None,
        vfile: Optional[VirtualSafetensorsFile] = None,
    ) -> InodeInfo:
        existing = self._by_name.get((parent_inode, name))
        if existing is not None:
            info = self._inodes[existing]
            if commit_hash:
                info.commit_hash = commit_hash
            if branch_name:
                info.branch_name = branch_name
            if vfile:
                info.vfile = vfile
            return info

        inode = self._next_inode
        self._next_inode += 1
        info = InodeInfo(
            inode=inode,
            name=name,
            parent_inode=parent_inode,
            is_dir=is_dir,
            commit_hash=commit_hash,
            branch_name=branch_name,
            vfile=vfile,
        )
        self._inodes[inode] = info
        self._by_name[(parent_inode, name)] = inode
        return info

    def _looked_up(self, inode: int) -> pyfuse3.EntryAttributes:
        """Attributes for a reply that the kernel will count as a lookup."""
        info = self._inodes.get(inode)
        if info is not None:
            info.lookup_count += 1
        return self._get_entry_attrs(inode)

    def _get_entry_attrs(self, inode: int) -> pyfuse3.EntryAttributes:
        info = self._inodes.get(inode)
        if info is None:
            raise pyfuse3.FUSEError(errno.ENOENT)

        entry = pyfuse3.EntryAttributes()
        entry.st_ino = inode
        entry.generation = 1
        entry.entry_timeout = 1.0
        entry.attr_timeout = 1.0
        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_rdev = 0
        entry.st_atime_ns = self._mount_time_ns
        entry.st_mtime_ns = self._mount_time_ns
        entry.st_ctime_ns = self._mount_time_ns

        if info.is_dir:
            entry.st_mode = stat.S_IFDIR | 0o555
            entry.st_nlink = 2
            entry.st_size = 4096
        else:
            entry.st_mode = stat.S_IFREG | 0o444
            entry.st_nlink = 1
            # Size of the virtual safetensors file
            if info.vfile is not None:
                entry.st_size = info.vfile.total_size
            elif info.commit_hash:
                entry.st_size = self._commit_size(info.commit_hash)
            else:
                entry.st_size = 0

        return entry

    # -- pyfuse3.Operations API ---------------------------------------------

    async def lookup(
        self,
        parent_inode: int,
        name: bytes,
        ctx: pyfuse3.RequestContext,
    ) -> pyfuse3.EntryAttributes:
        """Lookup name in directory parent_inode."""
        name_str = name.decode("utf-8")
        parent = self._inodes.get(parent_inode)
        if parent is None or not parent.is_dir:
            raise pyfuse3.FUSEError(errno.ENOENT)

        # 1. Under Root (parent_inode == 1)
        if parent_inode == pyfuse3.ROOT_INODE:
            if name_str == "commits":
                return self._looked_up(self._commits_inode)

            # Check if it is a branch name
            if name_str in self._list_branches():
                commit_hash = self._get_branch_commit(name_str)
                info = self._alloc_inode(
                    name=name_str,
                    parent_inode=pyfuse3.ROOT_INODE,
                    is_dir=True,
                    branch_name=name_str,
                    commit_hash=commit_hash,
                )
                return self._looked_up(info.inode)

            raise pyfuse3.FUSEError(errno.ENOENT)

        # 2. Under commits/ (parent_inode == 2)
        if parent_inode == self._commits_inode:
            # Look up commit by hash or abbreviation
            commit_hash = None
            try:
                commit_hash = self.repo.resolve_ref(name_str)
            except Exception:
                pass

            if commit_hash is not None:
                info = self._alloc_inode(
                    name=name_str,
                    parent_inode=self._commits_inode,
                    is_dir=True,
                    commit_hash=commit_hash,
                )
                return self._looked_up(info.inode)

            raise pyfuse3.FUSEError(errno.ENOENT)

        # 3. Inside a Branch directory or Commit directory
        if parent.commit_hash or parent.branch_name:
            commit_hash = parent.commit_hash
            if parent.branch_name:
                commit_hash = self._get_branch_commit(parent.branch_name)

            if not commit_hash:
                raise pyfuse3.FUSEError(errno.ENOENT)

            # Check for .safetensors files (e.g. model.safetensors)
            if name_str.endswith(".safetensors"):
                vfile = self._resolve_commit(commit_hash)
                if vfile is None:
                    raise pyfuse3.FUSEError(errno.ENOENT)

                info = self._alloc_inode(
                    name=name_str,
                    parent_inode=parent_inode,
                    is_dir=False,
                    commit_hash=commit_hash,
                    vfile=vfile,
                )
                return self._looked_up(info.inode)

        raise pyfuse3.FUSEError(errno.ENOENT)

    async def getattr(
        self,
        inode: int,
        ctx: pyfuse3.RequestContext,
    ) -> pyfuse3.EntryAttributes:
        """Fetch attributes for inode."""
        return self._get_entry_attrs(inode)

    async def forget(self, inode_list) -> None:
        """Drop inodes the kernel is no longer referencing.

        Every reply to `lookup` increments a lookup count that only `forget`
        decrements. Not implementing it does not break correctness -- the
        entries stay valid -- but nothing is ever released, so a mount that
        walks many commits grows its inode table for the lifetime of the
        process. The permanent entries (root and `commits/`) are never
        dropped: the kernel can reference them again at any time without a
        fresh lookup."""
        permanent = {pyfuse3.ROOT_INODE, self._commits_inode}
        for inode, nlookup in inode_list:
            info = self._inodes.get(inode)
            if info is None or inode in permanent:
                continue
            info.lookup_count -= nlookup
            if info.lookup_count <= 0:
                self._inodes.pop(inode, None)
                self._by_name.pop((info.parent_inode, info.name), None)

    async def opendir(
        self,
        inode: int,
        ctx: pyfuse3.RequestContext,
    ) -> int:
        """Open directory inode."""
        info = self._inodes.get(inode)
        if info is None or not info.is_dir:
            raise pyfuse3.FUSEError(errno.ENOTDIR)
        return inode

    async def readdir(
        self,
        fh: int,
        start_id: int,
        token: pyfuse3.ReaddirToken,
    ) -> None:
        """Read directory entries for open directory fh."""
        info = self._inodes.get(fh)
        if info is None or not info.is_dir:
            return

        entries: List[Tuple[str, int, pyfuse3.EntryAttributes]] = []

        # Current dir (.) and Parent dir (..)
        dot_attr = self._get_entry_attrs(info.inode)
        parent_attr = self._get_entry_attrs(info.parent_inode)
        entries.append((".", info.inode, dot_attr))
        entries.append(("..", info.parent_inode, parent_attr))

        # Root listing
        if fh == pyfuse3.ROOT_INODE:
            commits_attr = self._get_entry_attrs(self._commits_inode)
            entries.append(("commits", self._commits_inode, commits_attr))

            for branch in self._list_branches():
                commit_hash = self._get_branch_commit(branch)
                b_info = self._alloc_inode(
                    name=branch,
                    parent_inode=pyfuse3.ROOT_INODE,
                    is_dir=True,
                    branch_name=branch,
                    commit_hash=commit_hash,
                )
                entries.append((branch, b_info.inode, self._get_entry_attrs(b_info.inode)))

        # commits/ listing
        elif fh == self._commits_inode:
            for commit in self._list_commits():
                c_info = self._alloc_inode(
                    name=commit,
                    parent_inode=self._commits_inode,
                    is_dir=True,
                    commit_hash=commit,
                )
                entries.append((commit, c_info.inode, self._get_entry_attrs(c_info.inode)))

        # Branch dir or Commit dir listing
        elif info.commit_hash or info.branch_name:
            commit_hash = info.commit_hash
            if info.branch_name:
                commit_hash = self._get_branch_commit(info.branch_name)

            if commit_hash:
                vfile = self._resolve_commit(commit_hash)
                if vfile:
                    f_info = self._alloc_inode(
                        name="model.safetensors",
                        parent_inode=fh,
                        is_dir=False,
                        commit_hash=commit_hash,
                        vfile=vfile,
                    )
                    entries.append(
                        ("model.safetensors", f_info.inode, self._get_entry_attrs(f_info.inode))
                    )

        # Emit entries starting after start_id
        for i, (name, _ino, attr) in enumerate(entries):
            entry_id = i + 1
            if entry_id <= start_id:
                continue
            if not pyfuse3.readdir_reply(token, name.encode("utf-8"), attr, entry_id):
                break

    async def open(
        self,
        inode: int,
        flags: int,
        ctx: pyfuse3.RequestContext,
    ) -> pyfuse3.FileInfo:
        """Open regular file inode."""
        info = self._inodes.get(inode)
        if info is None or info.is_dir:
            raise pyfuse3.FUSEError(errno.ENOENT if info is None else errno.EISDIR)

        # Reject write modes with EACCES (TeamInstructions section B)
        access_flags = flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)
        if access_flags in (os.O_WRONLY, os.O_RDWR) or (flags & (os.O_CREAT | os.O_TRUNC)):
            raise pyfuse3.FUSEError(errno.EACCES)

        vfile = info.vfile
        if vfile is None and info.commit_hash:
            vfile = self._resolve_commit(info.commit_hash)
            info.vfile = vfile

        if vfile is None:
            raise pyfuse3.FUSEError(errno.EIO)

        fh = self._next_fh
        self._next_fh += 1
        self._fh_to_vfile[fh] = vfile

        return pyfuse3.FileInfo(fh=fh, direct_io=False, keep_cache=True)

    async def read(
        self,
        fh: int,
        off: int,
        size: int,
    ) -> bytes:
        """Read data from open file handle fh.

        CPU-bound decompression and row reconstruction is offloaded from the
        Trio event loop into worker threads via trio.to_thread.run_sync.
        """
        vfile = self._fh_to_vfile.get(fh)
        if vfile is None:
            raise pyfuse3.FUSEError(errno.EBADF)

        # Offload decompression and row gather to thread pool
        return await trio.to_thread.run_sync(
            vfile.read, off, size, limiter=self._read_limiter
        )

    async def release(self, fh: int) -> None:
        """Release open file handle fh."""
        self._fh_to_vfile.pop(fh, None)

    async def releasedir(self, fh: int) -> None:
        """Release directory handle fh."""
        pass

    async def statfs(self, ctx: pyfuse3.RequestContext) -> pyfuse3.StatvfsData:
        """Return filesystem statistics."""
        stat_data = pyfuse3.StatvfsData()
        stat_data.f_bsize = 4096
        stat_data.f_frsize = 4096
        stat_data.f_blocks = 10_000_000
        stat_data.f_bfree = 10_000_000
        stat_data.f_bavail = 10_000_000
        stat_data.f_files = 1_000_000
        stat_data.f_ffree = 1_000_000
        stat_data.f_favail = 1_000_000
        stat_data.f_namemax = 255
        return stat_data

