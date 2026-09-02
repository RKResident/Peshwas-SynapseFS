"""Read-only virtual filesystem.

Reconstructs tensor regions on demand from base + residual blocks. Nothing is
ever pre-materialised to disk.

Owner: mmap/FUSE team. See old_docs/TeamInstructions.md section B.
"""

from synapsefs.fuse.cache import ChunkCache
from synapsefs.fuse.daemon import mount_fuse, unmount_fuse
from synapsefs.fuse.fs import SynapseFSOperations
from synapsefs.fuse.reconstruct import VirtualSafetensorsFile

__all__ = [
    "ChunkCache",
    "SynapseFSOperations",
    "VirtualSafetensorsFile",
    "mount_fuse",
    "unmount_fuse",
]
