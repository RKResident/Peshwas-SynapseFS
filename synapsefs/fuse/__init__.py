"""Read-only virtual filesystem.

Reconstructs tensor regions on demand from base + residual blocks. Nothing is
ever pre-materialised to disk.

Owner: mmap/FUSE team. See docs/TeamInstructions.md section B.
"""
