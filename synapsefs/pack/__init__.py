"""Packfile container and index.

Chunks are grouped into immutable, sealed packfiles with an mmap-able index, per
docs/FileFormat.md sections 5 and 6.

Owner: Compression team. See docs/TeamInstructions.md section A.
"""
