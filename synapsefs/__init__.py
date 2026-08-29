"""SynapseFS - permutation-aware version control for neural network checkpoints.

Package layout (see docs/TeamInstructions.md for ownership):

    synapsefs.align   Topology IR, permutation-group resolution, weight matching.
    synapsefs.codec   Residual encoding: monotone-int delta, zigzag, zstd.
    synapsefs.store   Content-addressed object store, refs, atomic writes.
    synapsefs.fuse    Read-only virtual mount.
    synapsefs.cli     Command-line entry points.

The on-disk contract these modules share is specified in docs/FileFormat.md.
Where this code and that document disagree, the document is wrong and should be
updated -- do not work around it in code.
"""

__version__ = "0.1.0"
