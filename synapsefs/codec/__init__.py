"""Residual codec.

Turns a pair of aligned tensors into bytes and back, exactly. Every operation is
integer arithmetic on raw bit patterns; no floating-point op occurs anywhere in
this package, which is what makes byte-exact reconstruction unconditional.

Owner: Compression team. See old_docs/TeamInstructions.md section A.
"""
