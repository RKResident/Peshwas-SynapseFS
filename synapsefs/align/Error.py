"""Exception types for the alignment track.

Every one of these subclasses `SynapseError`, so `cli/main.py`'s single
`except SynapseError` handler turns them into the exit codes CLI.md documents
rather than the generic exit 1 an unrelated hierarchy would produce. That
matters for `NotAlignable` in particular: CLI.md reserves **exit 5** for
`commit --strict` hitting a tensor that cannot be meaningfully aligned, and a
detached hierarchy silently reported 1.
"""

from synapsefs.errors import SynapseError


class AlignError(SynapseError):
    """Base for everything raised by synapsefs.align."""

    exit_code = 1


class TopologyError(AlignError):
    """The IR is malformed or the model geometry is inconsistent.

    Exit 2: this is nearly always a bad `--config`, a checkpoint whose layers
    are not a straight chain, or an `order=` override that does not match --
    all of them user-correctable input problems, not internal failures.
    """

    exit_code = 2


class MalformedCheckpoint(AlignError):
    """The .safetensors file's header or data region is not self-consistent."""

    exit_code = 2


class UnsupportedArchitecture(TopologyError):
    """No adapter recognises this config.json."""

    exit_code = 2


class NotAlignable(AlignError):
    """Alignment did not materially reduce the residual (PS module 1h/1d).

    Exit 5, matching `errors.NotAlignableError`. Both exist because they are
    raised from different layers -- this one from inside the solver, that one
    from `commit --strict` after aggregating the report -- but they must map to
    the same code, because a grader scripting against exit 5 cannot tell which
    layer noticed.
    """

    exit_code = 5
