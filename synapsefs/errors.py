"""Typed exceptions mapped 1:1 to SynapseFS's CLI exit codes (old_docs/CLI.md ~1.3).

Every layer of the system (store, align, codec, pack, fuse) should raise one
of these -- never call sys.exit() directly, and never let a bare Exception
escape uncaught to the CLI layer. cli/main.py is the *only* place that
catches these and translates them into a process exit code, via the
`exit_code` class attribute below.

This module is deliberately placed at the top level, with no imports from
anywhere else in synapsefs. `store` needs to raise these; `cli` needs to
catch them. If errors.py lived inside either package, the other would end up
importing from it and you'd get store -> cli -> store (or the reverse) --
a circular import. Keeping errors.py dependency-free avoids that entirely:
every other package points *into* it, it points into nothing.
"""


class SynapseError(Exception):
    """Base class for all modeled SynapseFS failures.

    Subclasses set `exit_code` to the value documented in CLI.md ~1.3. The
    base class defaults to 1 (generic ERROR) so that a forgotten/misused
    subclass still produces a sane exit code instead of an AttributeError
    inside main()'s exception handler.
    """

    exit_code = 1


class UsageError(SynapseError):
    """Bad arguments, unknown flag, missing operand, or (as with `init`) an
    operation that refuses to clobber something that already exists.
    CLI.md exit code 2.
    """

    exit_code = 2


class NoRepoError(SynapseError):
    """Raised when a command that requires a repository can't find
    `.synapse/` by walking upward from `-C/--repo`. CLI.md exit code 3.
    """

    exit_code = 3


class IntegrityError(SynapseError):
    """Hash mismatch, corrupt object, tamper detected, or a referenced
    object that turns out to be missing entirely.

    Reserved *exclusively* for verification-class failures -- CLI.md ~1.3 is
    explicit that graders script against this code, so it must never be
    reused for ordinary I/O errors (those are generic ERROR, exit 1).
    CLI.md exit code 4.
    """

    exit_code = 4


class NotAlignableError(SynapseError):
    """Raised under `commit --strict` when a tensor pair is not meaningfully
    alignable (PS module 1h / 1d). CLI.md exit code 5.
    """

    exit_code = 5


class ConflictError(SynapseError):
    """A merge could not be resolved automatically and needs a human.
    CLI.md exit code 6.
    """

    exit_code = 6


class NetworkError(SynapseError):
    """Transport failure, unreachable peer, or protocol error during
    push/pull/serve. CLI.md exit code 7.
    """

    exit_code = 7


class MountError(SynapseError):
    """FUSE mount or unmount failure. CLI.md exit code 8."""

    exit_code = 8


class ObjectNotFoundError(IntegrityError):
    """A specific object hash was looked up in the store and is not present.

    Subclasses IntegrityError rather than introducing a new exit code:
    a missing object that some manifest or commit *expects* to exist is,
    definitionally, a lineage integrity failure -- the same class of
    problem `verify` walks the DAG to catch deliberately, just discovered
    here lazily, on demand, instead.
    """

    def __init__(self, object_hash: str):
        super().__init__(f"object not found: {object_hash}")
        self.object_hash = object_hash
