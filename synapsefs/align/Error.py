"""Exception types for the alignment track."""


class AlignError(Exception):
    """Base for everything raised by synapsefs.align."""


class TopologyError(AlignError):
    """The IR is malformed or the model geometry is inconsistent."""


class MalformedCheckpoint(AlignError):
    """The .safetensors file's header or data region is not self-consistent."""


class UnsupportedArchitecture(TopologyError):
    """No adapter recognises this config.json."""


class NotAlignable(AlignError):
    """Alignment did not materially reduce the residual."""