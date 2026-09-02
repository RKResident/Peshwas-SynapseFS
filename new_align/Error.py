class AlignError(Exception):
    """Base for everything raised by synapsefs.align."""


class CheckpointError(AlignError):
    """The .safetensors file is malformed or its header does not add up."""


class TopologyError(AlignError):
    """The model's geometry is inconsistent, or it is not a shape we can wire."""