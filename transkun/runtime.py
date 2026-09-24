"""Optional runtime optimizations for TransKun."""

import torch


def maybe_compile_transformer(model, *, enabled=False, mode="default", backend="inductor"):
    """Compile only the Transformer blocks, leaving the rest of the model eager.

    Module.compile() changes the call path in place, so parameter identities and
    checkpoint state_dict keys stay the same. Dynamic shapes allow the same
    compiled layers to handle varying batch and sequence lengths where possible.
    """
    if not enabled:
        return model

    layers = getattr(getattr(model, "backbone", None), "encoderLayers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Expected model.backbone.encoderLayers to contain Transformer blocks")
    if not hasattr(torch.nn.Module, "compile"):
        raise RuntimeError("Transformer compilation requires PyTorch with nn.Module.compile")

    for layer in layers:
        layer.compile(backend=backend, mode=mode, fullgraph=False, dynamic=True)
    return model
