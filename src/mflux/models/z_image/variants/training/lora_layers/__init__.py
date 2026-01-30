# Lazy imports to avoid circular dependencies

__all__ = ["ZImageLoRALayers"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "ZImageLoRALayers":
        from mflux.models.z_image.variants.training.lora_layers.lora_layers import ZImageLoRALayers

        return ZImageLoRALayers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
