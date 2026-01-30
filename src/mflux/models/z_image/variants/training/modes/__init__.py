# Lazy imports to avoid circular dependencies

__all__ = ["FullTrainer", "LoRATrainer"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "FullTrainer":
        from mflux.models.z_image.variants.training.modes.full_trainer import FullTrainer

        return FullTrainer
    if name == "LoRATrainer":
        from mflux.models.z_image.variants.training.modes.lora_trainer import LoRATrainer

        return LoRATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
