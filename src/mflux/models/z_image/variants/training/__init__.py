# Z-Image Training Module
# Provides native MLX training for Z-Image-Base with LoRA and full fine-tuning support

# Use lazy imports to avoid circular dependencies
# Import these classes directly when needed:
#   from mflux.models.z_image.variants.training.trainer import ZImageTrainer
#   from mflux.models.z_image.variants.training.training_initializer import ZImageTrainingInitializer


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "ZImageTrainer":
        from mflux.models.z_image.variants.training.trainer import ZImageTrainer

        return ZImageTrainer
    if name == "ZImageTrainingInitializer":
        from mflux.models.z_image.variants.training.training_initializer import ZImageTrainingInitializer

        return ZImageTrainingInitializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ZImageTrainer",
    "ZImageTrainingInitializer",
]
