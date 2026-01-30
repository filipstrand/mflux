# Lazy imports to avoid circular dependencies

__all__ = [
    "AdaptiveSynchronizer",
    "BFloat16AdamW",
    "CompiledTrainStep",
    "DeferredSynchronizer",
    "MemoryOptimizer",
    "MixedPrecisionAdam",
    "Optimizer",
    "ZImageLoss",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "AdaptiveSynchronizer":
        from mflux.models.z_image.variants.training.optimization.deferred_sync import AdaptiveSynchronizer

        return AdaptiveSynchronizer
    if name == "BFloat16AdamW":
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import BFloat16AdamW

        return BFloat16AdamW
    if name == "CompiledTrainStep":
        from mflux.models.z_image.variants.training.optimization.compiled_train_step import CompiledTrainStep

        return CompiledTrainStep
    if name == "DeferredSynchronizer":
        from mflux.models.z_image.variants.training.optimization.deferred_sync import DeferredSynchronizer

        return DeferredSynchronizer
    if name == "MemoryOptimizer":
        from mflux.models.z_image.variants.training.optimization.memory_optimizer import MemoryOptimizer

        return MemoryOptimizer
    if name == "MixedPrecisionAdam":
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import MixedPrecisionAdam

        return MixedPrecisionAdam
    if name == "Optimizer":
        from mflux.models.z_image.variants.training.optimization.optimizer import Optimizer

        return Optimizer
    if name == "ZImageLoss":
        from mflux.models.z_image.variants.training.optimization.z_image_loss import ZImageLoss

        return ZImageLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
