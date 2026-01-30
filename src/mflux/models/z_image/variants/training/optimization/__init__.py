# Lazy imports to avoid circular dependencies

__all__ = [
    "AdaptiveSynchronizer",
    "BFloat16AdamW",
    "CompiledTrainStep",
    "CosineAnnealingLR",
    "DeferredSynchronizer",
    "EMAModel",
    "LinearWarmupLR",
    "MemoryOptimizer",
    "MixedPrecisionAdam",
    "NoOpEMA",
    "OneCycleLR",
    "Optimizer",
    "ZImageLoss",
    "create_ema",
    "create_scheduler",
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
    if name == "EMAModel":
        from mflux.models.z_image.variants.training.optimization.ema import EMAModel

        return EMAModel
    if name == "NoOpEMA":
        from mflux.models.z_image.variants.training.optimization.ema import NoOpEMA

        return NoOpEMA
    if name == "create_ema":
        from mflux.models.z_image.variants.training.optimization.ema import create_ema

        return create_ema
    if name == "CosineAnnealingLR":
        from mflux.models.z_image.variants.training.optimization.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR
    if name == "LinearWarmupLR":
        from mflux.models.z_image.variants.training.optimization.lr_scheduler import LinearWarmupLR

        return LinearWarmupLR
    if name == "OneCycleLR":
        from mflux.models.z_image.variants.training.optimization.lr_scheduler import OneCycleLR

        return OneCycleLR
    if name == "create_scheduler":
        from mflux.models.z_image.variants.training.optimization.lr_scheduler import create_scheduler

        return create_scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
