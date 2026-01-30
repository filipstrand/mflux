# Lazy imports to avoid circular dependencies
# Import directly from submodules when needed

__all__ = [
    "BlockRange",
    "ExampleSpec",
    "FullFinetuneSpec",
    "InstrumentationSpec",
    "LoraLayersSpec",
    "OptimizerSpec",
    "SaveSpec",
    "StatisticsSpec",
    "TrainingLoopSpec",
    "TrainingMode",
    "TrainingSpec",
    "TrainingState",
    "ZImageTransformerBlocks",
    "ZipUtil",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in (
        "BlockRange",
        "ExampleSpec",
        "FullFinetuneSpec",
        "InstrumentationSpec",
        "LoraLayersSpec",
        "OptimizerSpec",
        "SaveSpec",
        "StatisticsSpec",
        "TrainingLoopSpec",
        "TrainingMode",
        "TrainingSpec",
        "ZImageTransformerBlocks",
    ):
        from mflux.models.z_image.variants.training.state import training_spec

        return getattr(training_spec, name)
    if name == "TrainingState":
        from mflux.models.z_image.variants.training.state.training_state import TrainingState

        return TrainingState
    if name == "ZipUtil":
        from mflux.models.z_image.variants.training.state.zip_util import ZipUtil

        return ZipUtil
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
