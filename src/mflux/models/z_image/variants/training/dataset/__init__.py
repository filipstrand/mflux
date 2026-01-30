# Lazy imports to avoid circular dependencies
# Import directly from submodules when needed:
#   from mflux.models.z_image.variants.training.dataset.batch import Batch, Example
#   from mflux.models.z_image.variants.training.dataset.dataset import Dataset
#   from mflux.models.z_image.variants.training.dataset.iterator import Iterator
#   from mflux.models.z_image.variants.training.dataset.preprocessing import ZImagePreProcessing

__all__ = [
    "Batch",
    "Dataset",
    "Example",
    "Iterator",
    "ZImagePreProcessing",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("Batch", "Example"):
        from mflux.models.z_image.variants.training.dataset.batch import Batch, Example

        return Batch if name == "Batch" else Example
    if name == "Dataset":
        from mflux.models.z_image.variants.training.dataset.dataset import Dataset

        return Dataset
    if name == "Iterator":
        from mflux.models.z_image.variants.training.dataset.iterator import Iterator

        return Iterator
    if name == "ZImagePreProcessing":
        from mflux.models.z_image.variants.training.dataset.preprocessing import ZImagePreProcessing

        return ZImagePreProcessing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
