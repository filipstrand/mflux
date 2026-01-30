# Lazy imports to avoid circular dependencies

__all__ = ["Plotter", "Statistics", "TrainingProfiler", "NullProfiler", "create_profiler"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Plotter":
        from mflux.models.z_image.variants.training.statistics.plotter import Plotter

        return Plotter
    if name == "Statistics":
        from mflux.models.z_image.variants.training.statistics.statistics import Statistics

        return Statistics
    if name == "TrainingProfiler":
        from mflux.models.z_image.variants.training.statistics.profiler import TrainingProfiler

        return TrainingProfiler
    if name == "NullProfiler":
        from mflux.models.z_image.variants.training.statistics.profiler import NullProfiler

        return NullProfiler
    if name == "create_profiler":
        from mflux.models.z_image.variants.training.statistics.profiler import create_profiler

        return create_profiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
