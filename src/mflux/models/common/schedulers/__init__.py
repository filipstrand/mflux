from .flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from .linear_scheduler import LinearScheduler
from .seedvr2_euler_scheduler import SeedVR2EulerScheduler

__all__ = [
    "LinearScheduler",
    "FlowMatchEulerDiscreteScheduler",
    "SeedVR2EulerScheduler",
]


class SchedulerModuleNotFound(ValueError): ...


class SchedulerClassNotFound(ValueError): ...


class InvalidSchedulerType(TypeError): ...


SCHEDULER_REGISTRY = {
    "linear": LinearScheduler,
    "LinearScheduler": LinearScheduler,
    "flow_match_euler_discrete": FlowMatchEulerDiscreteScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "seedvr2_euler": SeedVR2EulerScheduler,
    "SeedVR2EulerScheduler": SeedVR2EulerScheduler,
}


def register_contrib(scheduler_object, scheduler_name=None):
    if scheduler_name is None:
        scheduler_name = scheduler_object.__name__
    SCHEDULER_REGISTRY[scheduler_name] = scheduler_object


def try_import_external_scheduler(scheduler_object_path: str):
    import importlib
    import inspect

    from .base_scheduler import BaseScheduler

    try:
        last_dot_index = scheduler_object_path.rfind(".")

        if last_dot_index < 0:
            raise SchedulerModuleNotFound(
                f"Invalid scheduler path format: {scheduler_object_path!r}. "
                "Expected format: some_library.some_package.maybe_sub_package.YourScheduler"
            )

        module_name_str = scheduler_object_path[:last_dot_index]
        scheduler_class_name = scheduler_object_path[last_dot_index + 1 :]
        module = importlib.import_module(module_name_str)
    except ImportError:
        raise SchedulerModuleNotFound(scheduler_object_path)

    try:
        # Step 2: Get the object from the module using its string name
        SchedulerClass = getattr(module, scheduler_class_name)
    except AttributeError:
        raise SchedulerClassNotFound(scheduler_object_path)

    # Step 3: Validate that it's a class and inherits from BaseScheduler
    if not inspect.isclass(SchedulerClass):
        raise InvalidSchedulerType(
            f"{scheduler_object_path!r} is not a class. Schedulers must be classes inheriting from BaseScheduler."
        )

    if not issubclass(SchedulerClass, BaseScheduler):
        raise InvalidSchedulerType(
            f"{scheduler_object_path!r} does not inherit from BaseScheduler. "
            f"All schedulers must inherit from mflux.models.common.schedulers.BaseScheduler."
        )

    return SchedulerClass
