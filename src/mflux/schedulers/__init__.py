from .linear_scheduler import LinearScheduler

__all__ = [
    "LinearScheduler",
]


class SchedulerModuleNotFound(ValueError): ...


class SchedulerClassNotFound(ValueError): ...


def try_import_external_scheduler(scheduler_object_path: str):
    import importlib

    try:
        last_dot_index = scheduler_object_path.rfind(".")

        if last_dot_index < 0:
            raise ValueError("Expected format: some_library.some_package.maybe_sub_package.YourScheduler")

        module_name_str = scheduler_object_path[:last_dot_index]
        scheduler_class_name = scheduler_object_path[last_dot_index + 1 :]
        module = importlib.import_module(module_name_str)
        print(f"Successfully imported module: '{module_name_str}'")
    except ImportError:
        raise SchedulerModuleNotFound(scheduler_object_path)

    try:
        # Step 2: Get the object from the module using its string name
        SchedulerClass = getattr(module, scheduler_class_name)
        print(f"Successfully found Scheduler class: '{SchedulerClass}'")
    except AttributeError:
        raise SchedulerClassNotFound(scheduler_object_path)

    return SchedulerClass
