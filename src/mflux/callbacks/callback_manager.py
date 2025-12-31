from argparse import Namespace

from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.depth_saver import DepthImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler


class CallbackManager:
    @staticmethod
    def register_callbacks(
        args: Namespace,
        model,
        latent_creator,
        enable_canny_saver: bool = False,
        enable_depth_saver: bool = False,
    ) -> MemorySaver | None:
        # Battery saver (always enabled)
        CallbackManager._register_battery_saver(args, model)

        # Specialized savers (based on flags)
        if enable_canny_saver:
            CallbackManager._register_canny_saver(args, model)

        if enable_depth_saver:
            CallbackManager._register_depth_saver(args, model)

        # Stepwise handler (if requested)
        CallbackManager._register_stepwise_handler(args, model, latent_creator)

        # Memory saver (if requested)
        return CallbackManager._register_memory_saver(args, model)

    @staticmethod
    def _register_battery_saver(args: Namespace, model) -> None:
        battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
        model.callbacks.register(battery_saver)

    @staticmethod
    def _register_canny_saver(args: Namespace, model) -> None:
        if hasattr(args, "controlnet_save_canny") and args.controlnet_save_canny:
            canny_image_saver = CannyImageSaver(path=args.output)
            model.callbacks.register(canny_image_saver)

    @staticmethod
    def _register_depth_saver(args: Namespace, model) -> None:
        if hasattr(args, "save_depth_map") and args.save_depth_map:
            depth_image_saver = DepthImageSaver(path=args.output)
            model.callbacks.register(depth_image_saver)

    @staticmethod
    def _register_stepwise_handler(args: Namespace, model, latent_creator) -> None:
        if args.stepwise_image_output_dir:
            handler = StepwiseHandler(
                model=model,
                latent_creator=latent_creator,
                output_dir=args.stepwise_image_output_dir,
            )
            model.callbacks.register(handler)

    @staticmethod
    def _register_memory_saver(args: Namespace, model) -> MemorySaver | None:
        memory_saver = None
        if args.low_ram:
            seeds = getattr(args, "seed", []) or []
            images = getattr(args, "image_path", [])
            if not isinstance(images, list):
                images = [images] if images is not None else []
            keep_transformer = len(seeds) > 1 or len(images) > 1
            memory_saver = MemorySaver(model=model, keep_transformer=keep_transformer, args=args)
            model.callbacks.register(memory_saver)
        return memory_saver
