from argparse import Namespace

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.depth_saver import DepthImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler


class CallbackManager:
    @staticmethod
    def register_callbacks(
        args: Namespace,
        flux,
        enable_canny_saver: bool = False,
        enable_depth_saver: bool = False,
    ) -> MemorySaver | None:
        # Battery saver (always enabled)
        CallbackManager._register_battery_saver(args)

        # VAE Tiling (if requested)
        CallbackManager._register_vae_tiling(args, flux)

        # Specialized savers (based on flags)
        if enable_canny_saver:
            CallbackManager._register_canny_saver(args)

        if enable_depth_saver:
            CallbackManager._register_depth_saver(args)

        # Stepwise handler (if requested)
        CallbackManager._register_stepwise_handler(args, flux)

        # Memory saver (if requested)
        return CallbackManager._register_memory_saver(args, flux)

    @staticmethod
    def _register_battery_saver(args: Namespace) -> None:
        battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
        CallbackRegistry.register_before_loop(battery_saver)

    @staticmethod
    def _register_vae_tiling(args: Namespace, flux) -> None:
        if args.vae_tiling:
            flux.vae.decoder.enable_tiling = True
            flux.vae.decoder.split_direction = args.vae_tiling_split

    @staticmethod
    def _register_canny_saver(args: Namespace) -> None:
        if hasattr(args, "controlnet_save_canny") and args.controlnet_save_canny:
            canny_image_saver = CannyImageSaver(path=args.output)
            CallbackRegistry.register_before_loop(canny_image_saver)

    @staticmethod
    def _register_depth_saver(args: Namespace) -> None:
        if hasattr(args, "save_depth_map") and args.save_depth_map:
            depth_image_saver = DepthImageSaver(path=args.output)
            CallbackRegistry.register_before_loop(depth_image_saver)

    @staticmethod
    def _register_stepwise_handler(args: Namespace, flux) -> None:
        if args.stepwise_image_output_dir:
            handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
            CallbackRegistry.register_before_loop(handler)
            CallbackRegistry.register_in_loop(handler)
            CallbackRegistry.register_interrupt(handler)

    @staticmethod
    def _register_memory_saver(args: Namespace, flux) -> MemorySaver | None:
        memory_saver = None
        if args.low_ram:
            memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
            CallbackRegistry.register_before_loop(memory_saver)
            CallbackRegistry.register_in_loop(memory_saver)
            CallbackRegistry.register_after_loop(memory_saver)
        return memory_saver
