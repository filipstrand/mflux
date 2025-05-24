from argparse import Namespace

from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.error.exceptions import PromptFileReadError
from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Default to a higher guidance value for fill related tasks.
    if args.guidance is None:
        args.guidance = 30

    # 1. Load the model
    flux = Flux1Fill(
        quantize=8,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = _register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=42,
                prompt="The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; [IMAGE1] Detailed product shot of a clothing; [IMAGE2] The same cloth is worn by a model in a lifestyle setting.",
                reference_garment_path="/Users/filipstrand/Desktop/garment.jpg",
                config=Config(
                    num_inference_steps=28,
                    height=1024,
                    width=768,
                    guidance=args.guidance,
                    image_path="/Users/filipstrand/Desktop/model.jpg",
                    masked_image_path="/Users/filipstrand/Desktop/mask.png",
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _register_callbacks(args: Namespace, flux: Flux1Fill) -> MemorySaver | None:
    # Battery saver
    battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
    CallbackRegistry.register_before_loop(battery_saver)

    # VAE Tiling
    if args.vae_tiling:
        flux.vae.decoder.enable_tiling = True
        flux.vae.decoder.split_direction = args.vae_tiling_split

    # Stepwise Handler
    if True:
        handler = StepwiseHandler(flux=flux, output_dir="/Users/filipstrand/Desktop/CATVTON")
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    # Memory Saver
    memory_saver = None
    if args.low_ram:
        memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
        CallbackRegistry.register_before_loop(memory_saver)
        CallbackRegistry.register_in_loop(memory_saver)
        CallbackRegistry.register_after_loop(memory_saver)
    return memory_saver


if __name__ == "__main__":
    main()
