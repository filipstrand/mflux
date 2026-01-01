from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.utils.exceptions import StopImageGenerationException


def main():
    # 1. Parse command line arguments
    parser = CommandLineParser(description="Upscale an image using SeedVR2 diffusion-based super-resolution.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 3. Load the SeedVR2 model
    model = SeedVR2(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=ModelConfig.seedvr2_3b(),
    )

    # 4. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=SeedVR2LatentCreator,
    )

    try:
        # 5. Upscale the image for each seed
        for image_path in args.image_path:
            for seed in args.seed:
                result = model.generate_image(
                    seed=seed,
                    image_path=image_path,
                    resolution=args.resolution,
                    softness=args.softness,
                )

                # 6. Save result
                result.save(args.output.format(seed=seed, image_name=image_path.stem))
    except StopImageGenerationException as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
