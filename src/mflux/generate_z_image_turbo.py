from mflux.callbacks.callback_manager import CallbackManager
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.z_image.variants.z_image_turbo.z_image_turbo import ZImageTurbo
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import PromptUtils
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image Turbo based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    model = ZImageTurbo(
        model_config=ModelConfig.z_image_turbo(),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, model=model)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtils.get_effective_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=None,  # Z-Image Turbo doesn't use guidance
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                ),
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
