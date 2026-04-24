from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.variants.ernie_image import ErnieImage
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an image using ERNIE-Image-Turbo (distilled, 8 steps) based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=False)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    # Set model alias so the parser can resolve the correct default step count (8)
    parser.set_defaults(model="ernie-image-turbo")
    args = parser.parse_args()

    model = ErnieImage(
        model_config=ModelConfig.ernie_image_turbo(),
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ErnieLatentCreator,
    )

    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
