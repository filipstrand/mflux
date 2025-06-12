from mflux import Config, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_manager import CallbackManager
from mflux.community.concept_attention.flux_concept_from_image import FluxConceptFromImage
from mflux.error.exceptions import PromptFileReadError
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image with concept attention based on a prompt, concept, and reference image.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.add_concept_from_image_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # 1. Load the concept attention model
    flux = FluxConceptFromImage(
        model_config=ModelConfig.from_name(model_name=args.model, base_model=args.base_model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                concept=args.concept,
                image_path=str(args.input_image_path),
                heatmap_timesteps=args.heatmap_timesteps,
                heatmap_layer_indices=args.heatmap_layer_indices,
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                ),
            )
            # 4. Save the image and heatmap
            image.save_with_heatmap(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
