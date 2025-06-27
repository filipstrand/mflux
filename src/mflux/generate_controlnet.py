from mflux import Config, Flux1Controlnet, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_manager import CallbackManager
from mflux.error.exceptions import PromptFileReadError
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments(mode="canny")
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=_get_controlnet_model_config(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux, enable_canny_saver=True)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                controlnet_image_path=args.controlnet_image_path,
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    controlnet_strength=args.controlnet_strength,
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _get_controlnet_model_config(model_name: str) -> ModelConfig:
    if model_name == "schnell":
        return ModelConfig.schnell_controlnet_canny()
    return ModelConfig.dev_controlnet_canny()


if __name__ == "__main__":
    main()
