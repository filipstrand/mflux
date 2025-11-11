import gc
import json

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import PromptUtils
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using FIBO model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    json_prompt = _get_json_prompt(args)

    # 1. Load the FIBO model
    fibo = FIBO(
        model_config=ModelConfig.fibo(),
        quantize=args.quantize,
        local_path=args.path,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, model=fibo)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = fibo.generate_image(
                seed=seed,
                prompt=json_prompt,
                negative_prompt=PromptUtils.get_effective_negative_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                    scheduler="flow_match_euler_discrete",
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _get_json_prompt(args):
    prompt = PromptUtils.get_effective_prompt(args)

    try:
        json.loads(prompt)
        json_prompt = prompt
    except json.JSONDecodeError:
        vlm = FiboVLM()
        json_prompt = vlm.generate(prompt=prompt, seed=42)
        del vlm
        gc.collect()
        mx.clear_cache()
    return json_prompt


if __name__ == "__main__":
    main()
