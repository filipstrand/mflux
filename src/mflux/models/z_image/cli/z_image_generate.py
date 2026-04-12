import sys
from datetime import datetime
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def _nodot(val):
    s = f"{val:g}"
    return s.replace(".", "") if "." in s else s + "0"


def _build_saveinfo_filename(args, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sched_tag = getattr(args, "scheduler", "linear")
    sa_parts = []
    if getattr(args, "cosine", False):
        sa_parts.append("cos")
    if getattr(args, "karras", False):
        sa_parts.append("karras")
    if getattr(args, "exponential", False):
        sa_parts.append("exp")
    if getattr(args, "shift", None) is not None:
        sa_parts.append(f"shift_{_nodot(args.shift)}")
    if getattr(args, "mcf_max_change", None) is not None:
        sa_parts.append(f"mcf_{_nodot(args.mcf_max_change)}")
    sa = "_".join(sa_parts)
    lora_tag = "NoLora"
    if getattr(args, "lora_paths", None):
        names = [Path(p).stem for p in args.lora_paths]
        lora_tag = "+".join(names)
        if getattr(args, "lora_scales", None):
            sc = "+".join(_nodot(s) for s in args.lora_scales)
            lora_tag = f"{lora_tag}-{sc}"
    parts = [timestamp, str(seed), f"S{args.steps}", lora_tag, sched_tag]
    if sa:
        parts.append(sa)
    output_dir = str(Path(args.output).parent)
    return str(Path(output_dir) / ("_".join(parts) + ".png"))


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    if "--scheduler" not in sys.argv:
        args.scheduler = "flow_match_euler_discrete"

    model_name = args.model or "z-image"
    model_config = ModelConfig.from_name(model_name=model_name)

    # 1. Load the model
    model = ZImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageLatentCreator,
    )

    try:
        # Resolve dimensions
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                shift=args.shift,
                mcf_max_change=args.mcf_max_change,
                sigma_schedule=args.sigma_schedule,
            )
            # 4. Save the image
            output_path = _build_saveinfo_filename(args, seed) if args.saveinfo else args.output.format(seed=seed)
            image.save(path=output_path)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
