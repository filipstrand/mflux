from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of a model to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Determine model class based on model name
    model_name_lower = args.model.lower()
    if "qwen" in model_name_lower:
        model_class = QwenImage
    elif "fibo" in model_name_lower:
        model_class = FIBO
    elif "z-image" in model_name_lower or "zimage" in model_name_lower:
        model_class = ZImageTurbo
    elif "flux2" in model_name_lower or "flux.2" in model_name_lower:
        model_class = Flux2Klein
    else:
        model_class = Flux1

    # 2. Load, quantize and save the model
    model = model_class(
        quantize=args.quantize,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        model_config=ModelConfig.from_name(args.model, base_model=args.base_model),
    )
    model.save_model(args.path)


if __name__ == "__main__":
    main()
