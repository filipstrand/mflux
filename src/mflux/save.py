from mflux.config.model_config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.ui.cli.parsers import CommandLineParser
from mflux.zimage import ZImage


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
    elif "zimage" in model_name_lower or model_name_lower == "turbo":
        model_class = ZImage
    else:
        model_class = Flux1

    # 2. Load, quantize and save the model
    model_kwargs = {
        "model_config": ModelConfig.from_name(args.model, base_model=args.base_model),
        "quantize": args.quantize,
    }
    # LoRA support (not available for FIBO or ZImage)
    if args.lora_paths and model_class not in (FIBO, ZImage):
        model_kwargs["lora_paths"] = args.lora_paths
        model_kwargs["lora_scales"] = args.lora_scales

    model = model_class(**model_kwargs)
    model.save_model(args.path)


if __name__ == "__main__":
    main()
