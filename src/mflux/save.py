from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of a model to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Determine model class based on model name
    model_class = QwenImage if "qwen" in args.model.lower() else Flux1

    # 2. Load, quantize and save the model
    model = model_class(
        model_config=ModelConfig.from_name(args.model, base_model=args.base_model),
        quantize=args.quantize,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )
    model.save_model(args.path)


if __name__ == "__main__":
    main()
