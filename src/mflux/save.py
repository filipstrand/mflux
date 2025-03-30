from mflux import Flux1, ModelConfig
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of Flux.1 to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Load, quantize and save the model
    flux = Flux1(
        model_config=ModelConfig.from_name(args.model, base_model=args.base_model),
        quantize=args.quantize,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )
    flux.save_model(args.path)


if __name__ == "__main__":
    main()
