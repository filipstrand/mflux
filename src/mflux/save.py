import argparse

from mflux import Flux1, ModelConfig


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Save a quantized version of Flux.1 to disk.")
    parser.add_argument("--path", type=str, required=True, help="Local path for loading a model from disk")
    parser.add_argument("--model", "-m", type=str, required=True, choices=["dev", "schnell"], help="The model to use (\"schnell\" or \"dev\").")
    parser.add_argument("--quantize", "-q", type=int, choices=[4, 8], default=8, help="Quantize the model (4 or 8, Default is 8)")
    parser.add_argument("--lora-paths", type=str, nargs="*", default=None, help="Local safetensors for applying LORA from disk")
    parser.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")
    # fmt: on

    args = parser.parse_args()

    print(f"Saving model {args.model} with quantization level {args.quantize}\n")

    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    flux.save_model(args.path)

    print(f"Model saved at {args.path}\n")


if __name__ == "__main__":
    main()
