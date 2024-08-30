import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.flux import Flux1
from flux_1.config.model_config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description='Save a quantized version of Flux.1 to disk.')
    parser.add_argument('--path', type=str, required=True, help='Local path for loading a model from disk')
    parser.add_argument('--model', "-m", type=str, required=True, choices=["dev", "schnell"], help='The model to use ("schnell" or "dev").')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=8, help='Quantize the model (4 or 8, Default is 8)')

    args = parser.parse_args()

    print(f"Saving model {args.model} with quantization level {args.quantize}\n")

    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize_full_weights=args.quantize,
    )

    flux.save_model(args.path)

    print(f"Model saved at {args.path}\n")


if __name__ == '__main__':
    main()
