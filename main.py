import os
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.config.model_config import ModelConfig
from flux_1.config.config import Config
from flux_1.flux import Flux1
from flux_1.post_processing.image_util import ImageUtil


def main():
    parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The textual description of the image to generate.')
    parser.add_argument('--output', type=str, default="image.png", help='The filename for the output image. Default is "image.png".')
    parser.add_argument('--model', "-m", type=str, required=True, choices=["dev", "schnell"], help='The model to use ("schnell" or "dev").')
    parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
    parser.add_argument('--height', type=int, default=1024, help='Image height (Default is 1024)')
    parser.add_argument('--width', type=int, default=1024, help='Image width (Default is 1024)')
    parser.add_argument('--steps', type=int, default=4, help='Inference Steps')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=None, help='Quantize the model (4 or 8, Default is None)')
    parser.add_argument('--path', type=str, default=None, help='Local path for loading a model from disk')

    args = parser.parse_args()

    if args.path and args.model is None:
        parser.error("--model must be specified when using --path")

    seed = int(time.time()) if args.seed is None else args.seed

    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize_full_weights=args.quantize,
        local_path=args.path
    )

    image = flux.generate_image(
        seed=seed,
        prompt=args.prompt,
        config=Config(
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=args.guidance,
        )
    )

    ImageUtil.save_image(image, args.output)


if __name__ == '__main__':
    main()
