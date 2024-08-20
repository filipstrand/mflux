import os
import re
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.config.config import ConfigImg2Img
from flux_1.flux import Flux1Img2Img
from flux_1.post_processing.image_util import ImageUtil


def main():
    parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The textual description of the image to generate.')
    parser.add_argument('--base-image', type=str, required=True, help='The name of the base image to use.')
    parser.add_argument('--output', type=str, default="image.png", help='The filename for the output image. Default is "image.png".')
    parser.add_argument('--model', type=str, default="schnell", help='The model to use ("schnell" or "dev"). Default is "schnell".')
    parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
    parser.add_argument('--height', type=int, default=1024, help='Image height (Default is 1024)')
    parser.add_argument('--width', type=int, default=1024, help='Image width (Default is 1024)')
    parser.add_argument('--steps', type=int, default=4, help='Inference Steps')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')
    parser.add_argument('--strength', type=float, default=0.5, help='Regulates much the final image should be influenced by base image. Default is 0.5.')

    args = parser.parse_args()

    seed = int(time.time()) if args.seed is None else args.seed

    flux_img2img = Flux1Img2Img.from_alias(args.model)
    base_image = ImageUtil.load_image(args.base_image)

    image = flux_img2img.generate_image(
        seed=seed,
        prompt=args.prompt,
        base_image=base_image,
        config=ConfigImg2Img(
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=args.guidance,
            strength=args.strength
        )
    )

    ImageUtil.save_image(image, args.output)


if __name__ == '__main__':
    main()
