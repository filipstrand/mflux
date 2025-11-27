from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil
from mflux.utils.exceptions import PromptFileReadError


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate FIBO JSON prompts from images using VLM.")
    # fmt: off
    parser.add_argument("--image-path", type=Path, required=True, help="Path to image file to inspire from")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt to blend with the image (e.g., 'Make futuristic')")
    parser.add_argument("--output", type=Path, default=Path("inspired.json"), help="Output path for generated JSON prompt (default: inspired.json)")
    parser.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8], default=None, help="Quantize the VLM model (3, 4, 5, 6, or 8 bits)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling for VLM (default: 0.9)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for VLM (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for VLM generation (default: 4096)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for VLM generation")
    # fmt: on
    args = parser.parse_args()

    try:
        # 1. Load the image
        image = FiboVLMUtil.load_image(args.image_path)

        # 2. Generate JSON prompt from image
        vlm = FiboVLM(quantize=args.quantize)
        inspired_json = vlm.inspire(
            seed=args.seed,
            image=image,
            prompt=args.prompt,
            top_p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # 3. Parse and save generated JSON prompt
        FiboVLMUtil.save_json_prompt(args.output, inspired_json)
    except (PromptFileReadError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
