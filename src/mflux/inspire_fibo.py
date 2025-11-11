import json
from pathlib import Path

from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.ui.cli.parsers import CommandLineParser
from mflux.utils.exceptions import PromptFileReadError


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate FIBO JSON prompts from images using VLM.")
    # fmt: off
    parser.add_argument("--image-path", type=Path, required=True, help="Path to image file to inspire from")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt to blend with the image (e.g., 'Make futuristic')")
    parser.add_argument("--output", type=Path, default=Path("inspired.json"), help="Output path for generated JSON prompt (default: inspired.json)")
    parser.add_argument("--path", type=str, default=None, help="Local path for loading the VLM model from disk")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling for VLM (default: 0.9)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for VLM (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for VLM generation (default: 4096)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for VLM generation")
    # fmt: on
    args = parser.parse_args()

    try:
        # 1. Load the image
        image = _load_image(args.image_path)

        # 2. Generate JSON prompt from image
        vlm = FiboVLM(local_path=args.path)
        inspired_json = vlm.inspire(
            image=image,
            prompt=args.prompt,
            top_p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        # 3. Parse and save generated JSON prompt
        _save_prompt(args, inspired_json)
    except (PromptFileReadError, ValueError) as exc:
        print(exc)


def _save_prompt(args, inspired_json):
    try:
        inspired_json_parsed = json.loads(inspired_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"VLM did not return valid JSON: {e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(inspired_json_parsed, f, indent=2, ensure_ascii=False)


def _load_image(image_path: Path):
    if not image_path.exists():
        raise PromptFileReadError(f"Image file does not exist: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, IOError, ValueError) as e:
        raise PromptFileReadError(f"Failed to load image: {e}")
    return image


if __name__ == "__main__":
    main()
