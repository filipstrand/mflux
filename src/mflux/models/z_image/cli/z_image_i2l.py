"""CLI entrypoint for Z-Image Image-to-LoRA (i2L).

Usage:
    mflux-z-image-i2l --images img1.jpg img2.jpg --output style_lora.safetensors
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Generate LoRA weights from style reference images using Z-Image i2L.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mflux-z-image-i2l --images style1.jpg style2.jpg style3.jpg style4.jpg
  mflux-z-image-i2l --images photo.png --output my_style.safetensors
  mflux-z-image-i2l --images *.jpg --output style.safetensors

The generated LoRA can then be used with mflux-generate-z-image-turbo:
  mflux-generate-z-image-turbo --prompt "a cat" --lora-paths style.safetensors
        """,
    )
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        required=True,
        type=str,
        help="One or more style reference images to encode into LoRA weights.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="lora.safetensors",
        help="Output path for the generated LoRA file. Default: lora.safetensors",
    )

    args = parser.parse_args()

    # Validate images exist
    image_paths = []
    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"Error: Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        image_paths.append(p)

    # Load images
    print(f"Loading {len(image_paths)} image(s)...")
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
        print(f"  {p.name}: {img.size[0]}x{img.size[1]}")

    # Import here to avoid slow startup for --help
    from mflux.models.z_image.model.z_image_i2l.i2l_pipeline import ZImageI2LPipeline

    # Create pipeline and generate LoRA
    pipeline = ZImageI2LPipeline.from_pretrained()
    pipeline.generate_lora(images=images, output_path=args.output)


if __name__ == "__main__":
    main()
