"""CLI entrypoint for Z-Image Image-to-LoRA (i2L).

Usage:
    mflux-z-image-i2l --image-path ./style_images/ --output style_lora.safetensors
    mflux-z-image-i2l --image-path img1.jpg img2.jpg --output style_lora.safetensors
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _collect_images(paths: list[str]) -> list[Path]:
    """Resolve a mix of files and directories into a sorted list of image paths."""
    result = []
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            print(f"Error: Path not found: {p_str}", file=sys.stderr)
            sys.exit(1)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            if not found:
                print(f"Error: No images found in directory: {p_str}", file=sys.stderr)
                sys.exit(1)
            result.extend(found)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(p)
        else:
            print(f"Error: Unsupported file type: {p_str}", file=sys.stderr)
            sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate LoRA weights from style reference images using Z-Image i2L.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mflux-z-image-i2l --image-path ./my_style/
  mflux-z-image-i2l --image-path ./my_style/ --output my_style.safetensors
  mflux-z-image-i2l --image-path img1.jpg img2.jpg img3.jpg img4.jpg
  mflux-z-image-i2l --image-path ./style_a/ ./style_b/photo.png

The generated LoRA can then be used with mflux-generate-z-image-turbo:
  mflux-generate-z-image-turbo --prompt "a cat" --lora-paths style.safetensors
        """,
    )
    parser.add_argument(
        "--image-path",
        "-i",
        nargs="+",
        required=True,
        type=str,
        help="Image files or directories containing style reference images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="lora.safetensors",
        help="Output path for the generated LoRA file. Default: lora.safetensors",
    )

    args = parser.parse_args()

    # Collect image paths from files and directories
    image_paths = _collect_images(args.image_path)

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
