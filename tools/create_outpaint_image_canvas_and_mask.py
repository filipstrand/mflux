import sys
from pathlib import Path

from mflux.post_processing.image_util import ImageUtil
from mflux.ui.box_values import AbsoluteBoxValues, BoxValues
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Create expanded canvas and mask for outpainting")
    parser.add_argument("image_path", type=Path, help="Path to the input image file")
    parser.add_image_outpaint_arguments(required=True)
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Input image not found: {args.image_path}")
        return 1

    try:
        # Load the image
        image = ImageUtil.load_image(args.image_path)
        # Get original dimensions
        orig_width, orig_height = image.width, image.height
        print(f"Loaded original image: {args.image_path} ({orig_width}x{orig_height})")

        # Calculate padding values
        padding: BoxValues = args.image_outpaint_padding
        print(f"{padding=}")

        # Create expanded canvas
        expanded_image = ImageUtil.expand_image(
            image=image,
            top=padding.top,
            right=padding.right,
            bottom=padding.bottom,
            left=padding.left,
        )

        abs_padding: AbsoluteBoxValues = args.image_outpaint_padding.normalize_to_dimensions(orig_width, orig_height)
        # Create mask image
        mask_image = ImageUtil.create_outpaint_mask_image(
            orig_width=orig_width,
            orig_height=orig_height,
            top=abs_padding.top,
            right=abs_padding.right,
            bottom=abs_padding.bottom,
            left=abs_padding.left,
        )

        # Construct output paths
        expanded_output_path = args.image_path.with_stem(f"{args.image_path.stem}_expanded")
        mask_output_path = args.image_path.with_stem(f"{args.image_path.stem}_mask")

        # Save the images
        ImageUtil.save_image(expanded_image, expanded_output_path)
        ImageUtil.save_image(mask_image, mask_output_path)

        print(f"Saved expanded canvas: {expanded_output_path}")
        print(f"Saved mask image: {mask_output_path}")

        return 0

    except Exception as e:  # noqa: BLE001
        print(f"Error processing image: {e}")
        raise e


if __name__ == "__main__":
    sys.exit(main())
