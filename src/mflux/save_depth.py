from pathlib import Path

from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a depth map of an input image")
    parser.add_save_depth_arguments()
    parser.add_argument("--transform", type=str, choices=["foreground", "background", "sigmoid", "log"], help="Apply non-linear transformation to depth map")  # fmt: off
    parser.add_argument("--strength", type=float, default=1.0, help="Strength of the transformation (default: 1.0)")
    args = parser.parse_args()

    # 1. Create the depth map
    depth_pro = DepthPro(quantize=args.quantize)
    depth_result = depth_pro.create_depth_map(image_path=args.image_path)

    # 2. Apply transformation if requested
    if hasattr(args, "transform") and args.transform:
        transformed_image = depth_result.apply_transformation(
            transform_type=args.transform,
            strength=args.strength,
        ).depth_image
    else:
        transformed_image = depth_result.depth_image

    # 3. Save the depth map with the same name + _depth suffix
    image_path = Path(args.image_path)
    transform_suffix = f"_{args.transform}{args.strength}" if hasattr(args, "transform") and args.transform else ""
    output_path = image_path.with_stem(f"{image_path.stem}_depth{transform_suffix}").with_suffix(".png")
    transformed_image.save(output_path)
    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()
