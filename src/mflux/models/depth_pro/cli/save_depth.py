from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.depth_pro.model.depth_pro import DepthPro


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a depth map of an input image")
    parser.add_save_depth_arguments()
    args = parser.parse_args()

    # 1. Create the depth map
    depth_pro = DepthPro(quantize=args.quantize)
    depth_result = depth_pro.create_depth_map(image_path=args.image_path)

    # 2. Save the depth map with the same name + _depth suffix
    image_path = Path(args.image_path)
    output_path = image_path.with_stem(f"{image_path.stem}_depth").with_suffix(".png")
    depth_result.depth_image.save(output_path)
    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()
