from mflux.flux_tools.depth.depth_util import DepthUtil
from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a depth map of an input image")
    parser.add_save_depth_arguments()
    args = parser.parse_args()

    # 1. Create and save the depth map
    depth_pro = DepthPro(quantize=args.quantize)
    DepthUtil.get_or_create_depth_map(depth_pro=depth_pro, image_path=args.image_path)


if __name__ == "__main__":
    main()
