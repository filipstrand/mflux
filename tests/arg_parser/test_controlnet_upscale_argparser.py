from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.utils.scale_factor import ScaleFactor


def _create_controlnet_upscale_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an upscaled image from a source image")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()

    # Add height/width with scale factor support (as in flux_upscale.py)
    parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
    parser.add_controlnet_arguments(require_image=True)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def controlnet_upscale_parser() -> CommandLineParser:
    return _create_controlnet_upscale_parser()


@pytest.fixture
def controlnet_upscale_minimal_argv() -> list[str]:
    return ["mflux-upscale-controlnet", "--prompt", "upscaled image", "--controlnet-image-path", "image.png"]


@pytest.mark.fast
def test_controlnet_upscale_with_all_arguments(controlnet_upscale_parser):
    full_argv = [
        "mflux-upscale-controlnet",
        "--prompt",
        "upscaled beautiful landscape",
        "--controlnet-image-path",
        "source.png",
        "--height",
        "2x",
        "--width",
        "1920",
        "--steps",
        "20",
        "--guidance",
        "7.5",
        "--controlnet-strength",
        "0.8",
        "--seed",
        "42",
        "--output",
        "upscaled.png",
    ]
    with patch("sys.argv", full_argv):
        args = controlnet_upscale_parser.parse_args()
        assert args.prompt == "upscaled beautiful landscape"
        assert args.controlnet_image_path == "source.png"
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 2
        assert isinstance(args.width, int)
        assert args.width == 1920
        assert args.steps == 20
        assert args.guidance == 7.5
        assert args.controlnet_strength == 0.8
        assert args.seed == [42]
        assert args.output == "upscaled.png"


@pytest.mark.fast
def test_mixed_dimension_types(controlnet_upscale_parser, controlnet_upscale_minimal_argv):
    with patch("sys.argv", controlnet_upscale_minimal_argv + ["--height", "2x", "--width", "1024"]):
        args = controlnet_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 2
        assert isinstance(args.width, int)
        assert args.width == 1024
