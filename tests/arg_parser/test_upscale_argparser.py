from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.ui.cli.parsers import CommandLineParser, int_or_special_value
from mflux.ui.scale_factor import ScaleFactor


def _create_custom_upscale_parser() -> CommandLineParser:
    """Create parser with custom dimension scale factor support"""
    parser = CommandLineParser(description="Generate an upscaled image from a source image")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()

    # Manually add the image generator arguments with scale factor support
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt", type=str, help="The textual description of the image to generate.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a file containing the prompt text.")
    parser.add_argument("--seed", type=int, default=None, nargs="+", help="Specify 1+ Entropy Seeds")
    parser.add_argument("--auto-seeds", type=int, default=-1, help="Auto generate N Entropy Seeds")

    # Add height/width with scale factor support
    parser.supports_image_generation = True
    parser.supports_dimension_scale_factor = True
    parser.add_argument(
        "--height", type=int_or_special_value, default="auto", help="Image height (Default is source image height)"
    )
    parser.add_argument(
        "--width", type=int_or_special_value, default="auto", help="Image width (Default is source image width)"
    )
    parser.add_argument("--steps", type=int, default=None, help="Inference Steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance Scale")

    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_upscale_parser() -> CommandLineParser:
    return _create_custom_upscale_parser()


@pytest.fixture
def mflux_upscale_minimal_argv() -> list[str]:
    return ["mflux-upscale", "--prompt", "upscaled image", "--controlnet-image-path", "image.png"]


def test_scale_factor_auto(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test that 'auto' gets parsed as a ScaleFactor with value 1"""
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "auto", "--width", "auto"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 1
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 1


def test_scale_factor_multiplier_format(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test scale factor formats like '1x', '2x', '3.5x'"""
    # Test integer scale factor
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "2x", "--width", "3x"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 2
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 3

    # Test float scale factor
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "1.5x", "--width", "2.5x"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 1.5
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 2.5

    # Test decimal scale factor
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "3.14x", "--width", "0.5x"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 3.14
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 0.5


def test_plain_integer_dimensions(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test plain integer values for dimensions"""
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "1024", "--width", "768"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, int)
        assert args.height == 1024
        assert isinstance(args.width, int)
        assert args.width == 768


def test_mixed_dimension_types(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test mixing scale factors and integers"""
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "2x", "--width", "1024"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 2
        assert isinstance(args.width, int)
        assert args.width == 1024

    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "768", "--width", "1.5x"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, int)
        assert args.height == 768
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 1.5


def test_default_dimensions(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test default values are 'auto' for upscale parser"""
    with patch("sys.argv", mflux_upscale_minimal_argv):
        args = mflux_upscale_parser.parse_args()
        # Default "auto" gets parsed into ScaleFactor(value=1)
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 1
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 1


def test_invalid_scale_factor_format(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test invalid scale factor formats raise errors"""
    # Invalid format without 'x'
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "2.5"]):
        with pytest.raises(SystemExit):
            mflux_upscale_parser.parse_args()

    # Invalid format with multiple 'x'
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "2xx"]):
        with pytest.raises(SystemExit):
            mflux_upscale_parser.parse_args()

    # Invalid non-numeric value
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "abcx"]):
        with pytest.raises(SystemExit):
            mflux_upscale_parser.parse_args()

    # Invalid empty value before 'x'
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "x"]):
        with pytest.raises(SystemExit):
            mflux_upscale_parser.parse_args()


def test_case_insensitive_scale_factor(mflux_upscale_parser, mflux_upscale_minimal_argv):
    """Test that scale factors are case insensitive"""
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "2X", "--width", "1.5X"]):
        args = mflux_upscale_parser.parse_args()
        assert isinstance(args.height, ScaleFactor)
        assert args.height.value == 2
        assert isinstance(args.width, ScaleFactor)
        assert args.width.value == 1.5


def test_upscale_with_all_arguments(mflux_upscale_parser):
    """Test upscale parser with all arguments"""
    full_argv = [
        "mflux-upscale",
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
        args = mflux_upscale_parser.parse_args()
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
