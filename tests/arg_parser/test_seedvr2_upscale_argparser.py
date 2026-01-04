from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.utils.scale_factor import ScaleFactor


def _create_seedvr2_upscale_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Upscale an image using SeedVR2")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def seedvr2_upscale_parser() -> CommandLineParser:
    return _create_seedvr2_upscale_parser()


@pytest.fixture
def seedvr2_upscale_minimal_argv() -> list[str]:
    return ["mflux-upscale-seedvr2", "--image-path", "image.png"]


@pytest.mark.fast
def test_seedvr2_resolution_integer(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "512"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.resolution == 512


@pytest.mark.fast
def test_seedvr2_resolution_scale_factor(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "2x"]):
        args = seedvr2_upscale_parser.parse_args()
        assert isinstance(args.resolution, ScaleFactor)
        assert args.resolution.value == 2


@pytest.mark.fast
def test_seedvr2_resolution_auto(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--resolution", "auto"]):
        args = seedvr2_upscale_parser.parse_args()
        assert isinstance(args.resolution, ScaleFactor)
        assert args.resolution.value == 1


@pytest.mark.fast
def test_seedvr2_multiple_images_and_seeds(seedvr2_upscale_parser):
    argv = [
        "mflux-upscale-seedvr2",
        "--image-path",
        "img1.png",
        "img2.png",
        "--seed",
        "42",
        "43",
    ]
    with patch("sys.argv", argv):
        args = seedvr2_upscale_parser.parse_args()
        assert args.image_path == [Path("img1.png"), Path("img2.png")]
        assert args.seed == [42, 43]
        # Verify output pattern is updated for multiple seeds
        assert "{seed}" in args.output


@pytest.mark.fast
def test_seedvr2_quantize_choices(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    # Valid choices
    for q in ["4", "8"]:
        with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--quantize", q]):
            args = seedvr2_upscale_parser.parse_args()
            assert args.quantize == int(q)

    # Invalid choice
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--quantize", "16"]):
        with pytest.raises(SystemExit):
            seedvr2_upscale_parser.parse_args()


@pytest.mark.fast
def test_seedvr2_model_arg(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    # Test with --model
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--model", "some/path"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.model == "some/path"
        assert args.model_path == "some/path"

    # Test with -m
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["-m", "another/path"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.model == "another/path"
        assert args.model_path == "another/path"


@pytest.mark.fast
def test_seedvr2_softness(seedvr2_upscale_parser, seedvr2_upscale_minimal_argv):
    with patch("sys.argv", seedvr2_upscale_minimal_argv + ["--softness", "0.5"]):
        args = seedvr2_upscale_parser.parse_args()
        assert args.softness == 0.5
