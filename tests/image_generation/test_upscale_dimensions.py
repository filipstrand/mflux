from unittest.mock import Mock, patch

import pytest

from mflux.ui.scale_factor import ScaleFactor


@pytest.mark.parametrize(
    "args_height,args_width,orig_height,orig_width,expected_height,expected_width",
    [
        # ScaleFactor dimensions
        (ScaleFactor(value=2), ScaleFactor(value=1.5), 768, 512, 1536, 768),
        # Integer dimensions
        (1024, 768, 512, 512, 1024, 768),
        # Mixed: ScaleFactor height, integer width
        (ScaleFactor(value=2.5), 1280, 480, 640, 1200, 1280),
        # Auto (ScaleFactor with value 1)
        (ScaleFactor(value=1), ScaleFactor(value=1), 512, 1024, 512, 1024),
    ],
)
def test_upscale_passes_correct_dimensions_to_generate_image(
    args_height, args_width, orig_height, orig_width, expected_height, expected_width
):
    """Test that upscale.py passes the correct dimensions to generate_image"""
    # Mock the image that will be opened
    mock_image = Mock()
    mock_image.size = (orig_width, orig_height)
    mock_image.height = orig_height
    mock_image.width = orig_width

    # Mock the flux object
    mock_flux = Mock()

    # Import and patch the actual upscale module
    with patch("PIL.Image.open", return_value=mock_image):
        with patch("mflux.upscale.Flux1Controlnet", return_value=mock_flux):
            with patch("mflux.upscale.ModelConfig"):
                with patch("mflux.upscale.CallbackManager"):
                    from mflux.upscale import main

                    # Mock command line args
                    mock_args = Mock()
                    mock_args.height = args_height
                    mock_args.width = args_width
                    mock_args.controlnet_image_path = "test.png"
                    mock_args.seed = [42]
                    mock_args.prompt = "test prompt"
                    mock_args.steps = 20
                    mock_args.controlnet_strength = 0.4
                    mock_args.quantize = None
                    mock_args.path = None
                    mock_args.lora_paths = None
                    mock_args.lora_scales = None

                    with patch("mflux.upscale.CommandLineParser") as mock_parser_class:
                        mock_parser = Mock()
                        mock_parser.parse_args.return_value = mock_args
                        mock_parser_class.return_value = mock_parser

                        with patch("mflux.upscale.get_effective_prompt", return_value="test prompt"):
                            # Call the main function
                            main()

                    # Verify generate_image was called with correct dimensions
                    mock_flux.generate_image.assert_called()
                    call_args = mock_flux.generate_image.call_args
                    config = call_args.kwargs["config"]

                    assert config.height == expected_height
                    assert config.width == expected_width
