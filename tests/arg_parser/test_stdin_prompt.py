from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


@pytest.fixture
def mflux_generate_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_lora_arguments()
    parser.add_image_to_image_arguments(required=False)
    parser.add_image_outpaint_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def temp_output_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mflux_stdin_test")


def test_prompt_from_stdin(mflux_generate_parser):
    """Test that --prompt - reads from stdin correctly."""
    stdin_content = "A beautiful sunset over the ocean"

    # Simulate stdin input
    with patch("sys.stdin", StringIO(stdin_content)):
        with patch("sys.argv", ["mflux-generate", "--prompt", "-", "--model", "dev"]):
            args = mflux_generate_parser.parse_args()

            # The parser returns the raw args, get_effective_prompt handles stdin
            assert args.prompt == "-"


def test_prompt_stdin_vs_regular(mflux_generate_parser):
    """Test that regular prompt still works when not using stdin."""
    regular_prompt = "A regular prompt not from stdin"

    with patch("sys.argv", ["mflux-generate", "--prompt", regular_prompt, "--model", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.prompt == regular_prompt
        assert get_effective_prompt(args) == regular_prompt


def test_prompt_stdin_with_whitespace(mflux_generate_parser):
    """Test that stdin prompt with surrounding whitespace is properly stripped."""
    stdin_content = "\n\n   A prompt with whitespace   \n\n"
    expected_prompt = "A prompt with whitespace"

    with patch("sys.stdin", StringIO(stdin_content)):
        with patch("sys.argv", ["mflux-generate", "--prompt", "-", "--model", "dev"]):
            args = mflux_generate_parser.parse_args()
            effective_prompt = get_effective_prompt(args)
            assert effective_prompt == expected_prompt


def test_prompt_file_takes_precedence_over_stdin(mflux_generate_parser, temp_output_dir):
    """Test that --prompt-file still works and takes precedence over stdin detection.
    because --prompt is not used in this scenario."""
    # Create a prompt file
    prompt_file = temp_output_dir / "prompt.txt"
    file_prompt = "Prompt from file"
    prompt_file.write_text(file_prompt)
    stdin_content = "This should not be used because --prompt is not provided in the command."

    with patch("sys.stdin", StringIO(stdin_content)):
        with patch("sys.argv", ["mflux-generate", "--prompt-file", str(prompt_file), "--model", "dev"]):
            args = mflux_generate_parser.parse_args()
            effective_prompt = get_effective_prompt(args)
            assert effective_prompt == file_prompt
