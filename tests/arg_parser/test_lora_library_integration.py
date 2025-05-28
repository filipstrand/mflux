import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from mflux.ui.cli.parsers import CommandLineParser
from mflux.weights import lora_library


@pytest.fixture
def temp_lora_library():
    """Create a temporary lora library for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lib_path = Path(temp_dir) / "lora_library"
        lib_path.mkdir()

        # Create some test .safetensors files
        (lib_path / "style1.safetensors").touch()
        (lib_path / "style2.safetensors").touch()
        (lib_path / "subdir").mkdir()
        (lib_path / "subdir" / "style3.safetensors").touch()

        yield lib_path


def test_parser_resolves_lora_paths_from_library(temp_lora_library):
    """Test that parser resolves lora names to full paths from library."""
    # Set up the environment variable
    with mock.patch.dict(os.environ, {"LORA_LIBRARY_PATH": str(temp_lora_library)}):
        # Re-initialize the registry
        lora_library._initialize_registry()

        parser = CommandLineParser()
        parser.add_model_arguments()
        parser.add_lora_arguments()
        parser.add_image_generator_arguments()

        # Test with lora names that should be resolved
        # Simulate command line arguments
        test_args = [
            "mflux-generate",  # argv[0] is program name
            "--model",
            "dev",
            "--prompt",
            "test prompt",
            "--lora-paths",
            "style1",
            "style3",
            "--lora-scales",
            "0.5",
            "0.8",
        ]

        with mock.patch.object(sys, "argv", test_args):
            args = parser.parse_args()

        assert len(args.lora_paths) == 2
        assert args.lora_paths[0] == str((temp_lora_library / "style1.safetensors").resolve())
        assert args.lora_paths[1] == str((temp_lora_library / "subdir" / "style3.safetensors").resolve())
        assert args.lora_scales == [0.5, 0.8]


def test_parser_preserves_full_paths(temp_lora_library):
    """Test that parser preserves full paths that already exist."""
    # Create a lora file outside the library
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
        external_lora = tmp_file.name

    try:
        parser = CommandLineParser()
        parser.add_model_arguments()
        parser.add_lora_arguments()
        parser.add_image_generator_arguments()

        test_args = ["mflux-generate-fill", "--model", "dev", "--prompt", "test prompt", "--lora-paths", external_lora]
        with mock.patch.object(sys, "argv", test_args):
            args = parser.parse_args()

        assert len(args.lora_paths) == 1
        assert args.lora_paths[0] == external_lora
    finally:
        os.unlink(external_lora)


def test_parser_mixed_lora_paths(temp_lora_library):
    """Test parser with mix of library names and full paths."""
    # Create an external lora file
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
        external_lora = tmp_file.name

    try:
        with mock.patch.dict(os.environ, {"LORA_LIBRARY_PATH": str(temp_lora_library)}):
            lora_library._initialize_registry()

            parser = CommandLineParser()
            parser.add_model_arguments()
            parser.add_lora_arguments()
            parser.add_image_generator_arguments()

            test_args = [
                "mflux-generate-redux",
                "--model",
                "dev",
                "--prompt",
                "test prompt",
                "--lora-paths",
                "style1",
                external_lora,
                "style2",
            ]
            with mock.patch.object(sys, "argv", test_args):
                args = parser.parse_args()

            assert len(args.lora_paths) == 3
            assert args.lora_paths[0] == str((temp_lora_library / "style1.safetensors").resolve())
            assert args.lora_paths[1] == external_lora
            assert args.lora_paths[2] == str((temp_lora_library / "style2.safetensors").resolve())
    finally:
        os.unlink(external_lora)


def test_parser_unknown_lora_names_error():
    """Test that unknown lora names raise an error."""
    # No library path set
    with mock.patch.dict(os.environ, {}, clear=True):
        lora_library._initialize_registry()

        parser = CommandLineParser()
        parser.add_model_arguments()
        parser.add_lora_arguments()
        parser.add_image_generator_arguments()

        # Should raise an error for unknown LoRA names
        test_args = [
            "mflux-generate-depth",
            "--model",
            "dev",
            "--prompt",
            "test prompt",
            "--lora-paths",
            "unknown_style",
        ]

        with mock.patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parser.parse_args()
