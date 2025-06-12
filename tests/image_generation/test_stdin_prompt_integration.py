import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    return tmp_path


def test_stdin_prompt_with_actual_generation(temp_output_dir):
    """Test that stdin prompt is correctly used in actual image generation and saved in metadata."""
    stdin_prompt = "A beautiful mountain landscape with snow"
    output_image = temp_output_dir / "test_stdin.png"
    metadata_file = temp_output_dir / "test_stdin.json"

    # Run the actual mflux.generate module with stdin
    cmd = [
        sys.executable,
        "-m",
        "mflux.generate",
        "--prompt",
        "-",
        "--model",
        "dev",
        "--steps",
        "1",
        "--seed",
        "42",
        "--height",
        "256",
        "--width",
        "256",
        "-q",
        "4",
        "--output",
        str(output_image),
        "--metadata",
    ]

    # Execute the command with stdin input
    process = subprocess.run(
        cmd,
        input=stdin_prompt,
        text=True,
        capture_output=True,
    )

    # Check that the command succeeded
    assert process.returncode == 0, f"Command failed with stderr: {process.stderr}"

    # Check that output files were created
    assert output_image.exists(), "Output image was not created"
    assert metadata_file.exists(), "Metadata file was not created"

    # Load and verify metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Verify the prompt in metadata matches our stdin input
    assert metadata["prompt"] == stdin_prompt


def test_stdin_prompt_multiline_with_actual_generation(temp_output_dir):
    """Test that multiline stdin prompt is correctly preserved in metadata."""
    stdin_prompt = """A fantasy scene with:
- Dragons flying in the sky
- A castle on a mountain
- Magical aurora lights"""

    output_image = temp_output_dir / "test_multiline.png"
    metadata_file = temp_output_dir / "test_multiline.json"

    cmd = [
        sys.executable,
        "-m",
        "mflux.generate",
        "--prompt",
        "-",
        "--model",
        "dev",
        "--steps",
        "1",
        "--seed",
        "123",
        "--height",
        "256",
        "--width",
        "256",
        "-q",
        "4",
        "--output",
        str(output_image),
        "--metadata",
    ]

    process = subprocess.run(cmd, input=stdin_prompt, text=True, capture_output=True)

    assert process.returncode == 0, f"Command failed with stderr: {process.stderr}"
    assert metadata_file.exists(), "Metadata file was not created"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Verify the multiline prompt is preserved (strip() removes leading/trailing whitespace)
    assert metadata["prompt"] == stdin_prompt.strip()


def test_empty_stdin_fails_generation(temp_output_dir):
    """Test that empty stdin causes generation to fail with appropriate error."""
    output_image = temp_output_dir / "test_empty.png"

    cmd = [
        sys.executable,
        "-m",
        "mflux.generate",
        "--prompt",
        "-",
        "--model",
        "dev",
        "--steps",
        "1",
        "--height",
        "256",
        "--width",
        "256",
        "-q",
        "4",
        "--output",
        str(output_image),
    ]

    process = subprocess.run(
        cmd,
        input="",  # Empty stdin
        text=True,
        capture_output=True,
    )

    # The application prints the error but exits with 0 (by design)
    assert process.returncode == 0
    # Error message should be in stdout (as it's printed, not written to stderr)
    assert "No prompt provided via stdin" in process.stdout


def test_pipe_from_echo_command(temp_output_dir):
    """Test using echo command to pipe prompt, simulating real usage."""
    prompt = "A serene lake at sunset"
    output_image = temp_output_dir / "test_echo.png"
    metadata_file = temp_output_dir / "test_echo.json"

    # Simulate: echo "prompt" | mflux-generate --prompt - ...
    echo_cmd = f'echo "{prompt}" | {sys.executable} -m mflux.generate --prompt - --model dev --steps 1 --height 256 --width 256 -q 4 --output {output_image} --metadata'

    process = subprocess.run(echo_cmd, shell=True, capture_output=True, text=True)

    assert process.returncode == 0, f"Command failed with stderr: {process.stderr}"
    assert metadata_file.exists(), "Metadata file was not created"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    assert metadata["prompt"] == prompt
