"""Display metadata information from MFLUX generated images."""

import json
import sys
from pathlib import Path

from mflux.post_processing.metadata_reader import MetadataReader
from mflux.ui.cli.parsers import CommandLineParser


def format_brief(metadata: dict) -> str:
    """Format metadata in brief mode showing only key information."""
    exif = metadata.get("exif", {})
    if not exif:
        return "No metadata found"

    lines = []
    lines.append("=" * 60)
    lines.append("MFLUX Image Information")
    lines.append("=" * 60)

    # Key information
    if prompt := exif.get("prompt"):
        lines.append(f"\nPrompt: {prompt}")

    if model := exif.get("model"):
        lines.append(f"Model: {model}")

    if width := exif.get("width"):
        lines.append(f"Width: {width}")

    if height := exif.get("height"):
        lines.append(f"Height: {height}")

    if seed := exif.get("seed"):
        lines.append(f"Seed: {seed}")

    if steps := exif.get("steps"):
        lines.append(f"Steps: {steps}")

    if guidance := exif.get("guidance"):
        lines.append(f"Guidance: {guidance}")

    if quantize := exif.get("quantize"):
        lines.append(f"Quantization: {quantize}-bit")

    if lora_paths := exif.get("lora_paths"):
        lines.append(f"\nLoRAs: {len(lora_paths)}")
        lora_scales = exif.get("lora_scales", [])
        for i, lora in enumerate(lora_paths):
            scale = lora_scales[i] if i < len(lora_scales) else 1.0
            lora_name = Path(lora).name
            lines.append(f"  - {lora_name} (scale: {scale})")

    if gen_time := exif.get("generation_time_seconds"):
        lines.append(f"\nGeneration Time: {gen_time:.2f}s")

    if version := exif.get("mflux_version"):
        lines.append(f"MFLUX Version: {version}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_full(metadata: dict) -> str:
    """Format metadata showing all available information."""
    lines = []
    lines.append("=" * 80)
    lines.append("MFLUX Image Metadata (Full)")
    lines.append("=" * 80)

    # EXIF Section
    if exif := metadata.get("exif"):
        lines.append("\n" + "─" * 80)
        lines.append("EXIF Metadata:")
        lines.append("─" * 80)
        lines.append(json.dumps(exif, indent=2))

    # XMP Section
    if xmp := metadata.get("xmp"):
        lines.append("\n" + "─" * 80)
        lines.append("XMP Metadata:")
        lines.append("─" * 80)
        for key, value in xmp.items():
            lines.append(f"{key}: {value}")

    if not exif and not xmp:
        lines.append("\nNo metadata found in this image.")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_json(metadata: dict) -> str:
    """Format metadata as JSON."""
    return json.dumps(metadata, indent=2)


def main():
    # Parse command line arguments
    parser = CommandLineParser(description="Display metadata from MFLUX generated images")
    parser.add_info_arguments()
    args = parser.parse_args()

    # Check if file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Read metadata
    metadata = MetadataReader.read_all_metadata(image_path)

    # Check if metadata was found
    if not metadata or (not metadata.get("exif") and not metadata.get("xmp")):
        print("No metadata found")
        sys.exit(1)

    # Format and display based on options
    if args.format == "json":
        print(format_json(metadata))
    elif args.brief:
        print(format_brief(metadata))
    else:
        print(format_full(metadata))


if __name__ == "__main__":
    main()

