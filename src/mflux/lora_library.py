#!/usr/bin/env python3
"""CLI tool for managing the MFLUX LoRA library."""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from mflux.weights.lora_library import _discover_lora_files


def list_loras(paths: list[str] | None = None) -> int:
    """List all discovered LoRA files from specified paths or LORA_LIBRARY_PATH.

    Args:
        paths: Optional list of paths to use instead of LORA_LIBRARY_PATH

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if paths:
        # Use provided paths
        library_paths = [Path(p.strip()) for p in paths]
    else:
        # Use environment variable
        library_path_env = os.environ.get("LORA_LIBRARY_PATH")

        if not library_path_env:
            print("LORA_LIBRARY_PATH environment variable is not set.", file=sys.stderr)
            print("Set it to one or more colon-separated directories containing .safetensors files.", file=sys.stderr)
            print("Alternatively, use --paths to specify directories directly.", file=sys.stderr)
            return 1

        # Parse library paths from environment
        library_paths = [Path(p.strip()) for p in library_path_env.split(":") if p.strip()]

    # Check which paths exist
    valid_paths = []
    for path in library_paths:
        if path.exists() and path.is_dir():
            valid_paths.append(path)
        else:
            print(f"Warning: Path does not exist or is not a directory: {path}", file=sys.stderr)

    if not valid_paths:
        print("No valid directories found in LORA_LIBRARY_PATH.", file=sys.stderr)
        return 1

    # Discover all LoRA files
    lora_registry = _discover_lora_files(valid_paths)

    if not lora_registry:
        print("No .safetensors files found in the specified directories.")
        return 0

    # Sort by basename for consistent output
    sorted_items = sorted(lora_registry.items())

    # Print all discovered LoRAs
    print("Discovered LoRA files:")
    print("-" * 80)
    for basename, full_path in sorted_items:
        print(f"{basename} -> {full_path}")

    # Calculate statistics per top-level directory
    stats = defaultdict(int)
    for full_path in lora_registry.values():
        # Find which library path this file belongs to
        for lib_path in valid_paths:
            try:
                # Check if full_path is relative to lib_path
                full_path.relative_to(lib_path.resolve())
                stats[str(lib_path)] += 1
                break
            except ValueError:
                # Not relative to this lib_path, continue
                continue

    # Print summary
    print("-" * 80)
    print(f"\nTotal LoRA files found: {len(lora_registry)}")

    if len(valid_paths) > 1:
        print("\nBreakdown by library path:")
        for lib_path in valid_paths:
            count = stats.get(str(lib_path), 0)
            print(f"  {lib_path}: {count} files")

    return 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="MFLUX LoRA Library management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  LORA_LIBRARY_PATH    Colon-separated list of directories containing .safetensors files
                       Example: /path/to/loras:/another/path/to/loras

Examples:
  # List all discovered LoRA files using LORA_LIBRARY_PATH
  mflux-lora-library list

  # With environment variable set
  LORA_LIBRARY_PATH=/home/user/loras:/opt/shared/loras mflux-lora-library list

  # Override with specific paths
  mflux-lora-library list --paths /path/to/loras /another/path/to/loras
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add 'list' command
    list_parser = subparsers.add_parser("list", help="List all discovered LoRA files")
    list_parser.add_argument(
        "--paths", nargs="+", help="Override LORA_LIBRARY_PATH with these directories (space-separated)"
    )

    args = parser.parse_args()

    if args.command == "list":
        return list_loras(paths=args.paths)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
