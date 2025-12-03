import os
import sys
from collections import defaultdict
from pathlib import Path

from mflux.models.common.resolution.lora_resolution import LoraResolution


class LoraLibraryUtil:
    @staticmethod
    def epilog() -> str:
        return """
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
"""

    @staticmethod
    def list_loras(paths: list[str] | None = None) -> int:
        if paths:
            # Use provided paths
            library_paths = [Path(p.strip()) for p in paths]
        else:
            # Use environment variable
            library_path_env = os.environ.get("LORA_LIBRARY_PATH")

            if not library_path_env:
                print("LORA_LIBRARY_PATH environment variable is not set.", file=sys.stderr)
                print(
                    "Set it to one or more colon-separated directories containing .safetensors files.", file=sys.stderr
                )
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
        lora_registry = LoraResolution.discover_files(valid_paths)

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
