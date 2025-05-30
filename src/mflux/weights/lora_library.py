import os
from pathlib import Path


def _discover_lora_files(library_paths: list[Path]) -> dict[str, Path]:
    """
    Discover all .safetensors files in the library paths and their subdirectories.
    Earlier paths in the list have higher precedence for duplicate basenames.

    Args:
        library_paths: List of paths to LORA library directories (in precedence order)

    Returns:
        Dictionary mapping basename (without extension) to full path
    """
    lora_files = {}

    # Process paths in reverse order so earlier paths overwrite later ones
    for library_path in reversed(library_paths):
        if not library_path.exists() or not library_path.is_dir():
            continue

        # Find all .safetensors files recursively
        for safetensor_path in library_path.rglob("*.safetensors"):
            # Use the basename without extension as the key
            basename = safetensor_path.stem

            # Skip files with digit-only names (0-9) in transformer directories
            if basename.isdigit() and safetensor_path.parent.name == "transformer":
                continue

            # Earlier paths in the list take precedence (overwrite)
            lora_files[basename] = safetensor_path.resolve()

    return lora_files


# Global registry that will be populated on module import
_LORA_REGISTRY: dict[str, Path] = {}


def _initialize_registry() -> None:
    """Initialize the global LORA registry from LORA_LIBRARY_PATH environment variable."""
    global _LORA_REGISTRY

    library_path_env = os.environ.get("LORA_LIBRARY_PATH")
    if library_path_env:
        # Split by colon to support multiple paths
        library_paths = [Path(p.strip()) for p in library_path_env.split(":") if p.strip()]
        _LORA_REGISTRY = _discover_lora_files(library_paths)


def get_lora_path(path_or_name: str) -> str:
    """
    Get the full path for a LORA file, resolving from library if needed.

    Args:
        path_or_name: Either a full path or a basename that exists in the library

    Returns:
        The resolved path as a string

    Raises:
        FileNotFoundError: If the file cannot be found either as a path or in the registry
    """
    # If it's already a path that exists, return it as-is
    path = Path(path_or_name)
    if path.exists():
        return str(path)

    # Otherwise, check if it's in the registry
    if path_or_name in _LORA_REGISTRY:
        return str(_LORA_REGISTRY[path_or_name])

    # If not found, raise FileNotFoundError
    raise FileNotFoundError(
        f"LoRA file not found: '{path_or_name}'. File does not exist and is not in the LoRA library."
    )


def get_registry() -> dict[str, Path]:
    """Get a copy of the current LORA registry."""
    return _LORA_REGISTRY.copy()


# Initialize the registry when the module is imported
_initialize_registry()
