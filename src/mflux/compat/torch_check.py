"""
Torch compatibility and optional dependency checking.

This module provides utilities for gracefully handling torch as an optional dependency.
"""

_TORCH_AVAILABLE = None
_TORCH_ERROR = None


def is_torch_available() -> bool:
    """
    Check if torch is available.

    Returns:
        bool: True if torch can be imported, False otherwise
    """
    global _TORCH_AVAILABLE, _TORCH_ERROR
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE

    try:
        import torch  # noqa: F401

        _TORCH_AVAILABLE = True
        return True
    except ImportError as e:
        _TORCH_ERROR = e
        _TORCH_AVAILABLE = False
        return False


def require_torch(feature_name: str = "this feature") -> None:
    """
    Raise a helpful error if torch is not available.

    Args:
        feature_name: Name of the feature requiring torch

    Raises:
        ImportError: With installation instructions if torch is not available
    """
    if not is_torch_available():
        msg = (
            f"\n{'=' * 70}\n"
            f"❌ {feature_name} requires PyTorch, but it's not installed.\n\n"
            f"To enable this feature, install PyTorch support:\n\n"
            f"  # For basic weight conversion (most models)\n"
            f"  pip install 'mflux[weights]'\n\n"
            f"  # For VLM models (FIBO-VLM, Qwen-VL)\n"
            f"  pip install 'mflux[vlm]'\n\n"
            f"  # For LoRA conversion\n"
            f"  pip install 'mflux[lora]'\n\n"
            f"  # For all features\n"
            f"  pip install 'mflux[all]'\n\n"
            f"Or install PyTorch directly:\n"
            f"  pip install torch\n"
            f"{'=' * 70}\n"
        )
        raise ImportError(msg) from _TORCH_ERROR


def optional_import_torch():
    """
    Optionally import torch with graceful fallback.

    Returns:
        torch module if available, None otherwise

    Example:
        >>> torch = optional_import_torch()
        >>> if torch:
        ...     tensor = torch.zeros(10)
    """
    if is_torch_available():
        import torch

        return torch
    return None


def get_torch_info() -> dict[str, str] | None:
    """
    Get information about the installed torch version.

    Returns:
        Dictionary with version info, or None if torch is not available
    """
    torch = optional_import_torch()
    if torch is None:
        return None

    return {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
