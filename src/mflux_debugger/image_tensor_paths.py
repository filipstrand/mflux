"""
Centralized path management for images and tensors in the debugger.

Defines the directory structure:
- Images: images/latest/{framework}/ and images/archive/
- Tensors: tensors/latest/ and tensors/archive/
"""

from pathlib import Path

from mflux_debugger.log_paths import get_debugger_root


def get_images_root() -> Path:
    """Get the images root directory."""
    images_dir = get_debugger_root() / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_images_latest_dir() -> Path:
    """Get latest images directory."""
    dir_path = get_images_root() / "latest"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_images_latest_framework_dir(framework: str) -> Path:
    """Get latest images directory for a specific framework (mlx or pytorch)."""
    dir_path = get_images_latest_dir() / framework
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_images_archive_dir() -> Path:
    """Get archived images directory."""
    dir_path = get_images_root() / "archive"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_tensors_root() -> Path:
    """Get the tensors root directory."""
    tensors_dir = get_debugger_root() / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    return tensors_dir


def get_tensors_latest_dir() -> Path:
    """Get latest tensors directory."""
    dir_path = get_tensors_root() / "latest"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_tensors_archive_dir() -> Path:
    """Get archived tensors directory."""
    dir_path = get_tensors_root() / "archive"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
