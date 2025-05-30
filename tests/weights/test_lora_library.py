import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from mflux.weights import lora_library


@pytest.fixture(autouse=True)
def reset_lora_registry():
    """Reset the global registry before and after each test."""
    original = lora_library._LORA_REGISTRY.copy()
    lora_library._LORA_REGISTRY.clear()
    yield
    lora_library._LORA_REGISTRY.clear()
    lora_library._LORA_REGISTRY.update(original)


@pytest.fixture
def temp_lora_dirs():
    """Create temporary directories with fake .safetensors files for testing."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create two library directories
        lib1 = Path(temp_root) / "library1"
        lib2 = Path(temp_root) / "library2"
        lib1.mkdir()
        lib2.mkdir()

        # Create some .safetensors files
        (lib1 / "style_a.safetensors").touch()
        (lib1 / "style_b.safetensors").touch()
        (lib1 / "subdir").mkdir()
        (lib1 / "subdir" / "style_c.safetensors").touch()

        # Create transformer directory with digit-named files (should be ignored)
        (lib1 / "transformer").mkdir()
        (lib1 / "transformer" / "0.safetensors").touch()
        (lib1 / "transformer" / "1.safetensors").touch()
        (lib1 / "transformer" / "style_valid.safetensors").touch()  # Should not be ignored

        (lib2 / "style_b.safetensors").touch()  # Duplicate name
        (lib2 / "style_d.safetensors").touch()

        yield lib1, lib2


def test_discover_lora_files_single_path(temp_lora_dirs):
    """Test discovering files in a single library path."""
    lib1, lib2 = temp_lora_dirs

    result = lora_library._discover_lora_files([lib1])

    assert len(result) == 4  # Should not include digit-named files in transformer/
    assert "style_a" in result
    assert "style_b" in result
    assert "style_c" in result
    assert "style_valid" in result
    assert "0" not in result  # Digit-named files in transformer/ should be excluded
    assert "1" not in result
    assert result["style_a"] == (lib1 / "style_a.safetensors").resolve()
    assert result["style_c"] == (lib1 / "subdir" / "style_c.safetensors").resolve()
    assert result["style_valid"] == (lib1 / "transformer" / "style_valid.safetensors").resolve()


def test_discover_lora_files_multiple_paths_precedence(temp_lora_dirs):
    """Test that earlier paths take precedence for duplicate names."""
    lib1, lib2 = temp_lora_dirs

    # lib1 should take precedence over lib2
    result = lora_library._discover_lora_files([lib1, lib2])

    assert len(result) == 5  # style_a, style_b, style_c, style_valid, style_d
    assert "style_a" in result
    assert "style_b" in result
    assert "style_c" in result
    assert "style_valid" in result
    assert "style_d" in result
    assert "0" not in result  # Digit-named files should still be excluded
    assert "1" not in result

    # style_b should come from lib1 (first in list)
    assert result["style_b"] == (lib1 / "style_b.safetensors").resolve()
    assert result["style_d"] == (lib2 / "style_d.safetensors").resolve()


def test_discover_lora_files_nonexistent_path():
    """Test handling of nonexistent paths."""
    result = lora_library._discover_lora_files([Path("/nonexistent/path")])
    assert result == {}


def test_get_lora_path_existing_file(temp_lora_dirs):
    """Test that existing file paths are returned as-is."""
    lib1, _ = temp_lora_dirs
    existing_file = lib1 / "style_a.safetensors"

    result = lora_library.get_lora_path(str(existing_file))
    assert result == str(existing_file)


def test_get_lora_path_from_registry(temp_lora_dirs):
    """Test resolving paths from the registry."""
    lib1, lib2 = temp_lora_dirs

    # Mock the registry
    with mock.patch.object(
        lora_library,
        "_LORA_REGISTRY",
        {
            "style_a": lib1 / "style_a.safetensors",
            "style_b": lib1 / "style_b.safetensors",
        },
    ):
        # Should resolve from registry
        assert lora_library.get_lora_path("style_a") == str(lib1 / "style_a.safetensors")
        assert lora_library.get_lora_path("style_b") == str(lib1 / "style_b.safetensors")

        # Should raise FileNotFoundError if not in registry
        with pytest.raises(FileNotFoundError, match="LoRA file not found: 'unknown_style'"):
            lora_library.get_lora_path("unknown_style")


def test_initialize_registry_from_env(temp_lora_dirs):
    """Test that registry is initialized from LORA_LIBRARY_PATH env var."""
    lib1, lib2 = temp_lora_dirs

    # Set up environment variable with colon-delimited paths
    env_path = f"{lib1}:{lib2}"

    with mock.patch.dict(os.environ, {"LORA_LIBRARY_PATH": env_path}):
        # Re-initialize the registry
        lora_library._initialize_registry()
        registry = lora_library.get_registry()

        assert len(registry) == 5  # Includes style_valid from transformer/
        assert "style_a" in registry
        assert "style_b" in registry
        assert "style_c" in registry
        assert "style_valid" in registry
        assert "style_d" in registry
        assert "0" not in registry  # Digit files should be excluded
        assert "1" not in registry

        # Check precedence - style_b should be from lib1
        assert registry["style_b"] == (lib1 / "style_b.safetensors").resolve()


def test_initialize_registry_empty_env():
    """Test that registry is empty when env var is not set."""
    # Save the original registry
    original_registry = lora_library._LORA_REGISTRY.copy()
    try:
        with mock.patch.dict(os.environ, {}, clear=True):
            lora_library._initialize_registry()
            registry = lora_library.get_registry()
            assert registry == {}
    finally:
        # Restore the original registry
        lora_library._LORA_REGISTRY = original_registry


def test_get_registry_returns_copy():
    """Test that get_registry returns a copy, not the original."""
    with mock.patch.object(lora_library, "_LORA_REGISTRY", {"test": Path("/test")}):
        registry1 = lora_library.get_registry()
        registry2 = lora_library.get_registry()

        # Should be equal but different objects
        assert registry1 == registry2
        assert registry1 is not registry2

        # Modifying the copy shouldn't affect the original
        registry1["new"] = Path("/new")
        assert "new" not in lora_library._LORA_REGISTRY


def test_transformer_digit_filtering(temp_lora_dirs):
    """Test that digit-named files in transformer directories are filtered out."""
    lib1, _ = temp_lora_dirs

    # Create additional test cases
    (lib1 / "9.safetensors").touch()  # Should be included (not in transformer/)
    (lib1 / "other_dir").mkdir()
    (lib1 / "other_dir" / "5.safetensors").touch()  # Should be included (not in transformer/)

    result = lora_library._discover_lora_files([lib1])

    # Digit files NOT in transformer/ should be included
    assert "9" in result
    assert "5" in result

    # Digit files in transformer/ should be excluded
    assert "0" not in result
    assert "1" not in result

    # Non-digit files in transformer/ should be included
    assert "style_valid" in result
