import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from mflux.models.common.lora.download.lora_library import LoRALibrary


@pytest.fixture(autouse=True)
def reset_lora_registry():
    """Reset the global registry before and after each test."""
    original = LoRALibrary._registry.copy()
    LoRALibrary._registry.clear()
    yield
    LoRALibrary._registry.clear()
    LoRALibrary._registry.update(original)


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

    result = LoRALibrary.discover_files([lib1])

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
    result = LoRALibrary.discover_files([lib1, lib2])

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
    result = LoRALibrary.discover_files([Path("/nonexistent/path")])
    assert result == {}


def test_get_lora_path_existing_file(temp_lora_dirs):
    """Test that existing file paths are returned as-is."""
    lib1, _ = temp_lora_dirs
    existing_file = lib1 / "style_a.safetensors"

    result = LoRALibrary.get_path(str(existing_file))
    assert result == str(existing_file)


def test_get_lora_path_from_registry(temp_lora_dirs):
    """Test resolving paths from the registry."""
    lib1, lib2 = temp_lora_dirs

    # Mock the registry
    with mock.patch.object(
        LoRALibrary,
        "_registry",
        {
            "style_a": lib1 / "style_a.safetensors",
            "style_b": lib1 / "style_b.safetensors",
        },
    ):
        # Should resolve from registry
        assert LoRALibrary.get_path("style_a") == str(lib1 / "style_a.safetensors")
        assert LoRALibrary.get_path("style_b") == str(lib1 / "style_b.safetensors")

        # Should raise FileNotFoundError if not in registry
        with pytest.raises(FileNotFoundError, match="LoRA file not found: 'unknown_style'"):
            LoRALibrary.get_path("unknown_style")


def test_initialize_registry_from_env(temp_lora_dirs):
    """Test that registry is initialized from LORA_LIBRARY_PATH env var."""
    lib1, lib2 = temp_lora_dirs

    # Set up environment variable with colon-delimited paths
    env_path = f"{lib1}:{lib2}"

    with mock.patch.dict(os.environ, {"LORA_LIBRARY_PATH": env_path}):
        # Re-initialize the registry
        LoRALibrary._initialize_registry()
        registry = LoRALibrary.get_registry()

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
    original_registry = LoRALibrary._registry.copy()
    try:
        with mock.patch.dict(os.environ, {}, clear=True):
            LoRALibrary._initialize_registry()
            registry = LoRALibrary.get_registry()
            assert registry == {}
    finally:
        # Restore the original registry
        LoRALibrary._registry = original_registry


def test_get_registry_returns_copy():
    """Test that get_registry returns a copy, not the original."""
    with mock.patch.object(LoRALibrary, "_registry", {"test": Path("/test")}):
        registry1 = LoRALibrary.get_registry()
        registry2 = LoRALibrary.get_registry()

        # Should be equal but different objects
        assert registry1 == registry2
        assert registry1 is not registry2

        # Modifying the copy shouldn't affect the original
        registry1["new"] = Path("/new")
        assert "new" not in LoRALibrary._registry


def test_transformer_digit_filtering(temp_lora_dirs):
    """Test that digit-named files in transformer directories are filtered out."""
    lib1, _ = temp_lora_dirs

    # Create additional test cases
    (lib1 / "9.safetensors").touch()  # Should be included (not in transformer/)
    (lib1 / "other_dir").mkdir()
    (lib1 / "other_dir" / "5.safetensors").touch()  # Should be included (not in transformer/)

    result = LoRALibrary.discover_files([lib1])

    # Digit files NOT in transformer/ should be included
    assert "9" in result
    assert "5" in result

    # Digit files in transformer/ should be excluded
    assert "0" not in result
    assert "1" not in result

    # Non-digit files in transformer/ should be included
    assert "style_valid" in result


# =============================================================================
# Tests for LoRA path format detection
# =============================================================================


def test_local_file_resolution():
    """Test that local file paths are resolved directly."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        temp_path = f.name
        f.write(b"test")

    try:
        result = LoRALibrary.get_path(temp_path)
        assert result == temp_path
    finally:
        os.unlink(temp_path)


def test_huggingface_repo_format_detection():
    """Test that HuggingFace repo format (org/model) is correctly detected."""
    # These should be recognized as HF repos (contain exactly one /)
    hf_repos = [
        "XLabs-AI/flux-RealismLora",
        "author/model",
        "org/lora-weights",
    ]

    for repo in hf_repos:
        # Should have exactly one / and no :
        assert repo.count("/") == 1
        assert ":" not in repo


def test_huggingface_collection_format_detection():
    """Test that HuggingFace collection format (repo:filename) is correctly detected."""
    # These should be recognized as HF collections (contain : and /)
    collections = [
        (
            "ali-vilab/In-Context-LoRA:film-storyboard.safetensors",
            "ali-vilab/In-Context-LoRA",
            "film-storyboard.safetensors",
        ),
        (
            "RiverZ/normal-lora:pytorch_lora_weights.safetensors",
            "RiverZ/normal-lora",
            "pytorch_lora_weights.safetensors",
        ),
        ("author/collection:my-lora.safetensors", "author/collection", "my-lora.safetensors"),
    ]

    for full_path, expected_repo, expected_file in collections:
        assert ":" in full_path
        assert "/" in full_path
        repo_id, filename = full_path.split(":", 1)
        assert repo_id == expected_repo
        assert filename == expected_file


def test_resolve_paths_with_none():
    """Test that resolve_paths handles None input."""
    result = LoRALibrary.resolve_paths(None)
    assert result == []


def test_resolve_paths_with_empty_list():
    """Test that resolve_paths handles empty list input."""
    result = LoRALibrary.resolve_paths([])
    assert result == []


def test_resolve_scales_with_none():
    """Test that resolve_scales handles None input."""
    result = LoRALibrary.resolve_scales(None)
    assert result == []


def test_resolve_scales_with_empty_list():
    """Test that resolve_scales handles empty list input."""
    result = LoRALibrary.resolve_scales([])
    assert result == []


def test_resolve_scales_with_values():
    """Test that resolve_scales returns the input when provided."""
    result = LoRALibrary.resolve_scales([0.5, 1.0, 0.8])
    assert result == [0.5, 1.0, 0.8]


def test_resolve_paths_with_local_files():
    """Test that resolve_paths correctly resolves local files."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f1:
        path1 = f1.name
        f1.write(b"test1")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f2:
        path2 = f2.name
        f2.write(b"test2")

    try:
        result = LoRALibrary.resolve_paths([path1, path2])
        assert len(result) == 2
        assert path1 in result
        assert path2 in result
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_resolve_paths_filters_nonexistent():
    """Test that resolve_paths filters out nonexistent paths gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        existing_path = f.name
        f.write(b"test")

    try:
        # Mix of existing and nonexistent paths
        result = LoRALibrary.resolve_paths([existing_path, "/nonexistent/lora.safetensors"])
        # Should only contain the existing file
        assert len(result) == 1
        assert existing_path in result
    finally:
        os.unlink(existing_path)


def test_get_path_raises_for_unknown():
    """Test that get_path raises FileNotFoundError for unknown paths."""
    with pytest.raises(FileNotFoundError, match="LoRA file not found"):
        LoRALibrary.get_path("completely_unknown_lora_that_does_not_exist")


# =============================================================================
# Integration tests for HuggingFace downloads (require network)
# =============================================================================


@pytest.mark.integration
def test_huggingface_collection_download():
    """Test downloading a LoRA from a HuggingFace collection using repo:filename format."""
    from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR

    test_lora = "ali-vilab/In-Context-LoRA:film-storyboard.safetensors"
    expected_filename = "film-storyboard.safetensors"
    cached_path = MFLUX_LORA_CACHE_DIR / expected_filename

    # 1. Delete cached file to ensure fresh download
    if cached_path.exists():
        cached_path.unlink()
    assert not cached_path.exists(), "Cache should be cleared before test"

    # 2. Download the LoRA
    result = LoRALibrary.get_path(test_lora)

    # 3. Verify download succeeded
    assert result is not None
    result_path = Path(result)
    assert result_path.exists(), "Downloaded file should exist"
    assert result_path.suffix == ".safetensors"

    # 4. Verify file has reasonable size (> 1MB for a real LoRA)
    file_size_mb = result_path.stat().st_size / (1024 * 1024)
    assert file_size_mb > 1, f"LoRA file should be > 1MB, got {file_size_mb:.1f}MB"


@pytest.mark.integration
def test_huggingface_collection_cache_hit():
    """Test that cached LoRAs are reused without re-downloading."""
    from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR

    test_lora = "ali-vilab/In-Context-LoRA:film-storyboard.safetensors"
    expected_filename = "film-storyboard.safetensors"
    cached_path = MFLUX_LORA_CACHE_DIR / expected_filename

    # Ensure file exists (from previous test or download it)
    if not cached_path.exists():
        LoRALibrary.get_path(test_lora)

    # Record modification time before second call
    mtime_before = cached_path.stat().st_mtime

    # Second call should use cache (no download)
    result = LoRALibrary.get_path(test_lora)

    # File should not have been modified (cache hit)
    mtime_after = cached_path.stat().st_mtime
    assert mtime_before == mtime_after, "File should not be re-downloaded on cache hit"
    assert result == str(cached_path)
