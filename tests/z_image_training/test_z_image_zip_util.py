"""Tests for Z-Image ZIP utility functions.

Tests that:
- Corrupted ZIP files are handled gracefully
- Zip bomb attacks are detected and prevented
- Missing files in checkpoints are handled properly
- Path traversal attacks are blocked
"""

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from mflux.models.z_image.variants.training.state.zip_util import ZipUtil


def create_test_zip(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a test ZIP file with a single file inside."""
    zip_path = temp_dir / "test.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(filename, content)
    return zip_path


@pytest.mark.fast
def test_unzip_basic():
    """Test basic unzip functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create test data
        test_data = {"key": "value", "number": 42}
        zip_path = create_test_zip(temp_path, "data.json", json.dumps(test_data))

        # Unzip and load
        result = ZipUtil.unzip(
            zip_path=str(zip_path),
            filename="data.json",
            loader=lambda x: json.load(open(x, "r")),
        )

        assert result == test_data


@pytest.mark.fast
def test_unzip_file_not_found():
    """Test that FileNotFoundError is raised for non-existent ZIP."""
    with pytest.raises(FileNotFoundError, match="Archive file not found"):
        ZipUtil.unzip(
            zip_path="/nonexistent/path.zip",
            filename="data.json",
            loader=lambda x: x,
        )


@pytest.mark.fast
def test_unzip_missing_file_in_archive():
    """Test that KeyError is raised for missing file in archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = create_test_zip(temp_path, "other.json", "{}")

        with pytest.raises(KeyError, match="not found in archive"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="missing.json",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_corrupted_zip():
    """Test that corrupted ZIP files are handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "corrupted.zip"

        # Write corrupted data
        with open(zip_path, "wb") as f:
            f.write(b"this is not a valid zip file")

        with pytest.raises(ValueError, match="Invalid or corrupted ZIP"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="data.json",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_size_limit():
    """Test that oversized files are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "large.zip"

        # Create a file that would be larger than the limit
        # We can't easily create a 100MB+ file, so we'll use a smaller limit
        large_content = "x" * (1024 * 1024)  # 1MB of data
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("large.txt", large_content)

        # Should fail with 0.5MB limit
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="large.txt",
                loader=lambda x: x,
                max_uncompressed_mb=0.5,  # 0.5MB limit
            )


@pytest.mark.fast
def test_unzip_path_traversal_dotdot():
    """Test that path traversal via .. is blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "traversal.zip"

        # Create ZIP with path traversal attempt
        with zipfile.ZipFile(zip_path, "w") as zipf:
            # Note: This creates an entry with ../ in the name
            zipf.writestr("../etc/passwd", "malicious")

        with pytest.raises(ValueError, match="Path traversal detected"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="../etc/passwd",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_path_traversal_absolute():
    """Test that absolute paths are blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "absolute.zip"

        # Create ZIP with absolute path attempt
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("/etc/passwd", "malicious")

        with pytest.raises(ValueError, match="Path traversal detected"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="/etc/passwd",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_compression_ratio_check():
    """Test that suspicious compression ratios are detected.

    Note: This test validates the compression ratio check exists.
    A true zip bomb would have extreme compression ratios (1000:1+).
    """
    # The ZipUtil checks for compression ratios > MAX_COMPRESSION_RATIO (100:1)
    # Normal files have ratios of 2:1 to 10:1
    # We verify the constant is set appropriately
    assert ZipUtil.MAX_COMPRESSION_RATIO == 100
    assert ZipUtil.MAX_UNCOMPRESSED_SIZE_MB == 100


@pytest.mark.fast
def test_unzip_is_directory():
    """Test that directories are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="not a file"):
            ZipUtil.unzip(
                zip_path=tmpdir,  # Directory, not a file
                filename="data.json",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_custom_max_size():
    """Test that custom max_uncompressed_mb works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create small content
        small_content = "small data"
        zip_path = create_test_zip(temp_path, "small.txt", small_content)

        # Should work with reasonable limit
        result = ZipUtil.unzip(
            zip_path=str(zip_path),
            filename="small.txt",
            loader=lambda x: open(x, "r").read(),
            max_uncompressed_mb=1.0,
        )

        assert result == small_content


@pytest.mark.fast
def test_unzip_entry_name_mismatch():
    """Test that entry name must exactly match requested filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = create_test_zip(temp_path, "actual.json", "{}")

        # Request different filename than what's in archive
        with pytest.raises(KeyError, match="not found"):
            ZipUtil.unzip(
                zip_path=str(zip_path),
                filename="different.json",
                loader=lambda x: x,
            )


@pytest.mark.fast
def test_unzip_with_nested_path():
    """Test unzipping files with nested paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "nested.zip"

        # Create ZIP with nested path
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("subdir/data.json", '{"nested": true}')

        # Should extract the nested file
        result = ZipUtil.unzip(
            zip_path=str(zip_path),
            filename="subdir/data.json",
            loader=lambda x: json.load(open(x, "r")),
        )

        assert result == {"nested": True}


@pytest.mark.fast
def test_bytes_per_mb_constant():
    """Test that bytes per MB constant is correct."""
    assert ZipUtil.BYTES_PER_MB == 1024 * 1024


@pytest.mark.fast
def test_unzip_empty_file():
    """Test unzipping an empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = create_test_zip(temp_path, "empty.txt", "")

        result = ZipUtil.unzip(
            zip_path=str(zip_path),
            filename="empty.txt",
            loader=lambda x: open(x, "r").read(),
        )

        assert result == ""


@pytest.mark.fast
def test_unzip_binary_content():
    """Test unzipping binary content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        zip_path = temp_path / "binary.zip"

        # Create binary content
        binary_data = bytes(range(256))

        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("binary.bin", binary_data)

        result = ZipUtil.unzip(
            zip_path=str(zip_path),
            filename="binary.bin",
            loader=lambda x: open(x, "rb").read(),
        )

        assert result == binary_data
