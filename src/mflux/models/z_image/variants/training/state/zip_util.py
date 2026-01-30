import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


class ZipUtil:
    # Bytes per megabyte constant for readable size conversions
    BYTES_PER_MB = 1024 * 1024

    # Maximum uncompressed size for any single extracted file (100MB)
    # This protects against zip bomb attacks where small compressed files
    # expand to massive uncompressed data
    MAX_UNCOMPRESSED_SIZE_MB = 100

    # Maximum compression ratio (100:1)
    # Normal compression ratios are typically 2:1 to 10:1
    # Zip bombs can have ratios of 1000:1 or higher
    # 100:1 allows for highly compressible data (e.g., sparse tensors)
    # while still catching most zip bomb attacks
    MAX_COMPRESSION_RATIO = 100

    @staticmethod
    def unzip(
        zip_path: str,
        filename: str,
        loader: Callable[[str], T],
        max_uncompressed_mb: float | None = None,
    ) -> T:
        """Extract a file from a zip archive and process it with a loader function.

        Args:
            zip_path: Path to the zip archive
            filename: Name of the file to extract from the archive
            loader: Function to process the extracted file
            max_uncompressed_mb: Maximum allowed uncompressed size in MB.
                                 Defaults to MAX_UNCOMPRESSED_SIZE_MB (100MB).

        Returns:
            Result of calling loader on the extracted file

        Raises:
            FileNotFoundError: If zip_path doesn't exist
            ValueError: If uncompressed size exceeds limit (zip bomb protection)
            KeyError: If filename not found in archive
        """
        # Validate zip_path exists and is a file
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"Archive file not found: {zip_path}")
        if not zip_file.is_file():
            raise ValueError(f"Archive path is not a file: {zip_path}")

        max_size_mb = max_uncompressed_mb if max_uncompressed_mb is not None else ZipUtil.MAX_UNCOMPRESSED_SIZE_MB
        max_size_bytes = int(max_size_mb * ZipUtil.BYTES_PER_MB)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    # Security: Check uncompressed size BEFORE extraction (zip bomb protection)
                    try:
                        file_info = zipf.getinfo(filename)
                    except KeyError:
                        raise KeyError(f"File '{filename}' not found in archive: {zip_path}")

                    # Security: Validate that the entry name matches expected filename
                    # This prevents path traversal via malicious archive entries
                    if file_info.filename != filename:
                        raise ValueError(
                            f"Security: Archive entry name mismatch. Expected '{filename}', got '{file_info.filename}'"
                        )

                    # Security: Reject entries with path traversal sequences
                    if ".." in file_info.filename or file_info.filename.startswith("/"):
                        raise ValueError(f"Security: Path traversal detected in archive entry: {file_info.filename}")

                    uncompressed_size = file_info.file_size
                    if uncompressed_size > max_size_bytes:
                        raise ValueError(
                            "Security: Archive file exceeds maximum allowed size. Possible zip bomb attack."
                        )

                    # Security: Compression ratio check for zip bomb detection
                    compressed_size = file_info.compress_size
                    if compressed_size > 0:
                        # Use integer arithmetic to avoid floating-point overflow
                        # Check: uncompressed_size > compressed_size * MAX_COMPRESSION_RATIO
                        if uncompressed_size > compressed_size * ZipUtil.MAX_COMPRESSION_RATIO:
                            raise ValueError(
                                "Security: Archive rejected due to suspicious compression characteristics. "
                                "Possible zip bomb attack."
                            )
                    elif uncompressed_size > 0:
                        # Security: Handle stored (uncompressed) files with compress_size=0
                        # This is unusual and could indicate a malformed archive
                        raise ValueError(
                            f"Security: Invalid archive entry - uncompressed size {uncompressed_size} "
                            f"but compressed size is 0. Possible malformed or malicious archive."
                        )

                    # Security: Extract to controlled path to prevent path traversal
                    # Use only the basename of the filename for extraction
                    safe_filename = Path(filename).name
                    extracted_path = Path(temp_dir) / safe_filename

                    # Stream content to controlled path (bounds memory usage for large files)
                    # Using 64KB chunks avoids loading entire file into memory at once
                    with zipf.open(file_info) as source:
                        with open(extracted_path, "wb") as target:
                            shutil.copyfileobj(source, target, length=65536)

                    return loader(str(extracted_path))

            except zipfile.BadZipFile as e:
                raise ValueError(f"Invalid or corrupted ZIP archive: {zip_path}") from e
