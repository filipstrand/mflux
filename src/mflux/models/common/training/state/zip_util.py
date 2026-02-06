import os
import tempfile
from pathlib import Path
from typing import Callable
from zipfile import ZipFile


class ZipUtil:
    @staticmethod
    def unzip(zip_path: str | Path | None, filename: str, loader: Callable):
        if not zip_path:
            raise ValueError("zip_path cannot be None")
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found at: {zip_path}")

        if not callable(loader):
            raise ValueError("The loader must be callable (e.g., a function or lambda).")

        with ZipFile(zip_path, "r") as zipf:
            # Normalize file names in the archive by stripping folders
            namelist = {os.path.basename(name): name for name in zipf.namelist()}
            if filename not in namelist:
                raise FileNotFoundError(f"File '{filename}' not found in the ZIP archive.")

            archive_filename = namelist[filename]
            file_data = zipf.read(archive_filename)

            temp_file = tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False)
            try:
                temp_file.write(file_data)
                temp_file.flush()
                temp_file.close()
                return loader(temp_file.name)
            finally:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

    @staticmethod
    def extract_all(zip_path: str | Path, output_dir: str | Path):
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found at: {zip_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zipf:
            zipf.extractall(output_dir)
