import shutil
from pathlib import Path

from mflux.ui.defaults import MFLUX_LORA_CACHE_DIR
from mflux.utils.download import snapshot_download


class LoRAHuggingFaceDownloader:
    @staticmethod
    def download_loras(
        lora_names: list[str] | None = None,
        repo_id: str | None = None,
        cache_dir: Path | str | None = None,
        model_name: str = "LoRA",
    ) -> list[str]:
        if not lora_names or not repo_id:
            return []

        lora_paths = []
        for lora_name in lora_names:
            lora_path = LoRAHuggingFaceDownloader.download_lora(
                repo_id=repo_id,
                lora_name=lora_name,
                cache_dir=cache_dir,
                model_name=model_name,
            )
            lora_paths.append(lora_path)

        return lora_paths

    @staticmethod
    def download_lora(
        repo_id: str,
        lora_name: str,
        cache_dir: Path | str | None = None,
        model_name: str = "LoRA",  # For logging purposes
    ) -> str:
        # Ensure cache_dir is a Path object
        if cache_dir is None:
            cache_path = MFLUX_LORA_CACHE_DIR
        else:
            cache_path = Path(cache_dir)

        cache_path.mkdir(parents=True, exist_ok=True)

        # Check if already cached
        cached_file_path = cache_path / lora_name
        if cached_file_path.exists() and cached_file_path.is_file():
            try:
                # Verify the file is actually readable (catches broken symlinks)
                with open(cached_file_path, "rb") as f:
                    f.read(1)  # Try to read just 1 byte to verify it works
                print(f"Using cached {model_name} LoRA: {cached_file_path}")
                return str(cached_file_path)
            except (OSError, IOError):
                # File exists but is not readable (broken symlink, permissions, etc.)
                print(f"Cached {model_name} LoRA file is corrupted or inaccessible, re-downloading: {cached_file_path}")
                try:
                    cached_file_path.unlink()  # Remove the broken file/symlink
                except OSError:
                    pass  # Ignore if we can't remove it

        # Download the LoRA from Hugging Face
        print(f"Downloading {model_name} LoRA '{lora_name}' from {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{lora_name}*"],
                cache_dir=str(cache_path),
            )
        )

        # Find the downloaded file
        print(f"🔍 Searching for downloaded files in: {download_path}")
        found_files = list(download_path.glob(f"**/*{lora_name}*"))
        print(f"📁 Found files matching pattern: {found_files}")

        for file in found_files:
            print(f"📄 Checking file: {file} (suffix: {file.suffix}, size: {file.stat().st_size} bytes)")
            if file.is_file() and file.suffix in [".safetensors", ".bin"]:
                # Ensure the target path has the correct extension
                if not lora_name.endswith(file.suffix):
                    target_name = f"{lora_name}{file.suffix}"
                else:
                    target_name = lora_name

                target_path = cache_path / target_name
                if not target_path.exists():
                    # Create a symlink or copy the file
                    try:
                        target_path.symlink_to(file)
                        print(f"🔗 Created symlink: {target_path} -> {file}")
                    except (OSError, AttributeError):
                        shutil.copy2(file, target_path)
                        print(f"📋 Copied file: {file} -> {target_path}")

                print(f"{model_name} LoRA downloaded to: {target_path}")
                return str(target_path)

        raise FileNotFoundError(f"Could not find {model_name} LoRA file '{lora_name}' in the downloaded repository.")
