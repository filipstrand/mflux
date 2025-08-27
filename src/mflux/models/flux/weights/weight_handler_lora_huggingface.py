import shutil
from pathlib import Path

from mflux.ui.defaults import MFLUX_LORA_CACHE_DIR
from mflux.weights.download import snapshot_download


class WeightHandlerLoRAHuggingFace:
    @staticmethod
    def download_loras(
        lora_names: list[str] | None = None,
        repo_id: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> list[str]:
        if repo_id is None:
            return []

        lora_paths = []
        if lora_names:
            for lora_name in lora_names:
                lora_path = WeightHandlerLoRAHuggingFace.download_lora(
                    repo_id=repo_id,
                    lora_name=lora_name,
                    cache_dir=cache_dir,
                )
                lora_paths.append(lora_path)

        return lora_paths

    @staticmethod
    def download_lora(
        repo_id: str,
        lora_name: str,
        cache_dir: Path | str | None = None,
    ) -> str:
        # Ensure cache_dir is a Path object
        if cache_dir is None:
            cache_path = MFLUX_LORA_CACHE_DIR
        else:
            cache_path = Path(cache_dir)

        cache_path.mkdir(parents=True, exist_ok=True)

        # Check if the file already exists in the cache
        cached_file_path = cache_path / lora_name
        if cached_file_path.exists() and cached_file_path.is_file():
            try:
                # Verify the file is actually readable (catches broken symlinks)
                with open(cached_file_path, "rb") as f:
                    f.read(1)  # Try to read just 1 byte to verify it works
                print(f"Using cached LoRA: {cached_file_path}")
                return str(cached_file_path)
            except (OSError, IOError):
                # File exists but is not readable (broken symlink, permissions, etc.)
                print(f"Cached LoRA file is corrupted or inaccessible, re-downloading: {cached_file_path}")
                try:
                    cached_file_path.unlink()  # Remove the broken file/symlink
                except OSError:
                    pass  # Ignore if we can't remove it

        # Download the LoRA from Hugging Face
        print(f"Downloading LoRA '{lora_name}' from {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{lora_name}*"],
                cache_dir=str(cache_path),
            )
        )

        # Find the downloaded file
        for file in download_path.glob(f"**/*{lora_name}*"):
            if file.is_file() and file.suffix in [".safetensors", ".bin"]:
                # Copy or link the file to the cache directory with the expected name
                target_path = cache_path / lora_name
                if not target_path.exists():
                    # Create a symlink or copy the file
                    try:
                        target_path.symlink_to(file)
                    except (OSError, AttributeError):
                        shutil.copy2(file, target_path)

                print(f"LoRA downloaded to: {target_path}")
                return str(target_path)

        raise FileNotFoundError(f"Could not find LoRA file '{lora_name}' in the downloaded repository.")
