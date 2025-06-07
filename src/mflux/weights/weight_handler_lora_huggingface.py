import os
from pathlib import Path

from huggingface_hub import snapshot_download


class WeightHandlerLoRAHuggingFace:
    @staticmethod
    def download_loras(
        lora_names: list[str] | None = None,
        repo_id: str | None = None,
        cache_dir: str | None = None,
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
        cache_dir: str | None = None,
    ) -> str:
        # Create cache directory if it doesn't exist
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mflux", "loras")

        os.makedirs(cache_dir, exist_ok=True)

        # Check if the file already exists in the cache
        cached_file_path = os.path.join(cache_dir, lora_name)
        if os.path.exists(cached_file_path):
            print(f"Using cached LoRA: {cached_file_path}")
            return cached_file_path

        # Download the LoRA from Hugging Face
        print(f"Downloading LoRA '{lora_name}' from {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{lora_name}*"],
                cache_dir=cache_dir,
            )
        )

        # Find the downloaded file
        for file in download_path.glob(f"**/*{lora_name}*"):
            if file.is_file() and file.suffix in [".safetensors", ".bin"]:
                # Copy or link the file to the cache directory with the expected name
                target_path = os.path.join(cache_dir, lora_name)
                if not os.path.exists(target_path):
                    # Create a symlink or copy the file
                    try:
                        os.symlink(file, target_path)
                    except (OSError, AttributeError):
                        import shutil

                        shutil.copy2(file, target_path)

                print(f"LoRA downloaded to: {target_path}")
                return target_path

        raise FileNotFoundError(f"Could not find LoRA file '{lora_name}' in the downloaded repository.")
