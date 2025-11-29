import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR


class LoRALibrary:
    _registry: dict[str, Path] = {}

    @staticmethod
    def get_path(path_or_name: str) -> str:
        path = Path(path_or_name)
        if path.exists():
            return str(path)

        if path_or_name in LoRALibrary._registry:
            return str(LoRALibrary._registry[path_or_name])

        # HuggingFace collection format: "repo_id:filename" (e.g., "ali-vilab/In-Context-LoRA:storyboard.safetensors")
        if ":" in path_or_name and "/" in path_or_name:
            repo_id, filename = path_or_name.split(":", 1)
            return LoRALibrary._download_from_collection(repo_id, filename)

        # HuggingFace repo_id format (contains exactly one /)
        if "/" in path_or_name and path_or_name.count("/") == 1:
            return LoRALibrary._download_from_huggingface(path_or_name)

        raise FileNotFoundError(
            f"LoRA file not found: '{path_or_name}'. File does not exist and is not in the LoRA library."
        )

    @staticmethod
    def resolve_paths(paths: list[str] | None) -> list[str]:
        if not paths:
            return []
        return [resolved for path in paths if (resolved := LoRALibrary._try_resolve_path(path))]

    @staticmethod
    def resolve_scales(scales: list[float] | None) -> list[float]:
        return scales or []

    @staticmethod
    def get_registry() -> dict[str, Path]:
        return LoRALibrary._registry.copy()

    @staticmethod
    def _try_resolve_path(path: str) -> str | None:
        try:
            return LoRALibrary.get_path(path)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            return None

    @staticmethod
    def _download_from_huggingface(repo_id: str) -> str:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading LoRA from HuggingFace: {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.safetensors"],
                cache_dir=str(cache_path),
            )
        )

        safetensor_files = list(download_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors file found in HuggingFace repo: {repo_id}")

        lora_path = str(safetensor_files[0])
        print(f"LoRA downloaded: {lora_path}")
        return lora_path

    @staticmethod
    def _download_from_collection(repo_id: str, filename: str) -> str:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        # Check if already cached
        cached_file_path = cache_path / filename
        if cached_file_path.exists() and cached_file_path.is_file():
            try:
                # Verify the file is actually readable (catches broken symlinks)
                with open(cached_file_path, "rb") as f:
                    f.read(1)
                print(f"Using cached LoRA: {cached_file_path}")
                return str(cached_file_path)
            except (OSError, IOError):
                # File exists but is not readable, re-download
                print(f"Cached LoRA file is corrupted, re-downloading: {cached_file_path}")
                try:
                    cached_file_path.unlink()
                except OSError:
                    pass

        # Download the specific file from the collection
        print(f"Downloading LoRA '{filename}' from {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{filename}*"],
                cache_dir=str(cache_path),
            )
        )

        # Find the downloaded file
        found_files = list(download_path.glob(f"**/*{filename}*"))
        for file in found_files:
            if file.is_file() and file.suffix in [".safetensors", ".bin"]:
                # Ensure the target has correct extension
                if not filename.endswith(file.suffix):
                    target_name = f"{filename}{file.suffix}"
                else:
                    target_name = filename

                target_path = cache_path / target_name
                if not target_path.exists():
                    try:
                        target_path.symlink_to(file)
                    except (OSError, AttributeError):
                        shutil.copy2(file, target_path)

                print(f"LoRA downloaded: {target_path}")
                return str(target_path)

        raise FileNotFoundError(f"Could not find LoRA file '{filename}' in repository {repo_id}")

    @staticmethod
    def discover_files(library_paths: list[Path]) -> dict[str, Path]:
        lora_files = {}

        # Process in reverse so earlier paths take precedence
        for library_path in reversed(library_paths):
            if not library_path.exists() or not library_path.is_dir():
                continue

            for safetensor_path in library_path.rglob("*.safetensors"):
                basename = safetensor_path.stem

                # Skip digit-only names in transformer directories
                if basename.isdigit() and safetensor_path.parent.name == "transformer":
                    continue

                lora_files[basename] = safetensor_path.resolve()

        return lora_files

    @staticmethod
    def _initialize_registry() -> None:
        library_path_env = os.environ.get("LORA_LIBRARY_PATH")
        if library_path_env:
            library_paths = [Path(p.strip()) for p in library_path_env.split(":") if p.strip()]
            LoRALibrary._registry = LoRALibrary.discover_files(library_paths)


# Initialize on module import
LoRALibrary._initialize_registry()
