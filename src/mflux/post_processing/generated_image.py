import importlib
import pathlib
import typing as t

import mlx.core as mx
import PIL.Image
import toml

from mflux.config.model_config import ModelConfig
from mflux.ui.defaults import DEFAULT_SCHEDULER


class GeneratedImage:
    def __init__(
        self,
        image: PIL.Image.Image,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        precision: mx.Dtype,
        quantization: int,
        generation_time: float,
        lora_paths: list[str],
        lora_scales: list[float],
        controlnet_image_path: str | pathlib.Path | None = None,
        controlnet_strength: float | None = None,
        image_path: str | pathlib.Path | None = None,
        image_strength: float | None = None,
        noise_scheduler: str = DEFAULT_SCHEDULER,
    ):
        self.image = image
        self.model_config = model_config
        self.seed = seed
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.precision = precision
        self.quantization = quantization
        self.generation_time = generation_time
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.controlnet_image_path = controlnet_image_path
        self.controlnet_strength = controlnet_strength
        self.image_path = image_path
        self.image_strength = image_strength
        self.noise_scheduler = noise_scheduler

    def get_right_half(self) -> "GeneratedImage":
        # Calculate the coordinates for the right half
        width, height = self.image.size
        right_half = self.image.crop((width // 2, 0, width, height))

        # Create a new GeneratedImage with the right half and the same metadata
        return GeneratedImage(
            image=right_half,
            model_config=self.model_config,
            seed=self.seed,
            prompt=self.prompt,
            steps=self.steps,
            guidance=self.guidance,
            precision=self.precision,
            quantization=self.quantization,
            generation_time=self.generation_time,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=self.controlnet_image_path,
            controlnet_strength=self.controlnet_strength,
            image_path=self.image_path,
            image_strength=self.image_strength,
            noise_scheduler=self.noise_scheduler,
        )

    def save(
        self,
        path: t.Union[str, pathlib.Path],
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        from mflux import ImageUtil

        ImageUtil.save_image(self.image, path, self._get_metadata(), export_json_metadata, overwrite)

    def _get_metadata(self) -> dict:
        """Generate metadata for reference as well as input data for
        command line --config-from-metadata arg in future generations.
        """
        return {
            # mflux_version is used by future metadata readers
            # to determine supportability of metadata-derived workflows
            "mflux_version": GeneratedImage.get_version(),
            "model": self.model_config.model_name,
            "base_model": str(self.model_config.base_model),
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance if self.model_config.supports_guidance else None,
            "precision": str(self.precision),
            "quantize": self.quantization,
            "generation_time_seconds": round(self.generation_time, 2),
            "lora_paths": [str(p) for p in self.lora_paths] if self.lora_paths else None,
            "lora_scales": [round(scale, 2) for scale in self.lora_scales] if self.lora_scales else None,
            "image_path": str(self.image_path) if self.image_path else None,
            "image_strength": self.image_strength if self.image_path else None,
            "controlnet_image_path": str(self.controlnet_image_path) if self.controlnet_image_path else None,
            "controlnet_strength": round(self.controlnet_strength, 2) if self.controlnet_strength else None,
            "noise_scheduler": str(self.noise_scheduler),
            "prompt": self.prompt,
        }

    @staticmethod
    def get_version() -> str:
        version = GeneratedImage._get_version_from_toml()
        if version:
            return version

        # Fallback to installed package version
        try:
            return str(importlib.metadata.version("mflux"))
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    @staticmethod
    def _get_version_from_toml() -> str | None:
        # Search for pyproject.toml by traversing up from the current working directory
        current_dir = pathlib.Path(__file__).resolve().parent
        for parent in current_dir.parents:
            pyproject_path = parent / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    pyproject_data = toml.load(pyproject_path)
                    return pyproject_data.get("project", {}).get("version")
                except (toml.TomlDecodeError, KeyError, TypeError):
                    return None
        return None
