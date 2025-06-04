import importlib
from pathlib import Path

import mlx.core as mx
import PIL.Image
import toml

from mflux.community.concept_attention.attention_data import ConceptHeatmap
from mflux.config.model_config import ModelConfig


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
        controlnet_image_path: str | Path | None = None,
        controlnet_strength: float | None = None,
        image_path: str | Path | None = None,
        image_strength: float | None = None,
        masked_image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        redux_image_paths: list[str] | list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        concept_heatmap: ConceptHeatmap | None = None,
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
        self.masked_image_path = masked_image_path
        self.depth_image_path = depth_image_path
        self.redux_image_paths = redux_image_paths
        self.redux_image_strengths = redux_image_strengths
        self.concept_heatmap = concept_heatmap

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
            masked_image_path=self.masked_image_path,
            depth_image_path=self.depth_image_path,
            concept_heatmap=self.concept_heatmap,
        )

    def save(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        from mflux import ImageUtil

        ImageUtil.save_image(self.image, path, self._get_metadata(), export_json_metadata, overwrite)

    def save_with_heatmap(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        # Save the main image
        self.save(path=path, export_json_metadata=export_json_metadata, overwrite=overwrite)

        # Save the concept heatmap if available
        if self.concept_heatmap:
            file_path = Path(path)
            heatmap_path = file_path.with_stem(file_path.stem + "_heatmap")
            self.save_concept_heatmap(path=heatmap_path, export_json_metadata=export_json_metadata, overwrite=overwrite)

    def save_concept_heatmap(
        self, path: str | Path, export_json_metadata: bool = False, overwrite: bool = False
    ) -> None:
        if self.concept_heatmap:
            from mflux import ImageUtil

            ImageUtil.save_image(
                image=self.concept_heatmap.image,
                path=path,
                metadata=self.concept_heatmap.get_metadata(),
                export_json_metadata=export_json_metadata,
                overwrite=overwrite,
            )
        else:
            raise ValueError("No concept heatmap available to save")

    def _format_redux_strengths(self) -> list[float] | None:
        if not self.redux_image_strengths:
            return None
        return [round(scale, 2) for scale in self.redux_image_strengths]

    def _get_metadata(self) -> dict:
        return {
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
            "masked_image_path": str(self.masked_image_path) if self.masked_image_path else None,
            "depth_image_path": str(self.depth_image_path) if self.depth_image_path else None,
            "redux_image_paths": str(self.redux_image_paths) if self.redux_image_paths else None,
            "redux_image_strengths": self._format_redux_strengths(),
            "prompt": self.prompt,
        }

    @staticmethod
    def get_version() -> str:
        version = GeneratedImage._get_version_from_toml()
        if version:
            return version

        # Fallback to an installed package version
        try:
            return str(importlib.metadata.version("mflux"))
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    @staticmethod
    def _get_version_from_toml() -> str | None:
        # Search for pyproject.toml by traversing up from the current working directory
        current_dir = Path(__file__).resolve().parent
        for parent in current_dir.parents:
            pyproject_path = parent / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    pyproject_data = toml.load(pyproject_path)
                    return pyproject_data.get("project", {}).get("version")
                except (toml.TomlDecodeError, KeyError, TypeError):
                    return None
        return None
