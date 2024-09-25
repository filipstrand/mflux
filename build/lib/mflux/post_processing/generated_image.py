import importlib

import PIL.Image
import mlx.core as mx

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
        controlnet_image_path: str | None = None,
        controlnet_strength: float | None = None,
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
        self.controlnet_image = controlnet_image_path
        self.controlnet_strength = controlnet_strength

    def save(self, path: str, export_json_metadata: bool = False) -> None:
        from mflux import ImageUtil

        ImageUtil.save_image(self.image, path, self._get_metadata(), export_json_metadata)

    def _get_metadata(self) -> dict:
        return {
            "mflux_version": str(GeneratedImage.get_version()),
            "model": str(self.model_config.alias),
            "seed": str(self.seed),
            "steps": str(self.steps),
            "guidance": "None" if self.model_config == ModelConfig.FLUX1_SCHNELL else str(self.guidance),
            "precision": f"{self.precision}",
            "quantization": "None" if self.quantization is None else f"{self.quantization} bit",
            "generation_time": f"{self.generation_time:.2f} seconds",
            "lora_paths": ", ".join(self.lora_paths) if self.lora_paths else "None",
            "lora_scales": ", ".join([f"{scale:.2f}" for scale in self.lora_scales]) if self.lora_scales else "None",
            "prompt": self.prompt,
            "controlnet_image": "None" if self.controlnet_image is None else self.controlnet_image,
            "controlnet_strength": "None" if self.controlnet_strength is None else f"{self.controlnet_strength:.2f}",
        }

    @staticmethod
    def get_version():
        try:
            return importlib.metadata.version("mflux")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
