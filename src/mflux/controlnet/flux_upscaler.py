from mflux import Flux1Controlnet
from mflux.config.model_config import ModelConfig


class Flux1Upscaler(Flux1Controlnet):
    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__(
            model_config=ModelConfig.dev_controlnet_upscaler(),
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
