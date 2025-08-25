import fal
from fal.container import ContainerImage
from fal.toolkit import (
    File as FalFile,
    Image as FalImage,
)
from pydantic import BaseModel, Field


class Input(BaseModel):
    prompt: str = Field(
        description="The image prompt",
        examples=[
            "A charming Parisian street scene with its vibrant red awning and outdoor seating area, surrounded by quant shops and lush greenery under a bright blue sky."
        ],
    )
    guidance: float = Field(description="CFG Scale", default=3.5)
    steps: int = Field(description="Number of Inference Steps", default=20)
    seed: int = Field(description="Generation Seed", default=42)


class Output(BaseModel):
    image: FalImage = Field(description="The mflux-generated image")


DOCKERFILE_STR = """
# this is python3.11 from 2025-08-19
FROM python@sha256:0dab67c838514eef83e6c9d2c0e53e960fc94237635e8996d08caeec98937abc
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_CACHE=/data/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_CHUNK_CACHE_SIZE_BYTES=1000000000000
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=32

ENV CPU_MODE=1

RUN python -m pip install -U pip && \
    python -m pip install 'mlx[cpu]==0.28.0' hf_transfer 'git+https://github.com/anthonywu/mflux.git@mlx-on-clouds' && \
    python -m pip list && python -V && \
    python -c 'import mlx.core as mx; print(mx.default_device()); print(mx.ones([2,1]) * 2)'
"""


class MFluxCpuModel(
    fal.App,
    name="mflux-linux-cpu",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE_STR),
    kind="container",
    keep_alive=300,
    min_concurrency=0,  # Scale to zero when idle
    max_concurrency=2,  # Limit concurrent requests
):
    machine_type = "L"

    def setup(self) -> None:
        import random

        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.flux.flux import Flux1

        self.flux = Flux1(model_config=ModelConfig.from_name("schnell"))
        print("Warming up model with first run")
        self.flux.generate_image(
            seed=random.randint(0, int(1e9)),
            prompt="Fluffy letters F A L in a cloud formation. Behind the clouds is a clear blue sky.",
            config=Config(num_inference_steps=1, height=512, width=512, guidance=3.5),
        )

    @fal.endpoint("/")
    def image_to_video(self, input: Input) -> Output:
        import time
        from pathlib import Path

        from mflux.config.config import Config

        image = self.flux.generate_image(
            seed=input.seed,
            prompt=input.prompt,
            config=Config(num_inference_steps=input.steps, height=512, width=512, guidance=input.guidance),
        )
        output_dir = Path("/data/mflux-output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"output-{int(time.time() * 1e6)}.png"
        image.save(output_path)
        return Output(image=FalFile.from_path(output_path))
