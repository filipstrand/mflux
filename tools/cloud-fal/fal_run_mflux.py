# run this file with: uvx --with fal==1.69.0 fal run
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
FROM nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_CACHE=/data/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_CHUNK_CACHE_SIZE_BYTES=1000000000000
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=32


ENV PATH=/mflux/bin:$PATH
RUN apt update && \
    apt install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . $HOME/.local/bin/env && \
    uv python install 3.13 && \
    uv venv -p 3.13 /mflux && \
    . /mflux/bin/activate && \
    uv pip install mlx[cuda13]>=0.30.3 mflux==0.15.5 hf_transfer && \
    which python && \
    /mflux/bin/python -c 'import mlx.core as mx; print(mx.default_device()); print(mx.ones([2,1]) * 2)' || true
"""


class MFluxCudaModel(
    fal.App,
    name="mflux-linux-cuda13",
    image=ContainerImage.from_dockerfile_str(DOCKERFILE_STR),
    kind="container",
    keep_alive=300,
    min_concurrency=0,  # Scale to zero when idle
    max_concurrency=2,  # Limit concurrent requests
):
    machine_type = "GPU-B200"

    def setup(self) -> None:
        import random

        from mflux.models.common.config import Config, ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

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
