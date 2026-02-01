import modal

app = modal.App("mflux-on-linux")

"""
Recommendation: do a one time warmup of model-store volume to store hf model caches, and another for your output artifacts:

    - modal volume create output-data
    - modal volume create model-store

First, download one or more of the models onto the model-store volume and verify its integrity:

    - modal shell --volume model-store --add-python 3.13
    - export HF_HUB_ENABLE_HF_TRANSFER=1
    - export HF_HUB_CACHE="/mnt/models-store/hf_hub"
    - export HF_TOKEN=<your token>
    - mkdir -p $HF_HUB_CACHE
    - uvx --with hf_transfer,huggingface_hub hf download black-forest-labs/FLUX.1-schnell
    - uvx hf cache verify black-forest-labs/FLUX.1-schnell
    - uvx cache verify /mnt/models-store/hf_hub/models--black-forest-labs--FLUX.1-schnell

    you should see something like:

    ✅ Verified 28 file(s) for 'black-forest-labs/FLUX.1-schnell' (model) in /mnt/models-store/hf_hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9
      All checksums match.
"""

model_store_volume = modal.Volume.from_name("model-store")
model_store_path = "/mnt/models-store"

output_data_volume = modal.Volume.from_name("output-data")
output_data_path = "/mnt/output-data"

# MLX is supported on CUDA 13+ with mlx[cuda13]>=0.30.3
mlx_on_cuda13 = modal.Image.from_registry(
    "nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04", add_python="3.13",
).apt_install(
    ["git"]
).uv_pip_install(
    "mlx[cuda13]>=0.30.3",
    "mflux==0.15.5",
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_CACHE": "/mnt/models-store/hf_hub"
})  # fmt: off


@app.function(
    # using Blackwell-class machine, because MLX Cuda support was initially tested on a DGX Spark
    gpu="B200",
    image=mlx_on_cuda13,
    # init secret via: modal secret create my-huggingface-secret HF_TOKEN="hf_...value..."
    secrets=[
        modal.Secret.from_name("my-huggingface-secret")
    ],
    timeout=3600,
    volumes={
        model_store_path: model_store_volume,
        output_data_path: output_data_volume
    },
)  # fmt: off
def generate_image():
    import os

    # these are useful assertions when starting a new project
    assert os.environ.get("HF_TOKEN", "").startswith("hf_")  # you set the secrets properly
    assert os.environ["HF_HUB_CACHE"] == "/mnt/models-store/hf_hub"  # you defined ENV in docker build

    import random
    import time
    from pathlib import Path

    from mflux.models.common.config import Config, ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1(model_config=ModelConfig.from_name("schnell"))

    output_path = Path(f"/mnt/output-data/output-{int(time.time() * 1e6)}.png")

    image = flux.generate_image(
        seed=random.randint(0, int(1e9)),
        prompt="Fluffy letters M L X in a cloud formation. Behind the clouds is a clear blue sky.",
        config=Config(num_inference_steps=15, height=1024, width=1024, guidance=3.5),
    )
    image.save(output_path)
    print(output_path)


# run this script with:
#   modal run
# ✓ Initialized. View run at https://modal.com/apps/anthonywu/main/ap-pUXeO8seEe0gxUZ04PKLtM
# Stopping app - uncaught exception raised locally: NotFoundError("Volume 'output-data' not found in environment 'main'").
# Building image im-aLYBWnjqNWtJGEVDUnAv7n
# => Step 0: FROM nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04
#
# see https://modal.com/docs for advanced usage
