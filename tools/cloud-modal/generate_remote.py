import modal

app = modal.App("mflux-on-linux")

"""
One time warmup of model-store layout:
    - modal shell --volume model-store
    - python -m pip install huggingface_hub hf_transfer
    - export HF_HUB_ENABLE_HF_TRANSFER=1
    - export HF_HUB_CACHE="/mnt/models-store/hf_hub"
    - export HF_TOKEN=<your token>
    - mkdir -p $HF_HUB_CACHE
    - hf download black-forest-labs/FLUX.1-schnell
    - confirm existence of: /mnt/models-store/hf_hub/models--black-forest-labs--FLUX.1-schnell
"""
model_store_volume = modal.Volume.from_name("model-store")
model_store_path = "/mnt/models-store"

output_data_volume = modal.Volume.from_name("output-data")
output_data_path = "/mnt/output-data"

# Demonstrate CPU only for now
# TODO: Linux GPU/CUDA blocker https://github.com/ml-explore/mlx/issues/2519
mlx_on_cpu_image = modal.Image.from_registry(
    "debian:stable-20250811-slim", add_python="3.11"
).apt_install(
    ["git"]
).uv_pip_install(
    "mlx[cpu]==0.28.0",
    "git+https://github.com/anthonywu/mflux.git@mlx-on-clouds"
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_CACHE": "/mnt/models-store/hf_hub"
})  # fmt: off


@app.function(
    image=mlx_on_cpu_image,
    # init secret via: modal secret create my-huggingface-secret HF_TOKEN="hf_...value..."
    secrets=[
        modal.Secret.from_name("my-huggingface-secret")
    ],
    timeout=3600,  # one hour, CPU = slow
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

    os.environ["CPU_MODE"] = "true"

    import random
    import time
    from pathlib import Path

    from mflux.config.config import Config
    from mflux.config.model_config import ModelConfig
    from mflux.flux.flux import Flux1

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
#   modal token new  # one time setup
#   modal run
#
# see https://modal.com/docs for advanced usage
