import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.flux.flux import Flux1


prompt = "Luxury food photograph"

# Load the model
flux = Flux1(
    model_config=ModelConfig.from_alias("dev"),
    quantize=None,
    local_path=None,
    lora_paths=["diffusion_pytorch_model.safetensors"],
    lora_scales=None,
)

# Generate an image
image = flux.generate_image(
    seed=3,
    prompt=prompt,
    config=Config(
        num_inference_steps=10,
        height=256,
        width=512,
        guidance=3.5,
    )
)

# Save the image
image.save(path="image.png", export_json_metadata=False)