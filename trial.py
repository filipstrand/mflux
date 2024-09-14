import os
import sys
from mflux.post_processing.image_util import ImageUtil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mflux.config.model_config import ModelConfig
from mflux.config.config import ConfigControlnet
from mflux.controlnet.flux_controlnet import Flux1Controlnet

prompt = "Luxury picture of food"


# Load the model
flux = Flux1Controlnet(
    model_config=ModelConfig.from_alias("dev"),
    quantize=8,
    lora_paths=["Flux_1_Dev_LoRA_Paper-Cutout-Style.safetensors"]
)

control_image = ImageUtil.load_image("image_1.png")

# Generate an image
image = flux.generate_image(
    seed=2500,
    prompt=prompt,
    control_image=control_image,
    config=ConfigControlnet(
        num_inference_steps=10,
        height=256,
        width=512,
        guidance=3.5,
        controlnet_strength=0.7,
    )
)

# Save the image
image.save(path="image.png", export_json_metadata=False)