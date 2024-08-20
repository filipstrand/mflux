import os
import re
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.config.config import ConfigImg2Img
from flux_1.flux import Flux1Img2Img
from flux_1.post_processing.image_util import ImageUtil

flux_img2img = Flux1Img2Img.from_alias("schnell")
base_image = ImageUtil.load_image("image.png")

image = flux_img2img.generate_image(
    seed=3,
    prompt="Luxury food photograph",
    base_image=base_image,
    config=ConfigImg2Img(
        num_inference_steps=2,
    )
)

ImageUtil.save_image(image, "image.png")