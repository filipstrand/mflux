import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1_schnell.config.config import Config
from flux_1_schnell.flux import Flux1
from flux_1_schnell.post_processing.image_util import ImageUtil

flux = Flux1("black-forest-labs/FLUX.1-schnell", max_sequence_length=256)

image = flux.generate_image(
    seed=3,
    prompt="Luxury food photograph of a birthday cake. In the middle it has three candles shaped like letters spelling the word 'MLX'. It has perfect lighting and a cozy background with big bokeh and shallow depth of field. The mood is a sunset balcony in tuscany. The photo is taken from the side of the cake. The scene is complemented by a warm, inviting light that highlights the textures and colors of the ingredients, giving it an appetizing and elegant look.",
    config=Config(
        num_inference_steps=2,
        height=768,
        width=1360,
    )
)

ImageUtil.save_image(image, "image.png")
