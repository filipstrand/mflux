from flux_1_schnell.config.config import Config
from flux_1_schnell.models.flux import Flux1Schnell

flux = Flux1Schnell("/Users/filipstrand/.cache/FLUX.1-schnell/")

for i in range(0, 10):
    image = flux.generate_image(
        seed=i,
        prompt="Luxury food photograph of an italian Linguine pasta alle vongole dish with lots of clams. It has perfect lighting and a cozy background with big bokeh and shallow depth of field. The mood is a sunset balcony in tuscany.  The photo is taken from the side of the plate. The pasta is shiny with sprinkled parmesan cheese and basil leaves on top. The scene is complemented by a warm, inviting light that highlights the textures and colors of the ingredients, giving it an appetizing and elegant look.",
        config=Config(
            num_inference_steps=2,
        )
    )

    image.save(f"/Users/filipstrand/Desktop/image_{i}.png")
