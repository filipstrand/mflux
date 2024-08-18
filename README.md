
![image](src/flux_1_schnell/assets/logo.png)
*A MLX port of FLUX based on the Huggingface Diffusers implementation.*

### About

Run the powerful [FLUX](https://blackforestlabs.ai/#get-flux) models from [Black Forest Labs](https://blackforestlabs.ai) locally on your Mac!

### Philosophy

MFLUX (MacFLUX) is a line-by-line port of the FLUX implementation in the [Huggingface Diffusers](https://github.com/huggingface/diffusers) library to [Apple MLX](https://github.com/ml-explore/mlx). 
MFLUX is purposefully kept minimal and explicit - Network architectures are hardcoded and no config files are used
except for the tokenizers. The aim is to have a tiny codebase with the single purpose of expressing these models 
(thereby avoiding too many abstractions). MFLUX priorities readability over generality and performance.

All models are implemented from scratch in MLX and only the tokenizers are used via the 
[Huggingface Transformers](https://github.com/huggingface/transformers) library. Other than that, there are only minimal dependencies 
like [Numpy](https://numpy.org) and [Pillow](https://pypi.org/project/pillow/) for simple image post-processing.

### Models

- [x] FLUX.1-Scnhell
- [ ] FLUX.1-Dev

### Installation
1. Clone the repo:
    ```
    git clone git@github.com:filipstrand/mflux.git
    ```
2. Navigate to the project and set up a virtual environment:
   ```
   cd mflux && python3 -m venv .venv && source .venv/bin/activate
   ``` 
3. Install the required dependencies:
    ```
   pip install -r requirements.txt
   ```
### Generating an image

Run the provided [main.py](main.py) 
```
python main.py
```

or make a new separate script like the following

```python
import sys

sys.path.append("/path/to/mflux/src")

from flux_1_schnell.config.config import Config
from flux_1_schnell.flux import Flux1Schnell
from flux_1_schnell.post_processing.image_util import ImageUtil

flux = Flux1Schnell("black-forest-labs/FLUX.1-schnell")

image = flux.generate_image(
   seed=3,
   prompt="Luxury food photograph of a birthday cake. In the middle it has three candles shaped like letters spelling the word 'MLX'. It has perfect lighting and a cozy background with big bokeh and shallow depth of field. The mood is a sunset balcony in tuscany. The photo is taken from the side of the cake. The scene is complemented by a warm, inviting light that highlights the textures and colors of the ingredients, giving it an appetizing and elegant look.",
   config=Config(
      num_inference_steps=2,
   )
)

ImageUtil.save_image(image, "image.png")
```

If the model is not already downloaded on your machine, it will start the download process and fetch the model weights (~34GB in size for the Schnell model).

Generating a single image (with 2 inference steps, Schnell model) takes between 2 and 3 minutes. This implementation has been tested on two Macbook Pro machines: 
- 2021 M1 Pro (32 GB) 
- 2023 M2 Max (32 GB)

Update:
On faster machines, [@karpathy](https://gist.github.com/awni/a67d16d50f0f492d94a10418e0592bde?permalink_comment_id=5153531#gistcomment-5153531) and [@awni](https://x.com/awnihannun/status/1823515121827897385) have reported times ~20s and below! 

### Equivalent to Diffusers implementation 

There is only a single source of randomness when generating an image: The initial latent array. 
In this implementation, this initial latent is fully deterministically controlled by the input `seed` parameter. 
However, if we were to import a fixed instance of this latent array saved from the Diffusers implementation, then MFLUX will produce an identical image to the Diffusers implementation (assuming a fixed prompt and using the default parameter settings in the Diffusers setup).


The images below illustrate this equivalence. 
In all cases the Schnell model was run for 2 time steps. 
The Diffusers implementation ran in CPU mode. 
The precision for MFLUX can be set in the [Config](src/flux_1_schnell/config/config.py) class. 
There is typically a noticeable but very small difference in the final image when switching between 16bit and 32bit precision.

---
```
Luxury food photograph
```
![image](src/flux_1_schnell/assets/comparison1.jpg)

---
```
detailed cinematic dof render of an old dusty detailed CRT monitor on a wooden desk in a dim room with items around, messy dirty room. On the screen are the letters "FLUX" glowing softly. High detail hard surface render
```
![image](src/flux_1_schnell/assets/comparison2.jpg)

---

```
photorealistic, lotr, A tiny red dragon curled up asleep inside a nest, (Soft Focus) , (f_stop 2.8) , (focal_length 50mm) macro lens f/2. 8, medieval wizard table, (pastel) colors, (cozy) morning light filtering through a nearby window, (whimsical) steam shapes, captured with a (Canon EOS R5) , highlighting (serene) comfort, medieval, dnd, rpg, 3d, 16K, 8K
```
![image](src/flux_1_schnell/assets/comparison3.jpg)

---


```
A weathered fisherman in his early 60s stands on the deck of his boat, gazing out at a stormy sea. He has a thick, salt-and-pepper beard, deep-set blue eyes, and skin tanned and creased from years of sun exposure. He's wearing a yellow raincoat and hat, with water droplets clinging to the fabric. Behind him, dark clouds loom ominously, and waves crash against the side of the boat. The overall atmosphere is one of tension and respect for the power of nature.
```
![image](src/flux_1_schnell/assets/comparison4.jpg)

---

```
Luxury food photograph of an italian Linguine pasta alle vongole dish with lots of clams. It has perfect lighting and a cozy background with big bokeh and shallow depth of field. The mood is a sunset balcony in tuscany.  The photo is taken from the side of the plate. The pasta is shiny with sprinkled parmesan cheese and basil leaves on top. The scene is complemented by a warm, inviting light that highlights the textures and colors of the ingredients, giving it an appetizing and elegant look.
```
![image](src/flux_1_schnell/assets/comparison5.jpg)

---



### Current limitations

- Images are generated one by one.
- Negative prompts not supported.

### TODO

- FLUX Dev implementation
- LoRA adapters
- Command line args