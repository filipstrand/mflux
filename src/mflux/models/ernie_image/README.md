# ERNIE-Image
This directory contains MFLUX's MLX implementation of **ERNIE-Image** and **ERNIE-Image-Turbo**.

MFLUX supports [ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) and [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) from Baidu. ERNIE-Image is an 8B single-stream Diffusion Transformer for text-to-image generation. ERNIE-Image-Turbo is a distilled variant that produces high-quality images in just 8 steps.

> [!NOTE]
> ERNIE-Image tends toward vivid, high-contrast output — a characteristic of Baidu's training data, not a bug in the port. Prompts like "35mm film grain", "analog", or "soft lighting" can soften the look.

## ERNIE-Image-Turbo Example
ERNIE-Image-Turbo is the recommended starting point. It runs in 8 steps with no classifier-free guidance:

```sh
mflux-generate-ernie-image-turbo \
  --prompt "A serene mountain lake at dawn, mist rising from the water, pine trees reflected on the surface, soft morning light." \
  --width 1024 \
  --height 576 \
  --seed 42 \
  --steps 8 \
  -q 8
```

<details>
<summary>Python API (Turbo)</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.variants.ernie_image import ErnieImage

model = ErnieImage(
    model_config=ModelConfig.ernie_image_turbo(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="A serene mountain lake at dawn, mist rising from the water, pine trees reflected on the surface, soft morning light.",
    num_inference_steps=8,
    width=1024,
    height=576,
    guidance=1.0,
)
image.save("ernie_turbo.png")
```
</details>

## ERNIE-Image (Base) Example
The base SFT model uses more steps and supports classifier-free guidance:

> [!WARNING]
> Base (non-distilled) ERNIE-Image is typically slower. Use ERNIE-Image-Turbo for most tasks.

```sh
mflux-generate-ernie-image \
  --prompt "A serene mountain lake at dawn, mist rising from the water, pine trees reflected on the surface, soft morning light." \
  --width 1024 \
  --height 576 \
  --seed 42 \
  --steps 50 \
  --guidance 5.0 \
  -q 8
```

<details>
<summary>Python API (Base)</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.variants.ernie_image import ErnieImage

model = ErnieImage(
    model_config=ModelConfig.ernie_image(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="A serene mountain lake at dawn, mist rising from the water, pine trees reflected on the surface, soft morning light.",
    num_inference_steps=50,
    width=1024,
    height=576,
    guidance=5.0,
)
image.save("ernie_base.png")
```
</details>

> [!WARNING]
> ERNIE-Image weights are large (~22 GB unquantized). Use `-q 8` (~12 GB) or `-q 4` (~6.4 GB) for reduced memory usage.
