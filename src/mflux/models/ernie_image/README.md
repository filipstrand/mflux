# ERNIE-Image
This directory contains MFLUX's MLX implementation of **ERNIE-Image** and **ERNIE-Image-Turbo**.

MFLUX supports [ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) and [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) from Baidu. ERNIE-Image is an 8B single-stream Diffusion Transformer for text-to-image generation. ERNIE-Image-Turbo is a distilled variant that produces high-quality images in just 8 steps.

All the standard modes such as img2img, LoRA and quantizations are supported for this model.

> [!NOTE]
> ERNIE-Image tends toward vivid, high-contrast output — a characteristic of Baidu's training data, not a bug in the port. Prompts like "35mm film grain", "analog", or "soft lighting" can soften the look.

> [!NOTE]
> The official ERNIE-Image pipeline includes a **Prompt Enhancer (PE)** — a separate LLM that rewrites short prompts into longer, detailed descriptions before encoding. The model was trained predominantly on PE-expanded prompts, so very short prompts (e.g. "a cat") tend to produce incoherent results. This port does not include the PE. **Use detailed, descriptive prompts for best results.**

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
from mflux.models.ernie_image import ErnieImage

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
  --guidance 4.0 \
  -q 8
```

<details>
<summary>Python API (Base)</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image import ErnieImage

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
    guidance=4.0,
)
image.save("ernie_base.png")
```
</details>

> [!WARNING]
> ERNIE-Image weights are large (~22 GB unquantized). Use `-q 8` (~12 GB) or `-q 4` (~6.4 GB) for reduced memory usage.

## Image-to-image
Pass `--image-path` and `--image-strength` (0.0–1.0) to blend an input image with the denoising process:

```sh
mflux-generate-ernie-image-turbo \
  --prompt "A watercolor painting of the same scene, soft brush strokes, muted colors." \
  --image-path input.jpg \
  --image-strength 0.6 \
  --width 1024 \
  --height 576 \
  --seed 42 \
  --steps 8 \
  -q 8
```

## LoRA

Pre-trained LoRA files (`.safetensors`) can be applied at inference time with `--lora-paths` and `--lora-scales`. Both the `diffusion_model.layers.N.*` format (Lucie / ai-toolkit style) and the `lora_unet_layers_N_*` format (kohya / ComfyUI style) are supported.

```sh
mflux-generate-ernie-image-turbo \
  --prompt "..." \
  --lora-paths /path/to/lora.safetensors \
  --lora-scales 1.0 \
  --steps 8 -q 8
```

## Training

ERNIE-Image and ERNIE-Image-Turbo support LoRA fine-tuning via `mflux-train`. Start from the example configs:

- Turbo: [`train_ernie_image_turbo.json`](../common/training/_example/train_ernie_image_turbo.json)
- Base: [`train_ernie_image.json`](../common/training/_example/train_ernie_image.json)

For the dataset layout and shared training options, see the common training docs ([Training (LoRA)](../common/README.md#training-lora)).

### Dataset layout

```
dataset/
├── 01.jpg         # training image
├── 01.txt         # caption for 01.jpg
├── 02.jpg
├── 02.txt
└── ...
```

### Quick start

```sh
mflux-train --config src/mflux/models/common/training/_example/train_ernie_image_turbo.json
```

Validate the config without training:

```sh
mflux-train --config train_ernie_image_turbo.json --dry-run
```

Save a quantized copy after training:

```sh
mflux-save --model ernie-image-turbo --quantize 8 --path /path/to/save
```

### LoRA target modules

| Module path | Description |
|---|---|
| `layers.{block}.self_attention.to_q` | Query projection |
| `layers.{block}.self_attention.to_k` | Key projection |
| `layers.{block}.self_attention.to_v` | Value projection |
| `layers.{block}.self_attention.to_out.0` | Output projection |
| `layers.{block}.mlp.gate_proj` | FFN gate |
| `layers.{block}.mlp.up_proj` | FFN up |
| `layers.{block}.mlp.linear_fc2` | FFN down |
| `text_proj` | Text→image projection (global, no `blocks`) |
| `time_embedding.linear_1` | Timestep embedding (optional, global) |
| `time_embedding.linear_2` | Timestep embedding (optional, global) |
| `adaln_modulation` | AdaLN modulation (optional, global) |
| `final_norm.linear` | Output norm (optional, global) |

Blocks run from `0` to `35` (36 total). The example configs target attention, MLP, and `text_proj` on all layers — a good default. Add the optional global modules if you need broader stylistic control.
