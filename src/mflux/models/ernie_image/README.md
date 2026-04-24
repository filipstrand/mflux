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

ERNIE-Image and ERNIE-Image-Turbo support LoRA fine-tuning via `mflux-train`.

### Dataset layout

```
dataset/
├── 01.jpg         # training image
├── 01.txt         # caption for 01.jpg  (or use "prompt" inline in the JSON)
├── 02.jpg
├── 02.txt
└── ...
```

### Quick start (Turbo)

```sh
mflux-train --config train_ernie_image_turbo.json
```

See the example config at
`src/mflux/models/common/training/_example/train_ernie_image_turbo.json`.

<details>
<summary>Full JSON reference</summary>

```json
{
  "model": "ernie-image-turbo",
  "data": "/path/to/your/dataset",
  "seed": 42,
  "steps": 8,
  "guidance": 1.0,
  "quantize": 8,
  "low_ram": false,
  "max_resolution": 1024,
  "training_loop": {
    "num_epochs": 200,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "timestep_low": 1,
    "timestep_high": 8
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-4
  },
  "checkpoint": {
    "output_path": "train_ernie_turbo",
    "save_frequency": 50
  },
  "monitoring": {
    "preview_width": 640,
    "preview_height": 368,
    "plot_frequency": 20,
    "generate_image_frequency": 50,
    "preview_prompts": [
      "a photo of sks person smiling, natural light, photorealistic"
    ]
  },
  "lora_layers": {
    "targets": [
      { "module_path": "layers.{block}.self_attention.to_q",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_k",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_v",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_out.0", "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.mlp.gate_proj",           "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.mlp.up_proj",             "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.mlp.linear_fc2",          "blocks": { "start": 0, "end": 35 }, "rank": 16 }
    ]
  }
}
```
</details>

### Training with base ERNIE-Image

Use `"model": "ernie-image"` with more steps and guidance. See the example config at
`src/mflux/models/common/training/_example/train_ernie_image.json`.

```json
{
  "model": "ernie-image",
  "data": "/path/to/your/dataset",
  "seed": 42,
  "steps": 50,
  "guidance": 4.0,
  "quantize": 8,
  "low_ram": false,
  "max_resolution": 1024,
  "training_loop": {
    "num_epochs": 200,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "timestep_low": 5,
    "timestep_high": 45
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-4
  },
  "checkpoint": {
    "output_path": "train_ernie_base",
    "save_frequency": 50
  },
  "monitoring": {
    "preview_width": 640,
    "preview_height": 368,
    "plot_frequency": 20,
    "generate_image_frequency": 50,
    "preview_prompts": [
      "a photo of sks person smiling, natural light, photorealistic"
    ]
  },
  "lora_layers": {
    "targets": [
      { "module_path": "layers.{block}.self_attention.to_q",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_k",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_v",     "blocks": { "start": 0, "end": 35 }, "rank": 16 },
      { "module_path": "layers.{block}.self_attention.to_out.0", "blocks": { "start": 0, "end": 35 }, "rank": 16 }
    ]
  }
}
```

### Dry-run validation

```sh
mflux-train --config train_ernie_image_turbo.json --dry-run
```

### Saving the model

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

Blocks run from `0` to `35` (36 total). A lighter config targeting only attention (`to_q/k/v/out`) on all 36 layers is a good starting point; add MLP layers if the concept requires broader stylistic changes.
