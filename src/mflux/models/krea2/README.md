# Krea 2

This directory contains MFLUX's MLX implementation of **Krea 2** (the
[`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) release).

Krea 2 is a single-stream MMDiT text-to-image model built on the Qwen-Image
stack: it reuses the **Qwen-Image VAE** and conditions on a 12-layer hidden-state
tap from a **Qwen3-VL-4B** text encoder. The Turbo variant is distilled and
produces high-quality images in 8 steps.

## Example

```sh
mflux-generate-krea2 \
  --prompt "a photograph of a red fox sitting in a sunlit forest clearing, sharp focus, bokeh" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --steps 8 \
  -q 8
```

Weights download automatically from [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo)
on first run (accept the model's terms and set a Hugging Face token if prompted).
No `--model` is needed; pass `--model /path/to/local/dir` only to use a local copy.

Turbo defaults: 8 steps, guidance 1.0 (CFG off), `er_sde` sampler. The plain
flow-matching Euler sampler — which matches the official diffusers
`FlowMatchEulerDiscreteScheduler` — is available via `--scheduler euler`.

Standard CLI options are supported: `--metadata`, `--stepwise-image-output-dir`,
`--lora-paths` / `--lora-scales`, and multiple `--seed` values. Image conditioning
(edit / reference) is not yet implemented.

## LoRA

Krea 2 supports community LoRAs via `--lora-paths` (train on
[`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw), run on Turbo).
Paths can be local files, Hugging Face repos, or `org/repo:filename.safetensors`
when a repo ships multiple adapters.

```sh
mflux-generate-krea2 \
  --prompt "a misty forest path at dawn, painterly brush strokes" \
  --lora-paths krea/Krea-2-LoRA-darkbrush \
  --lora-scales 1.0 \
  --steps 8 \
  -q 8
```

Supported export formats include official Krea (`transformer.*`), diffusers/PEFT
(`base_model.model.*`), Comfy (`diffusion_model.*`), and flat `lora_unet_*` keys.
LoKr adapters with `.magnitude` tensors are only partially applied today.

> [!NOTE]
> Krea 2 ships as a diffusers repo whose `transformer/` subdir is *diffusers*-format
> (different key names). MFLUX loads the native single-file `turbo.safetensors` from
> the repo root instead, so the diffusers transformer shards are never downloaded.

## Quantization caching

Save a quantized model once and reload it without re-quantizing:

```sh
mflux-save --model krea/Krea-2-Turbo --quantize 8 --path /path/to/krea-2-mlx-q8
mflux-generate-krea2 --model /path/to/krea-2-mlx-q8 --prompt "..." --steps 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.krea2 import Krea2

model = Krea2(
    model_config=ModelConfig.krea2(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="a photograph of a red fox sitting in a sunlit forest clearing, sharp focus, bokeh",
    num_inference_steps=8,
    width=1024,
    height=1024,
    guidance=1.0,
)
image.save("krea2_fox.png")
```

</details>

## Architecture

- **Transformer**: 28-layer single-stream MMDiT — hidden 6144, GQA (48 query /
  12 KV heads, head_dim 128), SwiGLU, 3-axis Flux-style RoPE `[32, 48, 48]`,
  per-head QK-norm + sigmoid-gated attention, AdaLN-single 6-way modulation, and
  a `txtfusion` adapter that fuses the 12 text-encoder hidden states.
- **Text encoder**: Qwen3-VL-4B, 12-layer tap `[2, 5, …, 35]` flattened
  layer-major; the chat-template prefix is stripped so only prompt tokens
  condition the DiT.
- **VAE**: Qwen-Image VAE (Wan2.1 16-channel latent).
