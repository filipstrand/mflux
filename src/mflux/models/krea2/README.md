# Krea 2

This directory contains MFLUX's MLX implementation of **Krea 2** (the
[`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo) release).

Krea 2 is a single-stream MMDiT text-to-image model built on the Qwen-Image
stack: it reuses the **Qwen-Image VAE** and conditions on a 12-layer hidden-state
tap from a **Qwen3-VL-4B** text encoder. The Turbo variant is distilled and
produces high-quality images in 8 steps.

## Weights

Krea 2 ships as a diffusers repo whose `transformer/` subdir is *diffusers*-format
(different key names). MFLUX loads the **native single-file** `turbo.safetensors`
from the repo root instead, so point `--model` at a local directory laid out in
the standard MFLUX component structure:

```
krea-2-mlx/
  transformer/diffusion_pytorch_model.safetensors   # ← krea/Krea-2-Turbo: turbo.safetensors (repo root)
  vae/                                              # ← krea/Krea-2-Turbo: vae/
  text_encoder/model.safetensors                    # ← krea/Krea-2-Turbo: text_encoder/model.safetensors
  tokenizer/                                         # ← krea/Krea-2-Turbo: tokenizer/
```

(Symlinks into the Hugging Face cache work fine.) The standalone
`Qwen/Qwen3-VL-4B-Instruct` text encoder is also accepted — the loader strips
both the `language_model.` and `model.language_model.` prefixes.

## Example

```sh
mflux-generate-krea2 \
  --model /path/to/krea-2-mlx \
  --prompt "a photograph of a red fox sitting in a sunlit forest clearing, sharp focus, bokeh" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --steps 8 \
  -q 8
```

Turbo defaults: 8 steps, guidance 1.0 (CFG off), `er_sde` sampler. The plain
flow-matching Euler sampler — which matches the official diffusers
`FlowMatchEulerDiscreteScheduler` — is available via `--scheduler euler`.

Standard CLI options are supported: `--metadata`, `--stepwise-image-output-dir`,
and multiple `--seed` values. Image conditioning (edit / reference) is not yet
implemented.

## Quantization caching

Save a quantized model once and reload it without re-quantizing:

```sh
mflux-save --model /path/to/krea-2-mlx --quantize 8 --path /path/to/krea-2-mlx-q8
mflux-generate-krea2 --model /path/to/krea-2-mlx-q8 --prompt "..." --steps 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.krea2 import Krea2

model = Krea2(
    model_config=ModelConfig.krea2(),
    model_path="/path/to/krea-2-mlx",
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
