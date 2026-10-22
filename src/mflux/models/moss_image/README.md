# MOSS-Image

This directory contains the MFLUX MLX port of **MOSS-Image-SFT**
([`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)).

MOSS-Image-SFT is a text-to-image model from OpenMOSS (Apache 2.0). It is a 4.9B
single-stream diffusion transformer with a Qwen3.5-9B text encoder and the FLUX.2 VAE with
the small decoder. It accepts natural-language prompts, tag sequences and JSON layouts, and it
renders English and Chinese text.

The port runs all three parts in MLX. The text encoder is an MLX port of the hybrid Qwen3.5
decoder (Gated DeltaNet and gated attention layers), so the model does not need PyTorch at
run time.

## Example

```sh
mflux-generate-moss-image \
  --prompt 'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in English: "Buy milk and eggs!"' \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  -q 8
```

The defaults follow the model card: 50 steps, guidance 3 and an empty negative prompt. The
negative branch of CFG skips transformer layer 22. Each step runs the transformer two times.
`--uncond-skip-layers` sets other layers, and the flag with no values skips no layer. The image
metadata records the skipped layers, so `--config-from-metadata` replays them.

- Width and height must be multiples of 16.
- `--steps` accepts 1 to 218. Above 218, two steps snap to the same point of the training
  schedule, so the command stops with an error.
- `--scheduler` is ignored. The model samples on the schedule it was trained with.
- `--step-cache-ratio 0.25` skips the transformer on about a quarter of the steps. The
  timestep embedding of the model selects the steps.
- LoRA flags are not available for this model.

Quantization (`-q 3|4|5|6|8`) applies to the transformer and the text encoder. To keep a
quantized copy, save it once and load the copy with `--model`:

```sh
mflux-save --model moss-image -q 4 --path ~/models/moss-image-q4
mflux-generate-moss-image --model ~/models/moss-image-q4 --prompt "a lighthouse at dusk"
```

On an M-series Mac with 36 GB, a 1024x1024 image with the 4-bit copy uses about 10.5 GB of
MLX memory and takes about 9.7 seconds for each step. The 8-bit model at 512x512 takes about
2.2 seconds for each step.

<details>
<summary>Python API</summary>

```python
from mflux.models.moss_image import MOSSImage

model = MOSSImage(quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A majestic fluffy cat sitting on a wooden bench, golden hour lighting, photo",
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance=3.0,
)
image.save(path="moss_cat.png")
```

`generate_image` also takes `negative_prompt`, `uncond_skip_layers` (default `(22,)`, an empty
tuple turns the layer skip off) and `step_cache_ratio`.

You can also call the steps of `mflux-generate-moss-image` from Python. `MOSSImageCommand` in
`mflux.models.moss_image.cli.moss_image_generate` has `validate(args)`, `load(args)` and
`generate(model, args, seed, prompt)`.

</details>

## Weights

The model loads the official repository `OpenMOSS-Team/MOSS-Image-SFT` (about 27 GB of bf16
weights) or a copy that `mflux-save` wrote. The loader stops with an error when a checkpoint
does not have all the weights of the model. The MLX conversions in the `vanch007/MOSS-Image-SFT-mlx-*`
repositories use other weight names and do not include the text encoder, so this port does not load them.

## Validation

We compared the port with the reference pipeline (`pipeline_moss_image.py` in the model
repository) on the same weights:

- The transformer matches the reference to 6e-7 in float32 on random weights.
- The Qwen3.5 text model matches `transformers` to 5e-5 in float32 on random weights.
- With the real weights in bf16, the token ids are equal and the text embeddings have a
  cosine similarity of 0.999 or more.
- With the same initial noise and text embeddings (512x512, 20 steps, CFG 3), the MLX image
  has a PSNR of 34.9 dB against the PyTorch image in bf16 and 36.6 dB in 8-bit.
