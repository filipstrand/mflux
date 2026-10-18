# Iris-3B
This directory contains MFLUX's MLX implementation of **Iris-3B**.

MFLUX supports [Iris-3B Turbo](https://huggingface.co/speridlabs/iris-3b) (distilled, 4 steps) and [Iris-3B](https://huggingface.co/speridlabs/iris-3b) (the undistilled base model, 100 steps with guidance) from Speridlabs. It is a 3B-parameter diffusion transformer that generates directly in **pixel space**: there is no VAE and no latent space, so the image is predicted as RGB values and nothing is lost to a latent decoder. The text encoder is the language tower of Qwen3-VL-4B-Instruct (an 8 GB download of its own). Both models come from one Hugging Face repository, Turbo from its `turbo/` folder.

All the standard modes of mflux apply: quantization (`-q 4`, `-q 8`), `mflux-save`, `--low-ram`, metadata export and `--config-from-metadata`.

![Iris-3B Example](../../assets/iris_3b.jpg)

*Turbo, 1024×1024, 4 steps, seed 42: the fox prompt below.*

## Example
The following uses the distilled model (4 steps, no guidance) for a text-to-image generation:

```sh
mflux-generate-iris \
  --prompt "a red fox sleeping in fresh snow, golden hour" \
  --seed 42
```

The default size is 1024×1024. Width and height must be multiples of 16.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.iris.variants import Iris

model = Iris(model_config=ModelConfig.iris_3b())
image = model.generate_image(
    seed=42,
    prompt="a red fox sleeping in fresh snow, golden hour",
    width=1024,
    height=1024,
)
image.save("fox.png")
```
</details>

## Base model example
The base model is undistilled. It runs 100 steps (a second-order DPM-Solver on the flow schedule) with classifier-free guidance (`--guidance` defaults to 3.0, `--negative-prompt` defaults to empty), so every step is two transformer passes. These are the defaults of the upstream implementation.

```sh
mflux-generate-iris \
  --model iris-3b-base \
  --prompt "a red fox sleeping in fresh snow, golden hour" \
  --seed 42
```

For Turbo `--steps`, `--guidance` and `--negative-prompt` have no effect and mflux prints a warning.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.iris.variants import Iris

model = Iris(model_config=ModelConfig.iris_3b_base())
image = model.generate_image(
    seed=42,
    prompt="a red fox sleeping in fresh snow, golden hour",
    num_inference_steps=100,
    guidance=3.0,
    negative_prompt="",
)
image.save("fox_base.png")
```
</details>

## Quantization
`-q 4` and `-q 8` quantize the model when it loads, the transformer and the text encoder. Small layers (the pixel-stage embedder, MLPs and head, the 12-to-1 layer pooling) and the feed-forward output projections, whose input width of 6826 is not a multiple of the quantization group size, stay in bf16. At `-q 4` and below the per-pixel adaLN projections of the pixel stage (2560 to 16384, they set individual pixel values) stay in bf16 too: quantized to 4 bits they print a faint screen-door grain over flat areas and fur. Measured on an M4 Pro (64 GB), Turbo, 1024×1024, seed 42. The transformer and the text encoder are both in memory while the weights load and the prompt is encoded; mflux then releases the text encoder, which is why the peak while generating is lower. The PSNR compares each image with the unquantized one: the composition is the same, the fine detail differs, and `-q 4` is visibly softer.

| | On disk (`mflux-save`) | Peak while loading + encoding | Peak while generating | Time (4 steps) | PSNR vs bf16 |
|---|---|---|---|---|---|
| bf16 | – | 14.5 GB | 7.6 GB | 17.2 s | – |
| `-q 8` | 7.5 GB (transformer 3.5 GB, text encoder 4.0 GB) | 8.8 GB | 5.3 GB | 18.4 s | 34.6 dB |
| `-q 4` | 4.7 GB (transformer 2.6 GB, text encoder 2.1 GB) | 5.9 GB | 4.4 GB | 18.2 s | 20.7 dB |

MLX reads weights lazily, so the first image also pays for loading them from disk. Quantization saves memory, not time. Save a quantized model once and load it from disk:

```sh
mflux-save --model iris-3b --quantize 8 --path ~/models/iris-3b-q8
mflux-generate-iris --model ~/models/iris-3b-q8 --prompt "..."
```

The saved directory holds the transformer, the text encoder and the tokenizer, so it loads without the Hugging Face repositories. An image from the saved model is identical to the image of the same `-q` flag loaded from Hugging Face.

> [!IMPORTANT]
> A saved model is identified by its directory name, as for FLUX.2 and FLUX 3: the bare name is Turbo (`iris-3b-q8`, `iris-q8`, `my-iris`) and a **base** model needs `iris-3b-base` or `iris-base` in the name, or `--base-model iris-3b-base`. For example, save the base model to `~/models/iris-3b-base-q8`.

## Speed and size
- Turbo, 1024×1024, bf16: about 4.4 s per step on an M4 Pro, 18 s per image after the model is loaded.
- Base, 1024×1024, bf16: 8.6 s per step (two passes), so about 14 minutes for the default 100 steps.
- `--low-ram` reduces the memory of the loop little for this model, since there is no VAE to tile. It still releases the text encoder and the transformer when they are no longer needed.

> [!WARNING]
> mflux downloads the transformer (5.6 GB for Turbo, 11 GB for the base model; the repository holds both) and the Qwen3-VL-4B-Instruct checkpoint (8.2 GB, of which only the language tower is used).

## Notes
- The prompt is wrapped in the chat template the model was trained with and limited to 300 tokens in total. The template takes 5 of them, so a caption is cut after 295 tokens.
- Sizes must be multiples of 16. Turbo was trained at about one megapixel: much smaller sizes show 16×16 patch artifacts, in the reference implementation too.
- `mflux-generate-iris` has no `--scheduler`, LoRA, `--compute-precision` or `--vae-tiling` options: the model has no VAE and the samplers are part of the checkpoints. The transformer runs the attention queries and keys in float32; with bf16 queries and keys, single patches of flat regions came out visibly off.
- Both entries share one Hugging Face repository. `--model iris-3b` (default, aliases `iris-3b-turbo`, `iris-turbo`, `iris`) selects Turbo, `--model iris-3b-base` (alias `iris-base`) the base model.
- Not ported from the reference implementation: depth estimation and the upscaler fine-tunes, the community Turbo LoRA, and training. `mflux-train` is not available for this model.
- Slow golden tests are not part of this port; `tests/` holds fast tests for the weight mapping, quantization, prompt encoding, samplers, CLI and saving.

## License and attribution
Iris-3B is released by [Speridlabs](https://github.com/speridlabs/iris-3b) under the Apache License 2.0, and Qwen3-VL-4B-Instruct (the text encoder) by Alibaba Cloud under the Apache License 2.0.
