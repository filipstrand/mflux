# Ideogram 4

Ideogram 4 support targets the FP8 checkpoint layout from `ideogram-ai/ideogram-4-fp8`.

## Generate

Ideogram 4 accepts any prompt string, but the checkpoint was trained for structured JSON captions. For reliable
images, pass a JSON caption directly. The mflux port does not call Ideogram's hosted magic-prompt rewrite service
or require an Ideogram API key.

```sh
mflux-generate-ideogram4 \
  --prompt-file teapot-caption.json \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --preset V4_DEFAULT_20 \
  --use-preset-steps
```

Example caption file:

```json
{
  "high_level_description": "A white ceramic teapot on a simple studio table.",
  "style_description": {
    "aesthetics": "clean, calm, minimal",
    "lighting": "soft diffuse studio lighting",
    "photo": "eye-level, 50mm lens, shallow depth of field",
    "medium": "photograph",
    "color_palette": ["#FFFFFF", "#E5E0D8", "#2E2E2E"]
  },
  "compositional_deconstruction": {
    "background": "A neutral studio tabletop with a pale wall behind it.",
    "elements": [
      {
        "type": "obj",
        "bbox": [250, 320, 780, 690],
        "desc": "A glossy white ceramic teapot with a curved handle and short spout."
      }
    ]
  }
}
```

## Python API

```python
from mflux.models.ideogram4 import Ideogram4

model = Ideogram4()
image = model.generate_image(
    prompt={
        "high_level_description": "A white ceramic teapot on a simple studio table.",
        "style_description": {
            "aesthetics": "clean, calm, minimal",
            "lighting": "soft diffuse studio lighting",
            "photo": "eye-level, 50mm lens, shallow depth of field",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "A neutral studio tabletop with a pale wall behind it.",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [250, 320, 780, 690],
                    "desc": "A glossy white ceramic teapot with a curved handle and short spout.",
                }
            ],
        },
    },
    seed=42,
    width=1024,
    height=1024,
    preset="V4_DEFAULT_20",
    use_preset_steps=True,
)
image.save("ideogram4.png")
```

## Notes

- Width and height must be in `[256, 2048]` and multiples of 16.
- JSON captions are compacted with `ensure_ascii=False` before tokenization.
- Caption validation warns for plain prompts, malformed JSON, missing schema fields, key-order issues, invalid
  bounding boxes, and lowercase or shorthand hex colors. Use `--strict-caption-validation` to fail on warnings.
- Plain text prompts still run, but they can underperform and trigger the model's safety placeholder more often.
- `--use-preset-steps` uses the preset step count and guidance schedule.
- Without `--use-preset-steps`, `--steps` controls the step count and `--guidance` is constant across steps.
- Strict parity validation should compare against exported reference latents from the source implementation because PyTorch/MLX RNGs do not produce identical noise from the same seed.

## Parity

Run checkpoint-layout validation against the default Hugging Face cache. Set `MFLUX_IDEOGRAM4_MODEL_PATH` only when using a custom local checkpoint path:

```sh
uv run --extra dev python -m pytest tests/image_generation/test_ideogram4_checkpoint_layout.py -m fast
```

Export deterministic tensors from the local `mlx-vlm` reference, then compare mflux against them:

```sh
uv run --extra dev python tests/image_generation/ideogram4_parity/export_reference.py \
  --reference-repo /path/to/mlx-vlm \
  --model-path /path/to/ideogram-4-fp8 \
  --output /tmp/ideogram4-reference.safetensors

uv run --extra dev python tests/image_generation/ideogram4_parity/compare_mflux.py \
  --artifact /tmp/ideogram4-reference.safetensors
```
