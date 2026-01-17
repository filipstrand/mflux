# FLUX.2 (MLX)
This directory contains MFLUX’s MLX implementation of the **FLUX.2** family.

## Supported variants
- `flux2-klein-4b` (default)
- `flux2-klein-9b`

## Text-to-image

```sh
mflux-generate-flux2 \
  --model flux2-klein-4b \
  --prompt "Luxury food photograph" \
  --steps 4 \
  --seed 2
```

## Image-conditioned editing
Requires one or more `--image-paths`.

```sh
mflux-generate-flux2-edit \
  --model flux2-klein-4b \
  --image-paths input.png \
  --prompt "Turn this into a luxury food photograph" \
  --steps 4 \
  --seed 2
```

## Notes
- FLUX.2 does not support `--negative-prompt` or CFG-style guidance. Use `--guidance 1.0`.

# FLUX.2 (MLX)
This directory contains MFLUX’s MLX implementation of the **FLUX.2** family.

## Supported variants
- `flux2-klein-4b` (default)
- `flux2-klein-9b`

## Text-to-image

```sh
mflux-generate-flux2 \
  --model flux2-klein-4b \
  --prompt "Luxury food photograph" \
  --steps 4 \
  --seed 2
```

## Image-conditioned editing
Requires one or more `--image-paths`.

```sh
mflux-generate-flux2-edit \
  --model flux2-klein-4b \
  --image-paths input.png \
  --prompt "Turn this into a luxury food photograph" \
  --steps 4 \
  --seed 2
```

## Notes
- FLUX.2 does not support `--negative-prompt` or CFG-style guidance. Use `--guidance 1.0`.

