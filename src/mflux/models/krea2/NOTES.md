# Krea-2 port — status & open items

**STATUS: first light ✅** — text-to-image works end-to-end. Turbo, 8 steps,
CFG 1.0, q8, 512×512 in ~15s produces coherent, prompt-faithful images.

`Krea-2` is a **single-stream MMDiT** built on the Qwen-Image
stack. Reference implementation: ComfyUI `comfy/ldm/krea2/model.py`,
`comfy/text_encoders/krea2.py`, `supported_models.Krea2`, `model_base.Krea2`.

## Gotcha that cost the first-light hunt

`ComponentDefinition.weight_prefix_filters` only **filters** keys — it does NOT
strip the prefix. With `mapping_getter=None` the keys stay fully-qualified
(`model.language_model.*`) and `tree_unflatten` builds a tree that misses the
flat `Krea2TextEncoder` module, so the TE silently runs on **random init**
(garbage images, but no error). Fix: strip the prefix via `key_transform`
(`_strip_te_prefix` in `krea2_weight_definition.py`). Validated by running the TE
as an LM: "The capital of France is" → " Paris".

## Architecture (confirmed against the safetensors)

- DiT: `features=6144`, `heads=48`, `kvheads=12` (GQA, repeat 4), `head_dim=128`,
  `layers=28`, `multiplier=4` (SwiGLU → 16384), `patch=2`, `channels=16`,
  `tdim=256`, `txtdim=2560`, `txtlayers=12`, `txtheads=20`, `theta=1000`.
- RoPE: 3-axis, `axes_dim=[32,48,48]`, Flux-style adjacent-pair rotation
  (`[cos,-sin,sin,cos]`). Text pos `(0,0,0)`, image pos `(0,h,w)`.
- RMSNorm uses the `(1 + scale)` convention, fp32 reduction.
- Attention: per-head QK-RMSNorm + **sigmoid gate**: `wo(attn * sigmoid(gate(x)))`.
- Modulation: AdaLN-single, 6-way; shared `tvec = tproj(tmlp(t))` + per-block
  `mod.lin` bias. Final layer uses `SimpleModulation` on `t` (not `tvec`).
- Text fusion: 12 Qwen3-VL hidden states → 2 `layerwise_blocks` (attend across
  layers) → `projector` Linear(12→1) → 2 `refiner_blocks` (attend across seq).

## Companion components (reused, both already in mflux)

- **VAE**: Qwen-Image VAE (`models/qwen` → `QwenVAE`), latent format = Wan2.1
  (16-ch). `vae_key_prefix = ["vae."]`.
- **Text encoder**: Qwen3-VL-4B (`models/common_models/qwen3_vl`), tap
  `hidden_states[2,5,8,…,35]` (12 layers). KREA2 template (system "Describe the
  image by detailing…" + user), strip system+user prefix, `thinking=True` to
  drop the empty `<think>` block. TE output `(B,12,seq,2560)` → flatten to
  `(B,seq,12·2560)`, unpacked inside the DiT.

## Sampling

- Rectified flow (`ModelType.FLUX`), `sampling_settings = {multiplier:1.0, shift:1.15}`.
- Turbo: 8 steps, CFG 1.0, `er_sde` sampler, `simple` scheduler, 1280×720.
- Raw: full step count + CFG>1.
- Implemented (`model/krea2_sampler.py`): `ErSdeStepper` (default; flux-branch port
  of ComfyUI `sample_er_sde` — `ModelSamplingFlux` is NOT `CONST`, so `alpha=1`,
  `er_lambda=sigma`, half-logSNR `-log(sigma)`; 3-stage multistep + SDE noise) and
  `EulerStepper` (`--scheduler euler` fallback). Schedule is the flux static-shift
  `flow_sigmas` (≈ ComfyUI `simple` over `ModelSamplingFlux.sigmas`; not bit-exact).
  At 8 steps the two steppers are visually near-identical — er_sde is for reference
  parity, not a quality jump.

## Done

- Full DiT forward (`model/krea2_transformer/`), faithful to the reference.
- Qwen3-VL-4B text encoder (`model/krea2_text_encoder/`) — mirrors the proven
  flux2 `Qwen3TextEncoder` (common `Qwen3VLDecoderLayer` + plain text rotary,
  `rope_theta=5e6`); 12-layer tap `hidden_states[2,5,…,35]` (HF indexing, index
  0 = embeddings), **layer-major** flatten to `(B,seq,12·2560)` (matches the DiT
  `_unpack_context`). Validated as an LM ("…France is" → " Paris").
- `txt2img` pipeline (`variants/txt2img/krea2.py`) + flow sampler
  (`model/krea2_sampler.py`) + `ModelConfig.krea2()` registry entry.
- Weight mapping — transformer ~1:1; VAE reuses Qwen-Image mapping; TE direct-load
  via `key_transform=_strip_te_prefix`.
- **End-to-end generation validated** (fox @ 512², turbo q8, ~15s). VAE round-trip
  verified bit-clean; TE weights verified `match=True` against the shards.
- **CLI**: `mflux-generate-krea2` (`cli/krea2_generate.py`, registered in
  `pyproject.toml`). `generate_image` returns a `GeneratedImage` → PNG + metadata
  sidecar. Pass `--model <staged dir>`. Full standard UX: live tqdm denoising bar
  (loop iterates `config.time_steps` + callback context), `--metadata`,
  `--stepwise-image-output-dir` (per-step previews + composite via
  `Krea2LatentCreator.unpack_latents` = identity), memory/battery callbacks.

## Assembled model dir

`~/mlx-forge/models/krea-2-mlx/` — staged from the **official** `krea/Krea-2-Turbo`
diffusers repo (symlinks into HF cache):

- `transformer/` ← official root `turbo.safetensors` (native/ComfyUI single-file,
  430 tensors, no vestigial `last.up`/`last.down`). NOTE: the repo's `transformer/`
  subdir is *diffusers*-format (different keys) — use the root file, which matches
  our mapping.
- `vae/` ← official `vae/` (`AutoencoderKLQwenImage`; key-identical to `Qwen/Qwen-Image`).
- `text_encoder/` ← official `text_encoder/model.safetensors` (single file,
  `language_model.*` + `visual.*`; loaded `mlx_native`, prefix stripped to the LM).
- `tokenizer/` ← official `tokenizer/`.

`model_index.json` confirms the port: `text_encoder_select_layers [2,5,…,35]`,
`patch_size 2`, VAE `AutoencoderKLQwenImage`, TE `Qwen3VLModel`. The diffusers
scheduler is `FlowMatchEulerDiscreteScheduler` — so our `EulerStepper` is the
official-reference sampler; `er_sde` is ComfyUI's default (both shipped).

Point mflux at this path via `--model`. The standalone `Qwen/Qwen3-VL-4B-Instruct`
TE (sharded, `model.language_model.*`) also loads — `_strip_te_prefix` accepts both
prefixes — but the official bundled TE is authoritative.

## Open items (next) — quality/features, not blockers

1. **Vision / edit path**: add the Qwen3-VL vision tower + mRoPE + image-token
   splice (mirror `qwen` `init_edit` + `common_models/qwen3_vl` vision model) for
   image references and edits. Vision weights are already in the staged TE file
   (`visual.*`, currently dropped at load). Pinned — official README doesn't
   document edits.
2. **Validation**: bit-exact DiT forward vs ComfyUI on a fixed latent + context
   (incl. the er_sde sampler — current port is faithful but not reference-verified).

## Done (continued)

- **Prompt prefix-strip**: `get_prompt_embeds` drops the system + `<|im_start|>user\n`
  template positions from the conditioning (token-id `template_end` logic, mirrors
  ComfyUI) so only real prompt tokens condition the DiT.
- **Official `krea/Krea-2-Turbo` weights** staged + validated end-to-end (transformer
  + single-file TE + VAE). Confirms `last.up`/`last.down` were vestigial (absent here).
- **Quantization caching**: `Krea2.save_model` + `mflux-save` dispatch. Saves the
  quantized model (q8 ≈ 21 GB: q8 transformer + bf16 TE + VAE) for fast reload with
  no re-quantize. Reloaded q8 is visually identical to on-the-fly q8 (sub-perceptual
  numerical diff). `mflux-save --model <staged> --quantize 8 --path <out>`.
