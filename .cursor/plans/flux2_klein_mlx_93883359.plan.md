---
name: Flux2 Klein MLX
overview: Port the full Flux2 Klein stack from diffusers into mflux with a new `flux2` model folder and a `Flux2Klein` variant, supporting separate configs for the 4B and 9B weights.
todos:
  - id: inspect-diffusers
    content: Review Flux2 Klein pipeline + model files in diffusers
    status: completed
  - id: mlx-flux2-models
    content: Implement Flux2 transformer/VAE/pos-embeds in MLX
    status: cancelled
    dependencies:
      - inspect-diffusers
  - id: qwen3-encoder
    content: Implement Qwen3 tokenizer + encoder prompt embeds
    status: completed
    dependencies:
      - inspect-diffusers
  - id: weights-config
    content: Add Flux2 Klein weight mappings + ModelConfig entry
    status: completed
    dependencies:
      - mlx-flux2-models
      - qwen3-encoder
  - id: pipeline-cli
    content: Add Flux2 Klein pipeline + CLI/defaults/completions
    status: cancelled
    dependencies:
      - weights-config
  - id: smoke-test
    content: Add a lightweight end-to-end smoke test
    status: cancelled
    dependencies:
      - pipeline-cli
---

# Flux2 Klein MLX Port

## Scope & References

- Review local diffusers implementation for Flux2 Klein to mirror behavior and shapes: [`/Users/filipstrand/Desktop/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`](/Users/filipstrand/Desktop/diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py), [`/Users/filipstrand/Desktop/diffusers/src/diffusers/models/transformers/transformer_flux2.py`](/Users/filipstrand/Desktop/diffusers/src/diffusers/models/transformers/transformer_flux2.py), [`/Users/filipstrand/Desktop/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`](/Users/filipstrand/Desktop/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py), and scheduler logic in [`/Users/filipstrand/Desktop/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`](/Users/filipstrand/Desktop/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py).

## Implementation Plan

- **Skeleton first (Flux2Klein class)**: create the `src/mflux/models/flux2` package and a bare `Flux2Klein` variant class under `src/mflux/models/flux2/variants/`, wired for initialization but not full functionality yet.
- **Init setup**: make sure the `Flux2Klein` initializer mirrors mflux style (`*_initializer.py`), sets config, loads weights, and attaches modules in the correct order.
- **Weight handling (no quantization yet)**: implement `Flux2KleinWeightDefinition` and mapping files, then validate against the reference 4B weights in the local HF cache to ensure names and shapes align; keep quantization disabled for now.
- **Minimal hardcoded runner**: add a simple Python script (like a CLI entry point but with hardcoded values) to load the model and run a tiny inference, mirroring your `diffusers/flux_kelin.py` flow.
- **Port backwards from VAE**:
- **VAE decode**: export latents from diffusers just before VAE decode, load them inline in the MLX codepath, run VAE decode, and send you the image for inspection.
- **VAE encode**: run an encode→decode roundtrip in MLX and compare the images visually; notify you when it is ready to inspect.
- **Transformer**: once VAE matches, port transformer blocks and validate with intermediate latent checks.
- **Text encoder**: port Qwen3 text encoder last, fully mirroring diffusers behavior.
- **Keep debugging inline**: prefer inline `mx.load`/`mx.save` or equivalent within the code paths instead of many separate scripts to avoid drift.
- **Optional**: add `.cursor` skills documenting the porting flow for mflux if helpful.

## Implementation Todos

- `inspect-diffusers` — Review Flux2 Klein pipeline, transformer, VAE, and scheduler usage for exact tensor shapes and parameters.
- `flux2-skeleton` — Add `flux2` package and `Flux2Klein` skeleton + init wiring.
- `weights-config` — Add Flux2 Klein weight mappings + ModelConfig entries (4B/9B), no quantization.
- `hardcoded-runner` — Add a minimal hardcoded runner script mirroring `diffusers/flux_kelin.py`.
- `vae-decode` — Inline MLX VAE decode using diffusers-exported latents; validate visually.
- `vae-encode` — Encode→decode roundtrip check; validate visually.
- `transformer-port` — Port transformer blocks and validate with intermediate checks.
- `qwen3-encoder` — Port Qwen3 tokenizer + text encoder stack from diffusers for Flux2 Klein (last).
- `optional-cursor-skill` — Add `.cursor` skill notes for mflux porting, if useful.