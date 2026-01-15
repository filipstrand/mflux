# mflux Flux2 porting

## Goal
Provide a repeatable workflow for porting Flux2 components into mflux (MLX), with inline debug hooks.

## Steps
1. Add the `flux2` package skeleton and initializer under `src/mflux/models/flux2`.
2. Implement weight definitions and mapping stubs first so weight loading is wired early.
3. Add a hardcoded runner in `src/mflux/models/flux2/cli/flux2_klein_run.py` for quick iteration.
4. VAE decode:
   - Save packed latents from diffusers just before VAE decode.
   - Use `Flux2Klein.debug_decode_packed_latents()` to load and decode inline.
5. VAE encode:
   - Use `Flux2Klein.debug_roundtrip_image()` to run encode→decode and inspect.
6. Transformer:
   - Port blocks and update `Flux2WeightMapping.get_transformer_mapping()`.
   - Add small intermediate-shape checks inline (avoid extra scripts).
7. Text encoder:
   - Port Qwen3 encoder and tokenizer; ensure hidden layer stacking matches diffusers.

## Notes
- Keep debug `mx.load`/`mx.save` inline in code paths to avoid script drift.
- Avoid quantization until correctness is validated.
