# mflux model porting (general)

## Goal
Provide a repeatable, MLX-focused workflow for porting ML models (typically from diffusers repo located near mflux repo in the system) into mflux with correctness first, then refactor to mflux style.

## Principles
- Match the reference implementation first; prove correctness before cleanup.
- Lock correctness with deterministic tests before refactoring.
- Refactor toward shared components and clean APIs once tests are green.

## Workflow (checklist)
1. **Scope and parity**
   - Define target parity (outputs, speed, memory) and acceptable tolerances.
   - Identify reference files, configs, and checkpoints to mirror.
2. **Port fast to reference**
   - Add the model package skeleton and a variant class + initializer.
   - Wire weight definitions/mappings early so loading is exercised (implement quantization in the initializer, but skip it during early runs).
   - Add a minimal hardcoded runner for quick iteration, seeded with diffusers-style defaults (e.g., 1024×1024, default prompt).
   - Add lightweight shape checks close to the code paths.
   - Use `mx.save`/`mx.load` at critical points; it is OK to add these to the reference (without changing logic) to export latents.
3. **Port order (work backwards from image)**
   - Typical image generation flow: `prompt → text_encoder → transformer_loop → VAE → image`.
   - For porting, invert the order so you can validate pixel space early.
   - Start with VAE decode/encode to validate output images quickly:
     - Export packed latents from the reference just before VAE decode.
     - Load latents inline and decode to an image for visual inspection.
     - Run an encode→decode roundtrip to sanity check reconstruction; a good-looking image reconstruction increases confidence in the implementation.
   - Then port the transformer loop and its schedulers with intermediate latent checks.
     - If the reference uses a novel scheduler, port it; otherwise, reuse the existing mflux scheduler.
   - Finish with the text encoder and tokenizer details.
4. **Deterministic validation**
   - Create a deterministic MLX test (image or tensor) that locks the output.
   - Run tests via `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run <test command>`.
5. **Post-test refactor (explicit step)**
   - Review commits after the first deterministic test to capture refactoring preferences.
   - Consolidate shared components into common modules.
   - Remove debug paths and one-off schedulers once validated.
   - Move configuration defaults into standard config/scheduler paths.
    - Simplify and decompose large files into focused modules once behavior is locked.
    - Prefer shared scheduler implementations when they already exist in mflux.
    - Keep running the deterministic image test during refactors to avoid regressions.
6. **Finalize**
   - Re-run tests and basic perf checks.
   - Add CLI/pipeline defaults and completions later, once core output is stable.
   - Document any new mapping rules, shape constraints, or tolerances.

## Tooling expectations
- Use `uv` for running scripts and tests: `uv run <command>`.
- Prefer `uv run python -m <module>` for local modules.

## Deliverables
- Deterministic MLX test that verifies correctness.
- Documented weight mapping, shape constraints, and any known tolerances.
- Cleaned, shared components aligned with mflux style.
