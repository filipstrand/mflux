---
name: mflux-debugging
description: Debug MLX ports by comparing against a PyTorch/diffusers reference via exported tensors/images (export-then-compare).
---
# mflux debugging (MLX parity vs PyTorch/diffusers)

Use this skill when you are porting a model to MLX and need to **prove numerical parity** (or isolate where it diverges) versus a **PyTorch reference implementation** (often from diffusers).

This skill defaults to **export-then-compare**:
- Run the reference once and **export deterministic artifacts** (tensors + optional images).
- Load those artifacts in MLX and compare with clear thresholds.

## When to Use

- You suspect a port mismatch (wrong shapes/layout, RoPE, scheduler math, dtype casting, etc).
- You want a repeatable workflow to narrow down the first layer/block where outputs diverge.
- You need evidence of correctness before refactoring (see `mflux-model-porting`).

## Ground Rules (repo norms)

- Use `uv` to run Python: `uv run python -m ...`
- If you run pytest, preserve outputs: `MFLUX_PRESERVE_TEST_OUTPUT=1` (see `mflux-testing` and `.cursor/commands/test*.md`).
- Do **not** update or replace reference (“golden”) images unless explicitly asked.
- Debug artifacts (tensor dumps) should live in a local folder and **must not be committed** unless explicitly asked.
- If you need the broader porting workflow (milestones, ordering, when to refactor), follow `mflux-model-porting`.
- **RNG warning**: PyTorch and MLX RNGs are different. Matching the same integer `seed` is not enough for parity—export the *exact* initial noise/latents from the reference and load them in MLX.
- Practical setup: the PyTorch reference repo (often `diffusers/`) and `mflux/` are frequently **next to each other** on disk (e.g. both on your Desktop). Use absolute paths when in doubt.

## Default Workflow (export-then-compare)

### Preferred workflow: two tiny scripts + inline dumps

For day-to-day debugging, prefer a **minimal paired repro**:
- One simple script in the reference repo (often `diffusers/`), e.g. `diffusers/flux2_klein_edit_debug.py`
- One simple script in `mflux/`, e.g. `mflux/flux2_klein_edit_debug.py`

Keep them “boring”: hardcoded variables, no cli, no framework, and just a few `np.savez(...)` / `mx.save(...)` lines at the right spot.

The key trick for RNG parity:
- In the reference script, **compute latents once**, save them, then **pass them back into the pipeline** (`latents=...`) so the run definitely uses the dumped tensor.
- In rare cases where a tensor needs to be saved from within a loop, make sure its name reflects the loop conditions (e.g the 4th noise predcition in a 10 step loop etc).
- In the MLX script, **load that same latents file** and feed it into the MLX run (do not rely on matching integer seeds).

### 1) Pick a single deterministic repro

- Fix **seed(s)**, **prompt(s)**, **height/width**, **steps**, **guidance**, and any **input image** paths.
- Keep the first repro small if possible (fewer steps, smaller resolution) to iterate quickly.

### 2) Decide your checkpoints (what to dump)

Start coarse, then narrow:
- **VAE**: packed latents before decode; optional intermediate activations for one block if needed.
- **Transformer**: hidden states at entry/exit of the model, then per-block (or every N blocks), then inside attention/MLP.
- **Text encoder**: token ids + attention mask, embeddings output, then per-layer hidden states if needed.
- **Scheduler**: timesteps/sigmas/alphas and the predicted noise/velocity per step.

Tip: work “backwards from pixels” like `mflux-model-porting` suggests: validate VAE decode first with exported latents, then the diffusion/transformer loop, then text encoder.

### 3) Export artifacts from the PyTorch reference (no logic changes)

Create a run directory like:
- `./debug_artifacts/<run_id>/ref/`

Export with one of these patterns:
- **NumPy**: `np.savez(path, **tensors_as_numpy)`
- **PyTorch**: `torch.save(dict_of_tensors, path)`

### 4) Run the MLX side with the same inputs and compare

Create a matching run directory:
- `./debug_artifacts/<run_id>/mlx/`

Load and compare tensors. For each checkpoint, report:
- **Shape + dtype**
- **max_abs_diff**, **mean_abs_diff**
- **max_rel_diff** (guarding division by zero)
- Pass/fail with a clearly stated **rtol/atol**
- It is also very instructive to look at actual tensor value, say the first 10 elements.

Suggested tolerance starting points (adjust per component):
- **fp32 comparisons**: `atol=1e-5`, `rtol=1e-5`
- **fp16/bf16 comparisons**: `atol=1e-2`, `rtol=1e-2`
- If comparing images: compare both (a) tensor space before final clamp and (b) saved `png` visually, since tiny numeric diffs can look identical.

If a checkpoint fails:
- Add an **earlier** checkpoint and repeat (binary search through the forward path).

## Common Causes of Divergence (high-signal checklist)

- **Layout mistakes**: NCHW vs NHWC, transposes around convs/attention, flatten/reshape ordering. Some operations like convolutions can have different conventions between libraries.
- **Broadcasting**: scale/shift vectors applied on the wrong axis (common in RoPE and modulation).
- **Dtype casting**: reference silently upcasts to fp32 for norm/softmax; MLX path stays in fp16.
- **RoPE details**: position ids, reshape order, whether cos/sin are broadcast over heads vs sequence.
- **Scheduler math**: timestep indexing, sigma/alpha definitions, and off-by-one step order.
- **Seed/RNG**: ensure you aren’t comparing stochastic paths (dropout, noise sampling) without controlling RNG.

## Artifact Hygiene

- Prefer `debug_artifacts/<run_id>/...` at repo root.
- Do not commit `debug_artifacts/` unless explicitly asked.
- If you convert the parity check into a test, follow the repo’s testing conventions and preserve outputs (see `mflux-testing`).
- Clean up old artifacts when they are no longer needed, only focus on the current problem and avoid confusion with older artifacts that are not relevant for the current task.

## See Also

- `mflux-model-porting`: correctness-first workflow (validate components and lock behavior before refactor).
- `mflux-testing`: how to run tests safely and handle image outputs/goldens.

