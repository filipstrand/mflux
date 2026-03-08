# mflux – Cursor Agent Rules (Project Rules)

These rules exist to make agent work in this repo **predictable, verifiable, and low-drama**.

## Commands / environment

- **Always use uv** for dependency management and running code.
  - Run scripts/binaries with `uv run <command>`.
  - Prefer `uv run python -m ...` for local modules.
  - Manage deps with `uv add <pkg>` / `uv remove <pkg>`.
- **Tool installs (CLI executables)**:
  - When you need to (re)install the local checkout as a `uv tool` (e.g. after changing CLI code), prefer an **editable install**:
    - `uv tool install --force --editable --reinstall .`
- **Prefer Makefile targets** when they exist (they encode project-specific setup):
  - `make install`, `make lint`, `make format`, `make test-fast`, `make test`, `make build`.

## Tests (goldens / image output)

- **Always preserve test outputs** (for visual inspection): run tests with `MFLUX_PRESERVE_TEST_OUTPUT=1` (the Makefile test targets already do this).
- **Do not update reference (“golden”) images** unless explicitly asked.
- Prefer faster scopes first (`make test-fast` → `make test-slow` → `make test`).
- For the full playbook (how to handle failures and golden diffs), use the `mflux-testing` skill.

## Lint / format

- Use the Makefile targets for repo workflows: `make lint`, `make format`, `make check`.

## Code style

- Avoid docstrings; prefer clear naming and focused helpers.
- Add comments only when logic is non-obvious; keep them short.
- Consider deeper modules and clear APIs over over-fragmented tiny functions; weigh the tradeoff.
- Prefer composition over inheritance when practical; avoid deep class hierarchies.
- Static methods are fine when they clarify stateless helpers.
- Avoid free-standing helper functions in Python modules; prefer placing helpers on proper classes (even if static). Thin module-level entrypoints like CLI `main()` functions are fine.
- Keep private methods (leading underscore) at the bottom of classes; public APIs at the top.
- Use type hints consistently for public APIs.
- Keep CLI entrypoints thin: parse args, resolve config, construct the model, register callbacks, run, save.
- Prefer extending existing shared abstractions over adding model-specific one-off paths unless the model truly needs a new abstraction.
- Reuse shared plumbing before adding bespoke code paths: `CommandLineParser`, `ModelConfig`, `CallbackManager`, `DimensionResolver`, shared schedulers, and existing save/metadata helpers.
- Keep public model APIs small and obvious; prefer a clear top-level method like `generate_image(...)` with implementation details pushed into private helpers.
- Keep one clear source of truth for model defaults and reflect those defaults consistently in CLI behavior, Python API examples, and README docs.
- Treat README examples as part of the API contract. If behavior, defaults, or flags change, update the relevant examples in the same pass.

## Releases

- When preparing a release, prefer the `mflux-release` skill.
- Tagging/publishing is handled by an external GitHub Action.

## Git safety

- **Never push or force-push** to the remote repository without explicit user approval for each push.
- Committing locally is fine and encouraged for progress tracking.
- Before any major squash/rewrite of local branch history, create a **local backup branch** at the current `HEAD` first (for example `backup/<branch>-pre-squash-YYYYMMDD-HHMM` so repeated backups stay unique to the minute).
- That backup branch is intentionally **local-only** for safety and rollback; do not push it unless the user explicitly asks.

## Agent workflow norms (modern Cursor best practices)

- For multi-file or high-risk work, **start with a short plan** (bullets: goals, constraints, files to touch, how you’ll verify).
- **Plan Mode Enforcement**: For any non-trivial task or high architectural risk, save your plan to `.cursor/plans/YYYY-MM-DD-feature-name.md` and ask for approval before coding.
- Keep changes tight, and prefer **verifiable goals** (tests/lint/build) over speculation.
- If the task scope changes materially, stop and re-align rather than continuing in a confused state.
- When users ask for CLI usage (e.g., “Can you help me generate an image using z-image?”), use the `mflux-cli` skill.
- For new models or major feature additions, the definition of done usually includes shared config wiring, a thin CLI entrypoint, verification coverage, and a README/example update that matches existing model docs.

## Bug/behavior reporting format (chat)

- When reporting bugs or behaviors, always use a simple story format.
- Use separate "Scenario" sections for separate issues.
- Each scenario must be step-by-step and end with a one-line "Fix:".

Example:
Scenario — Late preview crash
1) User starts edit training without data/preview.*.
2) Run crashes on first preview step.
Fix: Validate preview image at config load.

## Skills

- For image generation requests, **always use** `mflux-cli` to find the right command and flags.
- Use `mflux-cli` for CLI capability discovery and usage help.
- Use `mflux-dev-env` for setup, uv usage, and Makefile targets.
- Use `mflux-testing` for running tests and handling golden images.
- Use `mflux-manual-testing` for validating CLI outputs manually.
- Use `mflux-debugging` for MLX vs PyTorch/diffusers comparisons.
- Use `mflux-model-porting` when porting models into MLX.
- Use `mflux-release` for release preparation steps.
- Use `mflux-pr` for preparing clean PRs.

