---
name: mflux-manual-testing
description: Manually validate mflux CLIs by exercising the changed paths and reviewing output images/artifacts.
---
# mflux manual testing

Some regressions (especially in CLIs and image IO) are easiest to catch by running the commands and visually inspecting outputs. This skill provides a lightweight, change-driven manual test checklist.

## When to Use

- You changed any CLI entrypoint(s) under `src/mflux/models/**/cli/`.
- You touched callbacks (e.g. stepwise output, memory saver) or metadata/image saving.
- Tests are green but you want confidence in real command usage.

## Strategy (change-driven)

- Identify what changed on your branch (new flags, default behavior changes, new callbacks, new models).
- Only run manual checks for the touched areas; don’t try to exercise every CLI.
- Prefer 1–2 seeds and a small step count (e.g. 4) for fast iteration, unless the change affects convergence/quality.
- Before manual CLI testing, **reinstall the local tool executables** so you’re testing the latest code:

```bash
uv tool install --force --editable --reinstall --prerelease=allow .
```

## Core CLI checks (pick what’s relevant)

- **Basic generation**: run the CLI once with a representative prompt and confirm the output is not “all noise”.
- **Model saving** (if relevant): if you touched weight loading/saving or model definitions, run `mflux-save` for the affected model(s) and verify:
  - the output directory is created
  - the command completes without missing-file errors
- **Run from disk** (if relevant): if you touched save/load paths or model resolution, generate from a locally saved model directory by passing `--model /full/path/to/saved-model` and confirm it runs and produces a sane image.
- **Stepwise outputs** (if relevant): run with `--stepwise-image-output-dir` and confirm:
  - step images are written for each step
  - the final step image matches the final output image qualitatively
  - the composite image is created
- **Low-RAM path** (if relevant): run with `--low-ram` and confirm:
  - generation completes
  - output quality is sane (no unexpected all-noise output)
- **Metadata** (if relevant): run with `--metadata` and confirm the `.json` is emitted and looks consistent.

## Output review (human-in-the-loop)

- Always point the human reviewer at:
  - the final output image path
  - any stepwise directory / composites
  - any metadata JSON files
- Ask the human to visually confirm “looks correct” rather than attempting pixel-perfect parity manually.

## Notes

- If the installed `uv tool` executable behaves differently from `uv run python -m ...`, prefer the local module run to isolate environment/tooling issues.
- If you need to reinstall the local tool executables, see the repo rules for the current recommended command.

