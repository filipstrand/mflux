# mflux – Cursor Agent Rules (Project Rules)

These rules exist to make agent work in this repo **predictable, verifiable, and low-drama**.

## Commands / environment

- **Always use `uv`** for dependency management and running code.
  - Run scripts/binaries with `uv run <command>`.
  - Prefer `uv run python -m ...` for local modules.
  - Manage deps with `uv add <pkg>` / `uv remove <pkg>`.
- **Prefer Makefile targets** when they exist (they encode project-specific setup):
  - `make install`, `make lint`, `make format`, `make test-fast`, `make test`, `make build`.

## Tests (goldens / image output)

- **Always preserve test outputs** (for visual inspection):
  - Run tests with `MFLUX_PRESERVE_TEST_OUTPUT=1`.
- Prefer faster scopes first:
  - **Single test**: `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run python -m pytest path/to/test_file.py -k "pattern"`
  - **Fast suite**: `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run python -m pytest -m fast`
  - **Slow suite**: `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run python -m pytest -m slow`
- **Do not update reference (“golden”) images** unless explicitly asked. If a visual mismatch happens, keep the outputs and report paths for review.

## Lint / format

- Use `ruff` via existing Make targets:
  - Lint: `make lint`
  - Format: `make format`
  - Full pre-commit: `make check`

## Releases

When preparing a release:

- Bump version in `pyproject.toml`
- Add a descriptive entry to `CHANGELOG.md`
- Update lockfile: `uv lock`
- (Optional sanity) run `make test-fast` and `make build`
- Tagging is handled by an external GitHub Action when publishing.

## Agent workflow norms (modern Cursor best practices)

- For multi-file or high-risk work, **start with a short plan** (bullets: goals, constraints, files to touch, how you’ll verify).
- Keep changes tight, and prefer **verifiable goals** (tests/lint/build) over speculation.
- If the task scope changes materially, stop and re-align rather than continuing in a confused state.

