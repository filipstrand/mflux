---
name: mflux-dev-env
description: Set up and work in the mflux dev environment (arm64 expectation, uv, Makefile targets, lint/format/test).
---
# mflux dev environment

This repo expects macOS arm64 and prefers `uv` + Makefile targets.

## When to Use

- You’re setting up the repo locally or diagnosing environment/setup issues.
- You need the canonical way to run lint/format/check/build/test.

## Instructions

- Prefer Makefile targets:
  - Install: `make install`
  - Lint: `make lint`
  - Format: `make format`
  - Pre-commit suite: `make check`
  - Build: `make build`
- Prefer `uv run ...` for running Python commands to ensure the correct environment.
- When running tests, keep `MFLUX_PRESERVE_TEST_OUTPUT=1` enabled (the Makefile test targets already do this).

