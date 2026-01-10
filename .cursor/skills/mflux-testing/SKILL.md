---
name: mflux-testing
description: Run tests in mflux (fast/slow/full), preserve image outputs, and handle golden image diffs safely.
---
# mflux testing

This repo uses pytest with image-producing tests. Always preserve outputs for inspection and **never** update reference images unless explicitly asked.

## When to Use

- You need to run tests (fast/slow/full) or debug failing tests.
- There are image/golden mismatches and you need to report paths/output for review.

## Instructions

- Prefer the existing Cursor commands:
  - `/test-fast` (fast tests, no image generation)
  - `/test-slow` (slow tests, image generation)
  - `/test` (full suite)
- Always keep `MFLUX_PRESERVE_TEST_OUTPUT=1` on test runs (already built into the commands).
- If tests fail:
  - Summarize the failing test names and the key assertion output.
  - Point to any generated images/artifacts on disk for manual review.
- Do **not** regenerate/replace reference (“golden”) images unless the user explicitly requests it.

