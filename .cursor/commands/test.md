# /test

Run the full test suite while preserving generated images for inspection.

## Command

`MFLUX_PRESERVE_TEST_OUTPUT=1 uv run python -m pytest`

## Notes

- If tests fail, summarize the failing tests and the key assertion diffs/paths.
- Do **not** regenerate or replace reference images unless explicitly asked.

