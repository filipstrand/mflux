# /release-prep

Prepare a release PR (no tagging/publishing).

## Checklist

- Update version in `pyproject.toml`
- Add a descriptive entry to `CHANGELOG.md`
- Update lockfile: `uv lock`
- Sanity checks:
  - `make test-fast`
  - `make build`

## Notes

- Tagging/publishing is handled by an external GitHub Action.
- If anything fails, stop and report what needs attention before continuing.

