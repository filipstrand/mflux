---
name: mflux-release
description: Prepare a release in mflux (version bump + uv lock; release notes are harvested from PR release-note blocks and live in GitHub Releases). Use when preparing a release branch or release PR.
---
# mflux release prep

Releases are prepared in-repo; tagging/publishing is handled by GitHub Actions
(`release.yml`, dispatched from main with the `publish` confirmation).

## How notes work now (#685)

- Every PR carries a fenced ` ```release-note ` block in its body (CI enforces it;
  every PR gets a note, and contributors do not write `none`). That block is the only source of release notes.
- On dispatch, the ungated `draft-notes` job harvests the blocks for every PR whose
  squash commit is in `previous-tag..HEAD` and fills a DRAFT GitHub release, grouped
  by label.
- Whoever approves the `pypi` deployment reads and edits that draft (fixing any
  `[needs edit]` lines, adding contributor thanks if wanted); the gated job publishes
  it exactly as edited. A re-dispatch never overwrites an existing draft; delete the
  draft to re-harvest.
- There is no `CHANGELOG.md`: the notes live in GitHub Releases (https://github.com/mflux-community/mflux/releases).

## Release-prep PR checklist

- Bump version in `pyproject.toml`
- Update lockfile: `uv lock`
- Release-note block of the prep PR itself: a note for contributors, for example `Prepare the <version> release.` Give the PR the `chore` label.
- Prefer one commit named `release: prepare <version>`
- Sanity checks (optional unless requested): `just test-fast`, `just build`
- Do not tag releases locally unless explicitly requested (normally handled by CI)
