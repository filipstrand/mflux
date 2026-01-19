---
name: mflux-release
description: Prepare a release in mflux (version bump, changelog, uv lock) without tagging/publishing.
---
# mflux release prep

Releases are prepared in-repo; tagging/publishing is handled by GitHub Actions.

## When to Use

- You’re preparing a release PR.
- You need to bump version and update the changelog/lockfile correctly.

## Instructions

- Prefer the existing Cursor command:
  - `/release-prep`
- Checklist:
  - Bump version in `pyproject.toml`
  - Review commits since last release tag:
    - `git log --oneline v.<last-version>..HEAD`
  - Browse relevant file contents if needed to clarify changes
  - Weigh commit impact: some are minor; skim earlier changelog entries to gauge what's worth reporting vs tiny fixes
  - Add a descriptive entry to `CHANGELOG.md` based on those commits
  - Update lockfile: `uv lock`
  - Sanity checks (optional unless requested): `make test-fast`, `make build`
  - Manual checks (optional): if the release includes CLI/callback/image-path changes, consider running the `mflux-manual-testing` skill to exercise the touched commands and visually review outputs.
- Do not tag releases locally unless explicitly requested (normally handled by CI).

