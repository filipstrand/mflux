---
name: mflux-pr
description: Make a clean PR in mflux (inspect diff, quick verification, commit, push, open PR) using repo conventions.
---
# mflux pull request workflow

## When to Use

- You’re about to open a PR (or want a safe sequence to do it).

## Instructions

- If you run tests as part of PR hygiene, prefer fast tests first:
  - `make test-fast`
- Keep commits focused and messages consistent with repo history.
- If the PR changes CLI defaults, public APIs, or model behavior, check for README/example drift before opening the PR.
- **Always ask for permission** before pushing to the remote repository.
- If `gh` isn’t available, fall back to the GitHub web UI (or stop and ask).

