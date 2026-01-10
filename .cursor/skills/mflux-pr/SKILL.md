---
name: mflux-pr
description: Make a clean PR in mflux (inspect diff, quick verification, commit, push, open PR) using repo conventions.
---
# mflux pull request workflow

## When to Use

- You’re about to open a PR (or want a safe sequence to do it).

## Instructions

- Prefer the existing Cursor command:
  - `/pr`
- If you run tests as part of PR hygiene, prefer fast tests first:
  - `/test-fast`
- Keep commits focused and messages consistent with repo history.
- **Always ask for permission** before pushing to the remote repository.
- If `gh` isn’t available, fall back to the GitHub web UI (or stop and ask).

