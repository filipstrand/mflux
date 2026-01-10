# /pr

Create a pull request for the current changes (safe, repo-aware workflow).

This follows the example workflow described in Cursor’s agent best practices, adapted for mflux conventions.

## Steps

1. **Inspect changes**
   - `git status`
   - `git diff`

2. **(Optional) Quick verification**
   - Run the pre-commit suite to catch linting/formatting errors:
     - `/check` (or `make check`)
   - Prefer fast tests:
     - `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run python -m pytest -m fast`

3. **Commit**
   - Write a clear commit message based on the diff.
   - `git add -A`
   - `git commit -m "<message>"`

4. **Push**
   - **Stop and ask for permission** before pushing.
   - Once approved, push to the current branch:
     - `git push -u origin HEAD`

5. **Open PR**
   - If GitHub CLI is available:
     - `gh pr create --fill`
   - Otherwise: stop and ask for the preferred PR creation method (web UI vs installing `gh`).

## Notes

- Do not run long/slow suites unless requested; prefer fast tests for PR hygiene.
- Keep `MFLUX_PRESERVE_TEST_OUTPUT=1` on for any test runs that might produce images.

