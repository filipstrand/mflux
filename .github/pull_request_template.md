## What

<!-- A headline sentence, then the benefits for each group of users. Link issues: Fixes #... -->

## Checklist (definition of done)

- [ ] **Tests added/updated run in CI by default**
Run `just test`.
Only mark `@pytest.mark.slow` or `@pytest.mark.high_memory_requirement` when a test exceeds CI's time or memory budget (typically weight downloads / image generation).<br><br>
- [ ] **`ruff check` and `ruff format` are clean**
`uv run ruff` uses the version pinned in the dev dependencies of `pyproject.toml`, which is the single source of truth for pre-commit and CI; `pre-commit run -a` covers it locally.<br><br>
- [ ] **Release note block below filled in**
Every PR gets a note. Do not write `none`.<br><br>
- [ ] **Docs updated where behavior changed**
README examples/table rows are part of the API contract (see `.cursor/rules/RULE.md`).<br><br>
- [ ] **New model: shared config wiring**
aliases, default steps, mflux-save dispatch, capabilities, completions, thin CLI entrypoint and `src/mflux/models/<name>/README.md`.<br><br>
- [ ] **New/changed CLI: ignored/rejected options declared**
`IGNORED_OPTIONS`/`REJECTED_OPTIONS` declared and `warn_ignored_options` actually called in `main()`, so `mflux-capabilities` stays truthful.

## Release note

```release-note
```

<!-- One or two sentences that describe the change, harvested into the release notes at
     release time. CI fails until you write the note. -->

## Verification

<!-- Commands you ran & what you observed - include generated images/screenshots for model-affecting changes. -->
