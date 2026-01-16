---
name: Skill Md Generalization
overview: Create a new, concise ML-focused SKILL.md that generalizes the Flux2Klein skill into a reusable checklist with brief narrative, includes uv usage and testing guidance, and documents the “port fast then refactor” workflow.
todos: []
---

# Generalize Porting Skill Doc

## Scope

- Create a new concise `SKILL.md` for internal ML model porting work (diffusers/MLX style), based only on the existing Flux2Klein skill as reference.
- Include `uv` usage and testing expectations; omit release guidance.
- Explicitly document the workflow: port to match reference style, validate with deterministic MLX test, then refactor to preferred style.
- Capture lessons from the post-test refactor phase by reviewing commits after `c5ad921bb2d6b38e0fa7bf2e5dcdc7da025e0a58` for style and cleanup preferences.

## Proposed Changes

- Review the existing Flux2Klein skill for reusable structure and terminology.
- Review the commit series after `c5ad921bb2d6b38e0fa7bf2e5dcdc7da025e0a58` to extract refactoring preferences and missing AI steps.
- Draft a short checklist + brief narrative that covers:
- Requirements intake and parity goals
- Reference implementation audit (key files, configs, weights)
- Porting steps (ops mapping, shape checks, weight conversion)
- Recommended workflow: port fast to match reference, lock in with deterministic test, then refactor
- Validation (golden outputs, visual tests, perf sanity)
- Tooling (use `uv run`, testing notes)
- Documentation deliverables (what to record for future ports)
- Place the new doc under `.cursor/skills/` in a new directory with a descriptive name.

## Files Likely Touched

- `.cursor/skills/<new-skill-dir>/SKILL.md`
- `.cursor/skills/mflux-flux2-porting/SKILL.md` as reference only

## Open Decision (already resolved)

- Include `uv` and testing guidance; skip release steps.

## Implementation Todos

- `scan-skill-plan`: Review existing Flux2Klein skill to extract reusable structure.
- `scan-commit-history`: Review commits after `c5ad921bb2d6b38e0fa7bf2e5dcdc7da025e0a58` to extract refactor preferences and missing AI steps.
- `draft-skill`: Write generalized short checklist + narrative with ML porting focus.
- `polish-skill`: Ensure `uv` usage and testing guidance are clear and concise.