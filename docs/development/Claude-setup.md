# Claude Setup

### *...separating shared MFlux Claude files from personal Claude files!*
<br><br>

### Claude Code Scopes
Claude Code separates shared files from personal files by file name and location. Git decides which files other developers get:

|Scope|Files|Committed?|
|---|---|---|
|**Shared (project)**<br><br><br>|`AGENTS.md`,<br>`.claude/settings.json`,<br>`.claude/skills/`|Yes|
|**Personal, this project only**<br><br><br>|`CLAUDE.md`,<br>`CLAUDE.local.md`,<br>`.claude/settings.local.json`|No (gitignored)|
|**Personal, all projects**<br><br><br>|`~/.claude/CLAUDE.md`,<br>`~/.claude/skills/`,<br>`~/.claude/settings.json`|Not in the repo|

- This repo ignores `CLAUDE.md`. Put shared agent rules in `AGENTS.md`.
- This repo also ignores `.claude/agents/` and `.mcp.json`. To share one of them, add a `!` rule to `.gitignore`.


### Skills
Each folder in `.claude/skills/` is a symlink to a folder in `.cursor/skills/`. Edit the skill in `.cursor/skills/`. Cursor and Claude Code then read the same file.

To add a skill:
1. Create `.cursor/skills/<name>/SKILL.md`.
2. Make the symlink: `ln -s ../../.cursor/skills/<name> .claude/skills/<name>`.


### Personal scratch files
Add personal files that are not Claude files to `.git/info/exclude`. Git reads this file only on your computer. Other developers do not get it. Example: `z*.*`.


### `.gitignore` setup

Git reads `#` as a comment only at the start of a line. Thus each comment is on its own line.

```
# Ignore the whole .claude directory by default.
.claude/*
# Share the skills.
!.claude/skills/
# Share the settings.
!.claude/settings.json
# Ignore the personal settings.
.claude/settings.local.json
# Ignore the GitNexus skills that `gitnexus analyze` makes.
.claude/skills/gitnexus-*/
# Ignore CLAUDE.md. Use AGENTS.md for shared rules.
CLAUDE.md
# Ignore the personal agent instructions.
CLAUDE.local.md
# Ignore the PR files that /mflux-pr-docs makes.
tmp-PR-content.md
tmp-PR-body.md
```

### Sources
- [Claude Code settings](https://code.claude.com/docs/en/settings.md)
- [Explore the .claude directory](https://code.claude.com/docs/en/claude-directory)
