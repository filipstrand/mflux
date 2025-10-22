# Cursor Terminal Hang Fix

This folder contains the shell configuration files that fix Cursor's terminal hanging issues.

## Problem

Cursor's AI agent uses non-interactive shells, but was sourcing `.zshrc` which is intended for interactive shells only. This caused terminal hangs, especially when:
- Interactive prompts were triggered (like Oh My Zsh update prompts)
- Shell initialization scripts expected user input
- Environment setup conflicted with Cursor's non-interactive execution

## Solution

We restructured the zsh configuration following proper zsh semantics:

### File Structure

1. **`.zshenv`** - Environment variables for ALL shells
   - Sourced by every zsh instance (interactive and non-interactive)
   - Contains: `LC_ALL`, `LANG`, `LANGUAGE`, `HF_HOME`
   - **Location**: `~/.zshenv`

2. **`.zprofile`** - Login shell setup
   - Sourced for login shells (which Cursor uses)
   - Contains: PATH adjustments via `~/.local/bin/env`
   - **Location**: `~/.zprofile`

3. **`.zshrc`** - Interactive shell only setup
   - Early return for Cursor agent and non-interactive shells
   - Only loads interactive features (like autojump)
   - **Location**: `~/.zshrc`

### Key Fix: Early Return in `.zshrc`

The critical fix is the early return check at the top of `.zshrc`:

```zsh
# Early return for Cursor agent and non-interactive shells (prevents terminal hangs)
# Cursor sets CURSOR_AGENT=1, and also uses non-interactive shells
if [[ -n "$CURSOR_AGENT" ]] || [[ ! -o interactive ]]; then
    return
fi
```

This checks for:
- `CURSOR_AGENT=1` - Environment variable set by Cursor
- Non-interactive shell mode - Standard zsh check

When either condition is true, `.zshrc` returns immediately, preventing any interactive prompts or hangs.

## What Changed

### Before
- All configuration was in `.zshrc`
- Cursor sourced `.zshrc` in non-interactive mode
- Terminal would hang on interactive prompts

### After
- Environment variables moved to `.zshenv` (available to all shells)
- PATH setup moved to `.zprofile` (available to login shells)
- `.zshrc` only loads for interactive shells
- Cursor gets environment variables without hanging

## Files in This Folder

- `.zshrc` - Your current `.zshrc` configuration
- `.zshenv` - Your current `.zshenv` configuration  
- `.zprofile` - Your current `.zprofile` configuration
- `.local-bin-env` - Copy of `~/.local/bin/env` (for reference)

## Restoration Instructions

If you need to restore these configurations:

1. **Restore `.zshenv`**:
   ```bash
   cp cursor-terminal-fix/.zshenv ~/.zshenv
   ```

2. **Restore `.zprofile`**:
   ```bash
   cp cursor-terminal-fix/.zprofile ~/.zprofile
   ```

3. **Restore `.zshrc`**:
   ```bash
   cp cursor-terminal-fix/.zshrc ~/.zshrc
   ```

## References

- [Cursor Forum Guide](https://forum.cursor.com/t/guide-fix-cursor-agent-terminal-hangs-caused-by-zshrc/107260/31)
- [Reddit Discussion](https://www.reddit.com/r/cursor/comments/1msdwto/i_really_wish_cursor_would_fix_the_agent_choking/)

## Technical Details

### Zsh Initialization Order

1. `.zshenv` - Always sourced first (all shells)
2. `.zprofile` - Sourced for login shells
3. `.zshrc` - Sourced for interactive shells only

### Cursor's Shell Environment

When Cursor runs commands, it:
- Uses login shells (sources `.zprofile`)
- Sets `CURSOR_AGENT=1` environment variable
- Uses non-interactive mode (doesn't source `.zshrc` fully)

### Why This Works

- Environment variables are still available via `.zshenv`
- PATH is still configured via `.zprofile`
- Interactive-only features (autojump) are skipped
- No interactive prompts can hang the terminal

## Result

✅ Terminal is now responsive  
✅ Environment variables still available  
✅ PATH still configured correctly  
✅ No more hangs on Cursor agent commands

