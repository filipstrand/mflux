# Cursor Terminal Hang Fix

This folder contains shell configuration files that fix Cursor's terminal hanging and execution issues. These files are reference copies of working zsh configurations that resolve problems where commands work fine on remote hosts but fail or hang when executed in Cursor's local terminal environment.

**Purpose:** When developing the mflux-debugger, we encountered issues where terminal commands (including debugger CLI commands) would hang or fail to execute in Cursor, but worked perfectly on remote hosts. This directory contains the shell configuration fixes that resolved those issues.

## Problem

Cursor's AI agent uses non-interactive shells, but was sourcing `.zshrc` which is intended for interactive shells only. This caused terminal hangs and execution issues, especially when:
- Interactive prompts were triggered (like Oh My Zsh update prompts)
- Shell initialization scripts expected user input
- Environment setup conflicted with Cursor's non-interactive execution
- Commands and tools didn't appear or execute properly (e.g., aliases, PATH-dependent tools)
- Operations would pause or hang indefinitely

**Real-world symptoms:**
- Commands that work fine on remote hosts fail to execute in Cursor's terminal
- Tools/aliases don't appear or aren't found
- Terminal operations pause indefinitely waiting for input
- Debugger commands and other CLI tools hang or don't respond

**Why it works on remote hosts:** Remote SSH sessions typically use proper interactive shells that handle `.zshrc` correctly, while Cursor's local terminal uses non-interactive shells that get stuck on interactive prompts in `.zshrc`.

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
   cp src/mflux_debugger/cursor-terminal-fix/.zshenv ~/.zshenv
   ```

2. **Restore `.zprofile`**:
   ```bash
   cp src/mflux_debugger/cursor-terminal-fix/.zprofile ~/.zprofile
   ```

3. **Restore `.zshrc`**:
   ```bash
   cp src/mflux_debugger/cursor-terminal-fix/.zshrc ~/.zshrc
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

## When to Use This Fix

Use these configuration files if you experience:
- Terminal hangs when Cursor's AI agent runs commands
- Commands that work on remote hosts but fail locally in Cursor
- Tools/aliases not appearing or not being found
- Operations pausing indefinitely
- Debugger CLI commands (`mflux-debug-*`) hanging or not responding

**Note:** This fix is specifically for Cursor's terminal environment. If you're only using remote hosts or standard terminals, you may not need these changes.

## Result

After applying this fix:
✅ Terminal is now responsive in Cursor  
✅ Commands execute properly without hanging  
✅ Environment variables still available  
✅ PATH still configured correctly  
✅ Tools and aliases work as expected  
✅ No more hangs on Cursor agent commands  
✅ Debugger CLI commands execute reliably

