# Early return for Cursor agent and non-interactive shells (prevents terminal hangs)
# Cursor sets CURSOR_AGENT=1, and also uses non-interactive shells
if [[ -n "$CURSOR_AGENT" ]] || [[ ! -o interactive ]]; then
    return
fi

# Interactive shell only setup
[ -f /opt/homebrew/etc/profile.d/autojump.sh ] && . /opt/homebrew/etc/profile.d/autojump.sh

