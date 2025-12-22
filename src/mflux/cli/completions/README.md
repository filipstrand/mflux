# mflux Shell Completions

This module provides ZSH shell completion support for all mflux CLI commands.

## Quick Start

1. **Install completions:**
   ```bash
   mflux-completions
   ```

2. **Reload your shell:**
   ```bash
   exec zsh
   ```

3. **Test it:**
   ```bash
   mflux-generate --<TAB>
   ```

## Features

- **Auto-completion for all 24 mflux commands**: Generate, train, save, upscale, and more
- **Smart completions**: 
  - Model names (dev, schnell, dev-fill, or HuggingFace repos)
  - Quantization levels (3, 4, 5, 6, 8)
  - LoRA styles (storyboard, portrait, etc.)
  - File paths with native completion
  - Common percentage values for battery limit
- **Dynamic generation**: Completions stay in sync with code changes
- **Mutually exclusive options**: Properly handles -m/--model style options

## Installation

### Automatic Installation

The simplest way is to let mflux-completions auto-detect the best location:

```bash
mflux-completions
```

### Custom Directory

Install to a specific directory:

```bash
mflux-completions --dir ~/.config/zsh/completions
```

### Update Existing Installation

Update an existing completion file:

```bash
mflux-completions --update
```

## Troubleshooting

### Check Installation

Verify that completions are properly installed:

```bash
mflux-completions --check
```

This will:
- Check if the completion file exists in your $fpath
- Verify that compinit is available
- Provide troubleshooting steps if needed

### Common Issues

1. **Completions not working after installation:**
   - Start a new shell: `exec zsh`

2. **Completions not updated after mflux upgrade:**
   - Update completions: `mflux-completions --update`
   - Then force reload: `compinit -u`

3. **Permission denied:**
   - Use `--dir` to specify a user-writable directory

4. **Completion shows \} or other artifacts:**
   - Update to the latest version: `mflux-completions --update`

For Zsh-specific configuration issues (fpath, compinit, etc.), please refer to the [official Zsh documentation](https://zsh.sourceforge.io/Doc/Release/Completion-System.html).

### Manual Installation

If automatic installation fails:

```bash
# Generate completion script
mflux-completions --print > _mflux

# Move to a directory in your $fpath
mkdir -p ~/.zsh/completions
mv _mflux ~/.zsh/completions/

# Restart your shell
exec zsh
```

## How It Works

The completion system:

1. **Introspects argparse**: Dynamically extracts all command-line options from the parsers
2. **Generates ZSH functions**: Creates completion functions for each mflux command
3. **Provides context-aware completions**: Different completion strategies based on argument type
4. **Handles special cases**: Custom completions for models, LoRA styles, etc.

## Development

### Adding New Commands

When adding a new mflux CLI command:

1. Create the command with argparse as usual
2. Add it to the `commands` list in `generator.py`
3. Add the parser creation logic to `create_parser_for_command()`
4. Completions will be automatically generated!

### Testing Completions

Test the generator without installing:

```bash
python -m mflux.cli.completions.generator
```

### File Structure

- `generator.py`: Core logic for generating ZSH completion scripts
- `install.py`: Installation utility with user-friendly CLI
- `README.md`: This documentation

## Platform Support

Currently supports ZSH (the default macOS shell since 2019). Supporting other shells is possible but is considered out of scope for the official repository.
