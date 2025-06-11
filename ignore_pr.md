# Add stdin support for prompt input

## Summary

This PR adds support for reading prompts from stdin when `--prompt -` is specified, following Unix/Linux conventions. This enables piping prompts from other commands, making mflux more composable in shell scripts and workflows.

## Changes

### Feature Implementation
- **Added stdin support in `prompt_utils.py`**: When `--prompt -` is provided, the prompt is read from stdin
- **Follows Unix conventions**: Uses `-` as the special marker for stdin input, consistent with tools like `cat`, `grep`, etc.
- **Proper error handling**: Raises `PromptFileReadError` with clear message when stdin is empty
- **Preserves existing behavior**: Regular prompts and `--prompt-file` continue to work as before

### Code Improvements
- **Converted print statements to logging**: Changed `print()` calls to `logger.info()` for better control over output
- **Improved type annotations**: Changed `get_effective_prompt(args: t.Any)` to use proper `argparse.Namespace` type
- **Added comprehensive tests**: Both unit tests and integration tests verify the functionality

## Usage Examples

```bash
# Pipe prompt from echo
echo "A beautiful sunset over mountains" | mflux-generate --prompt - --model dev --steps 20

# Pipe prompt from a file
cat prompt.txt | mflux-generate --prompt - --model dev --output result.png

# Use in a pipeline
generate-prompt-script.py | mflux-generate --prompt - --model dev --metadata
```

## Testing

### Unit Tests (`tests/arg_parser/test_stdin_prompt.py`)
- ✅ Basic stdin parsing (`--prompt -` returns "-")
- ✅ Regular prompt still works unchanged
- ✅ Whitespace trimming from stdin
- ✅ `--prompt-file` takes precedence over stdin

### Integration Tests (`tests/image_generation/test_stdin_prompt_integration.py`)
- ✅ Single-line stdin prompt generates image with correct metadata
- ✅ Multi-line stdin prompt is preserved correctly
- ✅ Empty stdin shows appropriate error message
- ✅ Shell piping with echo command works correctly

All tests verify that the stdin prompt is correctly captured in the output metadata when `--metadata` flag is used.

## Technical Details

- Stdin reading happens in `get_effective_prompt()` when `args.prompt == "-"`
- Uses `sys.stdin.read().strip()` to read and clean the input
- Logging messages indicate when prompt is read from stdin vs file
- Error handling for `IOError`, `OSError`, and `KeyboardInterrupt`

## Breaking Changes

None. This is a purely additive feature that doesn't affect existing functionality.

## Future Considerations

This implementation could be extended to other input arguments if needed, following the same `-` convention for stdin input.