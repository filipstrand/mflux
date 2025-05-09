import typing as t
from pathlib import Path

from mflux.error.exceptions import PromptFileReadError


def read_prompt_file(prompt_file_path: Path) -> str:
    """
    Read prompt from file and return the content as string.

    Args:
        prompt_file_path: Path to the prompt file

    Returns:
        String content of the prompt file

    Raises:
        PromptFileReadError: If the file cannot be read, doesn't exist, or has invalid content
    """
    # Check if file exists
    if not prompt_file_path.exists():
        raise PromptFileReadError(f"Prompt file does not exist: {prompt_file_path}")

    try:
        with open(prompt_file_path, "rt") as f:
            content: str = f.read().strip()

        # Validate content
        if not content:
            raise PromptFileReadError(f"Prompt file is empty: {prompt_file_path}")

        print(f"Using prompt from file: {prompt_file_path}")
        return content
    except (IOError, OSError) as e:
        raise PromptFileReadError(f"Error reading prompt file '{prompt_file_path}': {e}")


def get_effective_prompt(args: t.Any) -> str:
    """
    Get the effective prompt from either --prompt or --prompt-file.

    This helper function centralizes the logic for retrieving the prompt
    across all generate_*.py scripts.

    Args:
        args: The parsed command line arguments

    Returns:
        The effective prompt to use for generation

    Raises:
        PromptFileReadError: If there's an issue reading the prompt file
    """
    if args.prompt_file is not None:
        prompt = read_prompt_file(args.prompt_file)
        return prompt
    return args.prompt
