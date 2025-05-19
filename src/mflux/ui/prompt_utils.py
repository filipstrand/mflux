import typing as t
from pathlib import Path

from mflux.error.exceptions import PromptFileReadError


def read_prompt_file(prompt_file_path: Path) -> str:
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
    if args.prompt_file is not None:
        prompt = read_prompt_file(args.prompt_file)
        return prompt
    return args.prompt
