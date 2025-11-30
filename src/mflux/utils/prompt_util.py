import logging
import sys
from argparse import Namespace
from pathlib import Path

from mflux.utils.exceptions import PromptFileReadError

logger = logging.getLogger(__name__)


class PromptUtil:
    @staticmethod
    def read_prompt(args: Namespace) -> str:
        # Handle stdin input when prompt is "-"
        if args.prompt == "-":
            try:
                content = sys.stdin.read().strip()
                if not content:
                    raise PromptFileReadError("No prompt provided via stdin")
                logger.info("Using prompt from stdin")
                return content
            except (IOError, OSError, KeyboardInterrupt) as e:
                raise PromptFileReadError(f"Error reading from stdin: {e}")

        # Handle prompt file
        if args.prompt_file is not None:
            prompt = PromptUtil._read_prompt_file(args.prompt_file)
            return prompt

        # Return regular prompt
        return args.prompt

    @staticmethod
    def read_negative_prompt(args: Namespace) -> str:
        # Return negative prompt or empty string if not provided
        return getattr(args, "negative_prompt", "")

    @staticmethod
    def _read_prompt_file(prompt_file_path: Path) -> str:
        # Check if file exists
        if not prompt_file_path.exists():
            raise PromptFileReadError(f"Prompt file does not exist: {prompt_file_path}")

        try:
            with open(prompt_file_path, "rt") as f:
                content: str = f.read().strip()

            # Validate content
            if not content:
                raise PromptFileReadError(f"Prompt file is empty: {prompt_file_path}")

            logger.info(f"Using prompt from file: {prompt_file_path}")
            return content
        except (IOError, OSError) as e:
            raise PromptFileReadError(f"Error reading prompt file '{prompt_file_path}': {e}")
