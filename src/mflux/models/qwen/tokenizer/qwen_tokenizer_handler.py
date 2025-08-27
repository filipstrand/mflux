from pathlib import Path

import transformers

from mflux.tokenizer.qwen_tokenizer import TokenizerQwen
from mflux.weights.download import snapshot_download


class QwenTokenizerHandler:
    """
    Handler for Qwen tokenizer, following the same pattern as TokenizerHandler.

    This loads the Qwen2Tokenizer from the model repository and wraps it
    in our TokenizerQwen class for consistent interface.
    """

    def __init__(
        self,
        repo_id: str,
        max_length: int = 1024,
        local_path: str | None = None,
    ):
        root_path = Path(local_path) if local_path else QwenTokenizerHandler._download_or_get_cached_tokenizer(repo_id)

        # Load the Qwen2 tokenizer
        tokenizer_path = root_path / "tokenizer"
        self.tokenizer_raw = transformers.Qwen2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
            local_files_only=True,
        )

        # Wrap in our TokenizerQwen class
        self.qwen = TokenizerQwen(self.tokenizer_raw, max_length)

    @staticmethod
    def _download_or_get_cached_tokenizer(repo_id: str) -> Path:
        """Download or get cached tokenizer files."""
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "tokenizer/**",
                    "added_tokens.json",
                    "chat_template.jinja",
                ],
            )
        )
