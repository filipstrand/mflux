from pathlib import Path

import transformers

from mflux.models.qwen.tokenizer.qwen_tokenizer import TokenizerQwen
from mflux.utils.download import snapshot_download


class QwenTokenizerHandler:
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
