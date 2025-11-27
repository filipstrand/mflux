from pathlib import Path

import transformers

from mflux.models.fibo.tokenizer.fibo_tokenizer import TokenizerFibo
from mflux.utils.download import snapshot_download


class FiboTokenizerHandler:
    def __init__(
        self,
        repo_id: str,
        bot_token_id: int = 128000,
        local_path: str | None = None,
    ):
        root_path = Path(local_path) if local_path else FiboTokenizerHandler._download_or_get_cached_tokenizer(repo_id)

        # Try different possible tokenizer paths
        tokenizer_path = root_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = root_path / "text_encoder"
        if not tokenizer_path.exists():
            tokenizer_path = root_path

        # Load the raw tokenizer
        self.tokenizer_raw = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(tokenizer_path),
            local_files_only=True,
            fix_mistral_regex=True,  # Fix Mistral regex pattern warning
        )

        # Wrap in our TokenizerFibo class
        self.fibo = TokenizerFibo(self.tokenizer_raw, bot_token_id=bot_token_id)

    @staticmethod
    def _download_or_get_cached_tokenizer(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["tokenizer/**", "text_encoder/**"],
            )
        )
