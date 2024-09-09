from pathlib import Path

import transformers
from huggingface_hub import snapshot_download

from mflux.tokenizer.clip_tokenizer import TokenizerCLIP


class TokenizerHandler:

    def __init__(self, repo_id: str, max_t5_length: int = 256, local_path: str | None = None):
        root_path = Path(local_path) if local_path else TokenizerHandler._download_or_get_cached_tokenizers(repo_id)

        self.clip = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path / "tokenizer",
            local_files_only=True,
            max_length=TokenizerCLIP.MAX_TOKEN_LENGTH
        )
        self.t5 = transformers.T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path / "tokenizer_2",
            local_files_only=True,
            max_length=max_t5_length
        )

    @staticmethod
    def _download_or_get_cached_tokenizers(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "tokenizer/**",
                    "tokenizer_2/**"
                ]
            )
        )
