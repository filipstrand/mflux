from pathlib import Path

import transformers
from huggingface_hub import snapshot_download

from flux_1_schnell.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1_schnell.tokenizer.t5_tokenizer import TokenizerT5


class TokenizerHandler:

    def __init__(self, repo_id: str):
        root_path = TokenizerHandler._download_or_get_cached_tokenizers(repo_id)

        self.clip = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path / "tokenizer",
            local_files_only=True,
            max_length=TokenizerCLIP.MAX_TOKEN_LENGTH
        )
        self.t5 = transformers.T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path / "tokenizer_2",
            local_files_only=True,
            max_length=TokenizerT5.MAX_TOKEN_LENGTH
        )

    @staticmethod
    def load_from_disk_or_huggingface(repo_id: str) -> "TokenizerHandler":
        return TokenizerHandler(repo_id)

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
