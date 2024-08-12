import transformers

from flux_1_schnell.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1_schnell.tokenizer.t5_tokenizer import TokenizerT5


class TokenizerHandler:

    def __init__(self, root_path: str):
        self.clip = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path + "/tokenizer",
            local_files_only=True,
            max_length=TokenizerCLIP.MAX_TOKEN_LENGTH
        )
        self.t5 = transformers.T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=root_path + "/tokenizer_2",
            local_files_only=True,
            max_length=TokenizerT5.MAX_TOKEN_LENGTH
        )

    @staticmethod
    def load_from_disk_via_huggingface_transformers(root_path: str) -> "TokenizerHandler":
        return TokenizerHandler(root_path)
