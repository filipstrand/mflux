import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common_models.qwen3_5 import Qwen35TextModel


class MOSSImagePromptEncoder:
    @staticmethod
    def encode(prompt: str, tokenizer: LanguageTokenizer, text_encoder: Qwen35TextModel) -> mx.array:
        # Each prompt goes through the chat template on its own, without padding, as in the
        # reference: padding would change the Gated DeltaNet state. An empty prompt still gets the
        # template, so the negative branch of CFG has text tokens.
        tokens = tokenizer.tokenize(prompt, keep_empty=True)
        return text_encoder(tokens.input_ids)
