import mlx.core as mx
import pytest

from mflux.models.iris.model.iris_text_encoder.prompt_encoder import IrisPromptEncoder

PREFIX_IDS = [1000 + i for i in range(34)]
SUFFIX_IDS = [2000 + i for i in range(5)]
PAD = 151643


class _HFTokenizer:
    # One id per word; the chat template pieces map to their fixed 34 and 5 ids.
    pad_token_id = PAD
    eos_token_id = 99

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        if text == IrisPromptEncoder.PREFIX:
            return list(PREFIX_IDS)
        if text == IrisPromptEncoder.SUFFIX:
            return list(SUFFIX_IDS)
        raise AssertionError(text)

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [3000 + i for i in range(len(text.split()))]}


class _Tokenizer:
    tokenizer = _HFTokenizer()


@pytest.mark.fast
class TestIrisPromptEncoderTokenize:
    def test_a_short_prompt_is_prefix_caption_suffix_then_pad(self):
        input_ids, mask = IrisPromptEncoder.tokenize("a red fox", _Tokenizer())

        ids = input_ids[0].tolist()
        assert input_ids.shape == mask.shape == (1, 334)
        assert ids[:34] == PREFIX_IDS
        assert ids[34:37] == [3000, 3001, 3002]
        assert ids[37:42] == SUFFIX_IDS
        assert set(ids[42:]) == {PAD}
        assert mask[0].tolist() == [1] * 42 + [0] * 292

    def test_an_empty_prompt_is_prefix_and_suffix_only(self):
        input_ids, mask = IrisPromptEncoder.tokenize("", _Tokenizer())

        assert input_ids[0, :39].tolist() == PREFIX_IDS + SUFFIX_IDS
        assert int(mask.sum()) == 39

    def test_a_long_prompt_is_cut_to_295_caption_tokens_and_keeps_its_suffix(self):
        input_ids, mask = IrisPromptEncoder.tokenize(" ".join(["w"] * 500), _Tokenizer())

        ids = input_ids[0].tolist()
        assert ids[34:329] == [3000 + i for i in range(295)]
        assert ids[329:] == SUFFIX_IDS
        assert int(mask.sum()) == 334

    def test_the_caption_budget_is_exactly_295(self):
        _, at_budget = IrisPromptEncoder.tokenize(" ".join(["w"] * 295), _Tokenizer())
        input_ids, over = IrisPromptEncoder.tokenize(" ".join(["w"] * 296), _Tokenizer())

        assert int(at_budget.sum()) == int(over.sum()) == 334
        assert input_ids[0, 328].item() == 3000 + 294


class _Encoder:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        self.calls.append((input_ids.shape, output_hidden_states))
        hidden = [mx.full((1, input_ids.shape[1], 4), float(layer)) for layer in range(37)]
        return None, hidden


@pytest.mark.fast
def test_encode_prompt_stacks_the_twelve_layers_and_zeroes_padding():
    encoder = _Encoder()

    embeddings, mask = IrisPromptEncoder.encode_prompt("a red fox", _Tokenizer(), encoder)

    assert encoder.calls == [((1, 334), True)]
    assert embeddings.shape == (1, 300, 12, 4)
    assert mask.shape == (1, 300)
    assert int(mask.sum()) == 8
    assert embeddings[0, 0, :, 0].tolist() == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    assert float(mx.abs(embeddings[0, 8:]).max()) == 0.0
