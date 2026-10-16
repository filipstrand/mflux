from fnmatch import fnmatch

import pytest

from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)

# Files that mflux-save writes for a FLUX 3 model. A saved model hosted on the Hugging Face Hub is fetched with the
# download patterns, so a file that no pattern matches is missing after the download (the chat template was).
SAVED_FILES = [
    "transformer/0.safetensors",
    "transformer/model.safetensors.index.json",
    "text_encoder/0.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/tokenizer.json",
    "text_encoder/tokenizer_config.json",
    "text_encoder/chat_template.jinja",
    "vae/0.safetensors",
    "vae/model.safetensors.index.json",
]


@pytest.mark.parametrize("definition", [Flux3KleinWeightDefinition, Flux3KleinBaseWeightDefinition])
@pytest.mark.parametrize("path", SAVED_FILES)
def test_download_patterns_fetch_every_saved_file(definition, path):
    patterns = list(definition.get_download_patterns())
    for tokenizer in definition.get_tokenizers():
        patterns.extend(tokenizer.download_patterns or [])
    assert any(fnmatch(path, pattern) for pattern in patterns), f"{path} is not downloaded"
