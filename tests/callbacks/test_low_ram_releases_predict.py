import gc
import weakref
from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.variants import Flux3Klein
from mflux.models.krea2.variants import Krea2
from mflux.models.z_image.variants.z_image import ZImage


class _FakeTokenizer:
    def tokenize(self, prompt: str, images=None, max_length=None, **kwargs):
        length = max(len(prompt), 1)
        input_ids = mx.arange(length, dtype=mx.int32)[None, :]
        attention_mask = mx.ones((1, length), dtype=mx.int32)
        return type("TokenizerOutput", (), {"input_ids": input_ids, "attention_mask": attention_mask})()


class _FakeTextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask=None):
        return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.float32)


class _FakeKrea2Transformer:
    def __call__(self, hidden_states, timestep, context, attention_mask=None):
        return mx.ones_like(hidden_states)


class _FakeZImageTransformer:
    def __call__(self, timestep, x, cap_feats, sigmas):
        return x * 2


class _FakeVAE:
    def decode(self, latents):
        return mx.zeros((latents.shape[0], 3, latents.shape[2] * 8, latents.shape[3] * 8))


class _DropTransformerAfterLoop:
    # Does what MemorySaver._delete_transformer does, then records if the transformer is still alive.
    def __init__(self, model: Krea2) -> None:
        self.model = model
        self.transformer_ref = weakref.ref(model.transformer)
        self.alive_after_loop: bool | None = None

    def call_after_loop(self, seed, prompt, latents, config) -> None:
        self.model.transformer = None
        gc.collect()
        self.alive_after_loop = self.transformer_ref() is not None


def _krea2_model() -> Krea2:
    model = Krea2.__new__(Krea2)
    model.model_config = ModelConfig.krea2()
    model.callbacks = CallbackRegistry()
    model.tokenizers = {"qwen3vl": _FakeTokenizer()}
    model.transformer = _FakeKrea2Transformer()
    model.vae = _FakeVAE()
    model.text_encoder = _FakeTextEncoder()
    model.prompt_cache = {}
    model.tiling_config = None
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    return model


@pytest.mark.fast
def test_generate_image_releases_transformer_before_after_loop_callbacks():
    model = _krea2_model()
    callback = _DropTransformerAfterLoop(model)
    model.callbacks.register(callback)

    with patch("mflux.models.krea2.variants.txt2img.krea2.AppleSiliconUtil.is_m1_or_m2", return_value=False):
        model.generate_image(
            seed=1, prompt="x", num_inference_steps=2, height=64, width=64, guidance=1.0, scheduler="euler"
        )

    assert callback.alive_after_loop is False


@pytest.mark.fast
def test_compiled_z_image_predict_does_not_keep_transformer_after_release():
    transformer = _FakeZImageTransformer()
    transformer_ref = weakref.ref(transformer)
    with patch("mflux.models.z_image.variants.z_image.AppleSiliconUtil.is_m1_or_m2", return_value=False):
        predict = ZImage._predict(transformer)
    latents = mx.ones((1, 4, 2, 2))
    sigmas = mx.array([1.0, 0.0])
    # Call it one time so that MLX traces it and fills its compile cache.
    mx.eval(predict(latents, mx.array([0.5]), sigmas, mx.zeros((1, 3, 8)), None, 1.0))

    del transformer
    gc.collect()
    assert transformer_ref() is not None

    del predict
    gc.collect()
    assert transformer_ref() is None


class _FakeFlux3Transformer:
    flow_map = True

    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["hidden_states"])


class _FakePackedVAE(Flux2VAE):
    def __init__(self) -> None:  # no weights: only decode_packed_latents is exercised, and it is replaced below
        pass

    def decode_packed_latents(self, packed_latents, tiling_config=None):
        return mx.zeros((packed_latents.shape[0], 3, packed_latents.shape[2] * 16, packed_latents.shape[3] * 16))


@pytest.mark.fast
def test_flux3_generate_image_releases_transformer_before_after_loop_callbacks():
    # MemorySaver frees the transformer in call_after_loop; the predict closure must not keep it alive.
    model = Flux3Klein.__new__(Flux3Klein)
    model.model_config = ModelConfig.flux3_image_klein()
    model.callbacks = CallbackRegistry()
    model.transformer = _FakeFlux3Transformer()
    model.vae = _FakePackedVAE()
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    model.compute_precision = ComputePrecision()
    model.tiling_config = None
    model._noise_override = None
    model.prompt_cache = {"x": (mx.zeros((1, 80, 8)), mx.zeros((80, 4)))}
    callback = _DropTransformerAfterLoop(model)
    model.callbacks.register(callback)

    model.generate_image(seed=1, prompt="x", num_inference_steps=2, height=64, width=64)

    assert callback.alive_after_loop is False
