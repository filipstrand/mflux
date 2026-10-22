import json
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_vae import small_decoder
from mflux.models.flux2.model.flux2_vae.small_decoder import (
    SMALL_DECODER_BLOCK_OUT_CHANNELS,
    SMALL_DECODER_FILE,
    SMALL_DECODER_REPO,
    Flux2SmallDecoderLoader,
)
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit


def _weights_of(vae: Flux2VAE, seed: int = 0) -> dict:
    # A nested tree in the form WeightLoader returns, with values that differ from a fresh init.
    keys = mx.random.split(mx.random.key(seed), len(tree_flatten(vae.parameters())))
    flat = [(path, mx.random.normal(value.shape, key=keys[i])) for i, (path, value) in enumerate(tree_flatten(vae.parameters()))]  # fmt: off
    return tree_unflatten(flat)


def _loaded(weights: dict) -> LoadedWeights:
    return LoadedWeights(components={"vae": weights}, meta_data=MetaData())


def _bare(cls, small: bool):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.compute_precision = ComputePrecision(None)
    model.small_decoder = small
    return model


@pytest.mark.fast
def test_the_default_vae_keeps_the_full_decoder():
    # Flux2VAE is shared with FLUX 3, Lens, Ideogram 4 and ERNIE-Image: its default must not change.
    vae = Flux2VAE()
    assert vae.decoder.conv_in.weight.shape == (512, 3, 3, 32)
    assert vae.decoder.conv_out.weight.shape == (3, 3, 3, 128)


@pytest.mark.fast
def test_the_small_decoder_has_the_parameter_count_of_the_reference():
    # diffusers AutoencoderKLFlux2.from_pretrained(SMALL_DECODER_REPO).decoder has 27,946,179 parameters.
    vae = Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS)
    count = sum(value.size for _, value in tree_flatten(vae.decoder.parameters()))
    assert count == 27_946_179
    assert vae.decoder.conv_in.weight.shape == (384, 3, 3, 32)
    assert vae.decoder.conv_norm_out.weight.shape == (96,)
    # The encoder is the one of every FLUX.2 VAE, so image editing keeps working.
    assert all(a.shape == b.shape for (_, a), (_, b) in zip(tree_flatten(vae.encoder.parameters()), tree_flatten(Flux2VAE().encoder.parameters())))  # fmt: off


@pytest.mark.fast
def test_the_loader_reads_only_the_diffusers_file():
    # The repo root also holds small_decoder.safetensors (BFL key names) and
    # full_encoder_small_decoder.safetensors. Reading every *.safetensors would mix them in.
    component = Flux2SmallDecoderLoader.component()
    assert component.name == "vae"
    assert component.hf_subdir == ""
    assert component.weight_files == [SMALL_DECODER_FILE]


@pytest.mark.fast
@pytest.mark.parametrize("bits", [None, 8])
def test_only_the_decoding_parts_of_the_checkpoint_are_applied(monkeypatch, bits):
    # The checkpoint holds a whole VAE; its encoder, quant_conv and bn must not replace the model's own.
    supplied = _weights_of(Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS), seed=1)
    calls = []

    def load_single(component, repo_id, file_pattern):
        calls.append((component.name, repo_id, file_pattern))
        return _loaded(supplied)

    monkeypatch.setattr(WeightLoader, "load_single", staticmethod(load_single))
    model_vae = Flux2VAE()
    model_vae.update(_weights_of(model_vae, seed=2))
    vae = Flux2SmallDecoderLoader.with_small_decoder(model_vae, bits=bits)

    assert calls == [("vae", SMALL_DECODER_REPO, SMALL_DECODER_FILE)]
    assert np.array_equal(np.array(vae.post_quant_conv.weight), np.array(supplied["post_quant_conv"]["weight"]))
    assert np.array_equal(np.array(vae.decoder.conv_out.weight), np.array(supplied["decoder"]["conv_out"]["weight"]))
    # Edit references are encoded, and latents normalized, by the model's own VAE.
    assert vae.encoder is model_vae.encoder
    assert vae.quant_conv is model_vae.quant_conv
    assert vae.bn is model_vae.bn
    # -q quantizes the attention of the small decoder as it does for the full one.
    to_q = vae.decoder.mid_block.attentions[0].to_q
    assert isinstance(to_q, nn.QuantizedLinear) == (bits is not None)


@pytest.mark.fast
def test_full_width_weights_do_not_load_into_the_small_decoder(monkeypatch):
    # Module.update(strict=False) would take them without a word.
    monkeypatch.setattr(WeightLoader, "load_single", staticmethod(lambda *a, **k: _loaded(_weights_of(Flux2VAE()))))
    with pytest.raises(ValueError, match="Wrong shape"):
        Flux2SmallDecoderLoader.with_small_decoder(Flux2VAE())


@pytest.mark.fast
def test_a_missing_or_unexpected_key_stops_the_load():
    parameters = Flux2SmallDecoderLoader._decoding_parameters(
        Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS)
    )
    flat = [(p, mx.zeros(v.shape)) for p, v in tree_flatten(parameters)]
    without = tree_unflatten([(p, v) for p, v in flat if p != "post_quant_conv.weight"])
    with pytest.raises(ValueError, match=r"Missing: \['post_quant_conv.weight'\]"):
        Flux2SmallDecoderLoader.check_weights(parameters, without)
    extra = tree_unflatten(flat + [("decoder.extra.weight", mx.zeros((1,)))])
    with pytest.raises(ValueError, match=r"Unexpected: \['decoder.extra.weight'\]"):
        Flux2SmallDecoderLoader.check_weights(parameters, extra)
    Flux2SmallDecoderLoader.check_weights(parameters, tree_unflatten(flat))


@pytest.mark.fast
@pytest.mark.parametrize("small", [False, True])
def test_the_initializer_swaps_only_the_decoder(monkeypatch, small):
    full_vae = Flux2VAE()
    small_vae = Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS)
    calls = []

    def init_models(model):
        model.vae = full_vae
        model.transformer = None
        model.text_encoder = None

    def apply_weights(model, weights, quantize):
        model.bits = 4

    def with_small_decoder(vae, bits=None):
        calls.append((vae, bits))
        return small_vae

    monkeypatch.setattr(Flux2Initializer, "_load_weights", staticmethod(lambda path: None))
    monkeypatch.setattr(Flux2Initializer, "_init_tokenizers", staticmethod(lambda model, path: None))
    monkeypatch.setattr(Flux2Initializer, "_init_models", staticmethod(init_models))
    monkeypatch.setattr(Flux2Initializer, "_apply_weights", staticmethod(apply_weights))
    monkeypatch.setattr(Flux2Initializer, "_apply_lora", staticmethod(lambda *a: None))
    monkeypatch.setattr(small_decoder.Flux2SmallDecoderLoader, "with_small_decoder", staticmethod(with_small_decoder))

    model = _bare(Flux2Klein, False)
    Flux2Initializer.init(model=model, model_config=ModelConfig.flux2_klein_4b(), quantize=4, small_decoder=small)

    assert model.small_decoder is small
    assert model.vae is (small_vae if small else full_vae)
    assert calls == ([(full_vae, 4)] if small else [])


@pytest.mark.fast
@pytest.mark.parametrize("cls", [Flux2Klein, Flux2KleinEdit])
def test_the_metadata_records_the_option_only_when_it_is_used(cls):
    assert "small_decoder" not in _bare(cls, False)._generation_parameters()
    assert _bare(cls, True)._generation_parameters()["small_decoder"] is True


@pytest.mark.fast
def test_a_model_with_the_small_decoder_is_not_saved(tmp_path):
    with pytest.raises(ValueError, match="small_decoder"):
        _bare(Flux2Klein, True).save_model(str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.fast
def test_pid_decode_and_the_small_decoder_do_not_mix():
    with pytest.raises(ValueError, match="pid_decode"):
        _bare(Flux2Klein, True).generate_image(seed=1, prompt="x", pid_decode=True)


def _sidecar(tmp_path, **values) -> str:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"prompt": "x", "model": "flux2-klein-4b", "seed": 1, "steps": 4, **values}))
    return str(path)


@pytest.mark.fast
@pytest.mark.parametrize(
    "recorded, extra, expected",
    [
        (True, [], True),
        (True, ["--no-small-decoder"], False),
        (False, ["--small-decoder"], True),
        (None, [], False),
        ("null", [], False),
    ],
)
def test_config_from_metadata_replays_the_flag(monkeypatch, tmp_path, recorded, extra, expected):
    values = {} if recorded is None else {"small_decoder": None if recorded == "null" else recorded}
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", _sidecar(tmp_path, **values)] + extra)
    assert flux2_generate.build_parser().parse_args().small_decoder is expected


@pytest.mark.fast
@pytest.mark.parametrize("recorded", ["false", "true", 1])
def test_a_recorded_value_that_is_not_a_boolean_stops_the_replay(monkeypatch, tmp_path, capsys, recorded):
    # bool("false") is True: a hand-edited sidecar must not switch the decoder.
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", _sidecar(tmp_path, small_decoder=recorded)])
    with pytest.raises(SystemExit) as exit_info:
        flux2_generate.build_parser().parse_args()
    assert exit_info.value.code == 2
    assert "small_decoder in --config-from-metadata must be true or false" in capsys.readouterr().err


@pytest.mark.fast
def test_the_cli_rejects_the_flag_with_pid_decode(monkeypatch, capsys):
    monkeypatch.setattr(flux2_generate, "Flux2Klein", lambda **kwargs: pytest.fail("the model was loaded"))
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--small-decoder", "--pid-decode"])
    with pytest.raises(SystemExit) as exit_info:
        flux2_generate.main()
    assert exit_info.value.code == 2
    assert "--small-decoder cannot be used with --pid-decode" in capsys.readouterr().err


@pytest.mark.fast
@pytest.mark.parametrize(
    "module, cls_name, extra",
    [(flux2_generate, "Flux2Klein", []), (flux2_edit_generate, "Flux2KleinEdit", ["--image-paths", "in.png"])],
)
@pytest.mark.parametrize("flag", [[], ["--small-decoder"]])
def test_the_cli_passes_the_flag_to_the_model(monkeypatch, module, cls_name, extra, flag):
    received = {}

    class ReachedModelLoad(Exception):
        pass

    def fake_model(**kwargs):
        received.update(kwargs)
        raise ReachedModelLoad

    monkeypatch.setattr(module, cls_name, fake_model)
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--model", "flux2-klein-9b"] + extra + flag)
    with pytest.raises(ReachedModelLoad):
        module.main()
    assert received["small_decoder"] is bool(flag)
