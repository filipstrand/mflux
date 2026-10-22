import json
import sys
import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mflux.cli.capabilities import describe_command
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3 import flux3_initializer
from mflux.models.flux3.cli import flux3_edit_generate, flux3_generate
from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore

# 8x8 packed latent -> 16x16 VAE latent -> 128x128 image; 256 attention tokens.
PACKED_SHAPE = (1, 128, 8, 8)


def _tiny_vae() -> Flux2VAE:
    mx.random.seed(0)
    vae = Flux2VAE(decoder_block_out_channels=(32, 32, 64, 64))
    mx.eval(vae.decoder.parameters(), vae.post_quant_conv.parameters())
    return vae


def _decode(vae: Flux2VAE, mode: str, packed: mx.array, tiling_config: TilingConfig | None = None) -> np.ndarray:
    vae.decode_mode = mode
    return np.array(vae.decode_packed_latents(packed, tiling_config=tiling_config).astype(mx.float32))


def _packed() -> mx.array:
    mx.random.seed(1)
    return mx.random.normal(PACKED_SHAPE)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.fast
def test_legacy_is_the_default_decode_mode():
    assert Flux2VAE().decode_mode == Flux2VAE.DECODE_LEGACY


@pytest.mark.fast
def test_full_precision_matches_legacy_within_rounding():
    vae, packed = _tiny_vae(), _packed()

    legacy = _decode(vae, Flux2VAE.DECODE_LEGACY, packed)
    full_precision = _decode(vae, Flux2VAE.DECODE_FULL_PRECISION, packed)

    # The two paths run the same maths with different conv and upsample graphs. Different GPU kernels can
    # round the bf16 intermediates in a different way, so the test allows bf16 rounding, not more.
    assert full_precision.shape == legacy.shape == (1, 3, 128, 128)
    assert _relative_error(full_precision, legacy) < 1e-2


@pytest.mark.fast
def test_full_precision_with_attention_chunks_matches_legacy_within_rounding(monkeypatch):
    vae, packed = _tiny_vae(), _packed()
    legacy = _decode(vae, Flux2VAE.DECODE_LEGACY, packed)

    # 64-query chunks split the 256-token attention into 4 calls.
    monkeypatch.setattr(VAEBlockDecoder, "ATTENTION_CHUNK", 64)
    full_precision = _decode(vae, Flux2VAE.DECODE_FULL_PRECISION, packed)

    assert _relative_error(full_precision, legacy) < 1e-2


@pytest.mark.fast
def test_low_memory_strips_and_chunks_match_one_strip_and_one_chunk(monkeypatch):
    vae, packed = _tiny_vae(), _packed()
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 10_000)
    one_strip = _decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed)

    # 16-row strips split every layer from 32 rows up, and 64-query chunks split the 256-token attention.
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 16)
    monkeypatch.setattr(VAEBlockDecoder, "ATTENTION_CHUNK", 64)
    strips = _decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed)

    assert _relative_error(strips, one_strip) < 2e-2


@pytest.mark.fast
def test_low_memory_stays_within_bf16_rounding_of_legacy(monkeypatch):
    vae, packed = _tiny_vae(), _packed()
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 16)

    legacy = _decode(vae, Flux2VAE.DECODE_LEGACY, packed)
    low_memory = _decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed)

    assert _relative_error(low_memory, legacy) < 5e-2


@pytest.mark.fast
def test_low_memory_strips_have_no_seams(monkeypatch):
    # A wrong halo row shows as a larger error on the rows next to a strip edge: about 1.7x without the
    # halo, about 1.0x with it.
    vae, packed = _tiny_vae(), _packed()
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 10_000)
    one_strip = _decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed)
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 16)
    error_by_row = np.abs(_decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed) - one_strip).mean(axis=(0, 1, 3))

    edge_rows = [row for row in range(128) if row % 16 in (0, 15)]
    inner_rows = [row for row in range(128) if row % 16 not in (0, 15)]
    assert error_by_row[edge_rows].mean() < 1.3 * error_by_row[inner_rows].mean()


@pytest.mark.fast
def test_every_decoder_group_norm_uses_the_grouping_of_the_strip_statistics():
    norms = [module for _, module in Flux2VAE().decoder.named_modules() if isinstance(module, nn.GroupNorm)]

    assert norms
    assert all(norm.pytorch_compatible for norm in norms)


@pytest.mark.fast
@pytest.mark.parametrize(
    "conv",
    [
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),
        nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=1),
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, dilation=2),
    ],
)
def test_block_decode_rejects_a_conv_that_is_not_3x3_stride_1_padding_1(conv):
    with pytest.raises(ValueError, match="3x3 convs"):
        VAEBlockDecoder._conv3x3_rows(conv, lambda a, b: mx.zeros((1, b - a, 8, 4)), 0, 8, 8)


@pytest.mark.fast
def test_low_memory_rejects_a_group_norm_with_a_different_grouping():
    vae, packed = _tiny_vae(), _packed()
    vae.decoder.mid_block.resnets[0].norm1.pytorch_compatible = False

    with pytest.raises(ValueError, match="pytorch_compatible"):
        _decode(vae, Flux2VAE.DECODE_LOW_MEMORY, packed)


@pytest.mark.fast
@pytest.mark.parametrize("mode", [Flux2VAE.DECODE_LOW_MEMORY, Flux2VAE.DECODE_FULL_PRECISION])
def test_block_decode_modes_ignore_tiling_config(monkeypatch, mode):
    vae, packed = _tiny_vae(), _packed()
    untiled = _decode(vae, mode, packed)

    tiled = _decode(vae, mode, packed, tiling_config=TilingConfig(vae_decode_tile_size=128))

    assert np.array_equal(tiled, untiled)


@pytest.mark.fast
def test_flux3_uses_the_low_memory_decode(monkeypatch):
    monkeypatch.setattr(flux3_initializer, "Flux3Transformer", lambda **kwargs: None)
    monkeypatch.setattr(flux3_initializer, "Qwen3TextEncoder", lambda **kwargs: None)
    model = SimpleNamespace(model_config=SimpleNamespace(transformer_overrides={}, text_encoder_overrides={}))

    flux3_initializer.Flux3Initializer._init_models(model)

    assert model.vae.decode_mode == Flux2VAE.DECODE_LOW_MEMORY


class _ModelLoaded(Exception):
    pass


def _run_cli(monkeypatch, tmp_path, cli, extra_args: list[str]) -> tuple[SimpleNamespace, list]:
    reference = tmp_path / "ref.png"
    reference.touch()
    argv = ["prog", "--prompt", "x", *extra_args]
    if cli is flux3_edit_generate:
        argv += ["--image-paths", str(reference)]
    monkeypatch.setattr(sys, "argv", argv)
    model = SimpleNamespace(vae=SimpleNamespace(decode_mode=Flux2VAE.DECODE_LOW_MEMORY))
    monkeypatch.setattr(cli, "Flux3Klein" if cli is flux3_generate else "Flux3KleinEdit", lambda **kwargs: model)

    def stop(**kwargs):
        raise _ModelLoaded

    monkeypatch.setattr(cli.CallbackManager, "register_callbacks", stop)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(_ModelLoaded):
        warnings.simplefilter("always")
        cli.main()
    return model, caught


CLIS = [flux3_generate, flux3_edit_generate]


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
def test_vae_full_precision_flag_selects_the_full_precision_decode(monkeypatch, tmp_path, cli):
    model, _ = _run_cli(monkeypatch, tmp_path, cli, ["--vae-full-precision"])

    assert model.vae.decode_mode == Flux2VAE.DECODE_FULL_PRECISION


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
def test_default_cli_keeps_the_low_memory_decode(monkeypatch, tmp_path, cli):
    model, _ = _run_cli(monkeypatch, tmp_path, cli, [])

    assert model.vae.decode_mode == Flux2VAE.DECODE_LOW_MEMORY


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("flag", [["--vae-tiling"], ["--vae-tile-size", "256"]], ids=lambda flag: flag[0])
def test_vae_tiling_flags_warn_that_the_decode_does_not_tile(monkeypatch, tmp_path, cli, flag):
    _, caught = _run_cli(monkeypatch, tmp_path, cli, flag)

    assert any("row strips" in str(w.message) for w in caught)


@pytest.mark.fast
def test_capabilities_report_vae_tiling_status():
    def status(cli, flag):
        entry = describe_command("cmd", cli.__name__)
        return next(option for option in entry["options"] if option["flag"] == flag)["status"]

    assert status(flux3_generate, "--vae-tiling") == "ignored"
    assert status(flux3_edit_generate, "--vae-tiling") == "conditional"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(Flux2VAE.DECODE_FULL_PRECISION, {"vae_full_precision": True}), (Flux2VAE.DECODE_LOW_MEMORY, {})],
)
def test_metadata_records_the_full_precision_decode_only_when_on(mode, expected):
    model = SimpleNamespace(compute_precision=ComputePrecision(None), vae=SimpleNamespace(decode_mode=mode))

    assert _Flux3KleinCore._generation_parameters(model) == expected


@pytest.mark.fast
@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize(
    ("recorded", "argv_flag", "expected"),
    [
        (True, [], True),
        (True, ["--no-vae-full-precision"], False),
        (False, ["--vae-full-precision"], True),
        (None, [], False),
    ],
)
def test_a_sidecar_replays_the_full_precision_decode(monkeypatch, tmp_path, cli, recorded, argv_flag, expected):
    sidecar = tmp_path / "image.metadata.json"
    recorded_field = {} if recorded is None else {"vae_full_precision": recorded}
    sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4} | recorded_field))
    reference = tmp_path / "ref.png"
    reference.touch()
    extra = ["--image-paths", str(reference)] if cli is flux3_edit_generate else []
    monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar), *extra, *argv_flag])

    assert cli.build_parser().parse_args().vae_full_precision is expected


@pytest.mark.fast
@pytest.mark.parametrize("recorded", ["false", "true", 1])
def test_a_recorded_value_that_is_not_a_boolean_stops_the_replay(monkeypatch, tmp_path, capsys, recorded):
    # bool("false") is True: a hand-edited sidecar must not switch the decode mode.
    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4, "vae_full_precision": recorded}))
    monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar)])

    with pytest.raises(SystemExit) as exit_info:
        flux3_generate.build_parser().parse_args()
    assert exit_info.value.code == 2
    assert "vae_full_precision in --config-from-metadata must be true or false" in capsys.readouterr().err
