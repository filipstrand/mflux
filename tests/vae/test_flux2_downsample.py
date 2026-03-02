import mlx.core as mx

from mflux.models.flux2.model.flux2_vae.common.downsample_2d import Flux2Downsample2D


def test_flux2_downsample_preserves_even_half_dimensions():
    downsample = Flux2Downsample2D(channels=8, padding=0)
    hidden_states = mx.zeros((1, 8, 80, 80), dtype=mx.float32)
    downsampled = downsample(hidden_states)

    assert downsampled.shape == (1, 8, 40, 40)
