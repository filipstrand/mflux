import mlx.core as mx
import pytest

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil


def get_vae_models():
    from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
    from mflux.models.flux.model.flux_vae.vae import VAE as FluxVAE
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
    from mflux.models.z_image.model.z_image_vae.vae import VAE as ZImageVAE

    return [
        ("Flux", FluxVAE),
        ("Fibo", Wan2_2_VAE),
        ("SeedVR2", SeedVR2VAE),
        ("Qwen", QwenVAE),
        ("Z-Image", ZImageVAE),
    ]


@pytest.mark.slow
@pytest.mark.parametrize("name, vae_class", get_vae_models())
def test_real_vae_tiling_compatibility(name, vae_class):
    # Instantiate VAE
    vae = vae_class()

    # Use a size that will trigger tiling with 512 tile size
    H, W = 1024, 1024
    image = mx.zeros((1, 3, H, W))

    # Enable tiling
    config = TilingConfig(
        vae_encode_tiled=True, vae_encode_tile_size=512, vae_encode_tile_overlap=64, vae_decode_tiles_per_dim=2
    )

    # 1. Test Tiled Encoding
    # This should not crash!
    latent = VAEUtil.encode(vae, image, config)

    # 2. Check latent dimensions
    # VAETiler.encode_image_tiled always returns 5D (B, C, 1, H_lat, W_lat)
    assert latent.ndim == 5
    assert latent.shape[0] == 1
    assert latent.shape[2] == 1

    # 3. Test Tiled Decoding
    # This should not crash!
    decoded = VAEUtil.decode(vae, latent, config)

    # 4. Check decoded dimensions
    # VAEUtil.decode always squashes to 4D for T=1
    assert decoded.ndim == 4
    assert decoded.shape == (1, 3, H, W)


@pytest.mark.slow
@pytest.mark.parametrize("name, vae_class", get_vae_models())
def test_real_vae_standard_compatibility(name, vae_class):
    # Instantiate VAE
    vae = vae_class()

    # Small image for standard (non-tiled) path
    H, W = 256, 256
    image = mx.zeros((1, 3, H, W))

    # Disable tiling
    config = TilingConfig(vae_encode_tiled=False, vae_decode_tiles_per_dim=0)

    # 1. Test Standard Encoding
    latent = VAEUtil.encode(vae, image, config)

    # 2. Test Standard Decoding
    decoded = VAEUtil.decode(vae, latent, config)

    # 3. Check decoded dimensions
    assert decoded.ndim == 4
    assert decoded.shape == (1, 3, H, W)
