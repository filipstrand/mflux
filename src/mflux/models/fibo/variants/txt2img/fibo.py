import sys

import mlx.core as mx
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux_debugger.tensor_debug import debug_load


class FIBO(nn.Module):
    vae: VAE
    transformer: FiboTransformer

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        super().__init__()
        self.bits = quantize
        self.local_path = local_path

        # Load weights and initialize model components
        FIBOInitializer.init(
            fibo_model=self,
            quantize=quantize,
            local_path=local_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: RuntimeConfig,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        # Run transformer with PyTorch-saved inputs (for debugging)
        # 1 st timestep only is fine to check!

        hidden_states = debug_load("pt_hidden_states")
        timestep = debug_load("pt_timestep")
        encoder_hidden_states = debug_load("pt_encoder_hidden_states")
        stacked_prompt_layers = debug_load("pt_prompt_layers")
        text_ids = debug_load("pt_text_ids")
        latent_image_ids = debug_load("pt_latent_image_ids")
        attention_mask = debug_load("pt_attention_mask")
        prompt_layers = [stacked_prompt_layers[i] for i in range(stacked_prompt_layers.shape[0])]

        self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_encoder_layers=prompt_layers,
            timestep=timestep,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            attention_mask=attention_mask,
        )
        sys.exit(0)

        # We don't really care about the generation just now!
        decoded = mx.zeros((1, 3, config.height, config.width), dtype=mx.float32)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=None,
            lora_scales=None,
            image_path=None,
            image_strength=None,
            generation_time=0.0,
        )
