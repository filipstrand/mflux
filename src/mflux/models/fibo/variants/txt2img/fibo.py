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
        """
        Minimal FIBO txt2img pipeline in MLX.

        NOTE: For now we rely on PyTorch to provide the text-conditioning tensors
        via debug_save() (pt_encoder_hidden_states, pt_prompt_layers, etc.).
        The denoising loop and VAE decoding run fully in MLX.
        """
        # ---------------------------------------------------------------------
        # 1. Load transformer conditioning from PyTorch debug tensors
        # ---------------------------------------------------------------------
        # These tensors are produced by the PyTorch BriaFiboPipeline during a
        # reference run (see pipeline_bria_fibo.py debug_save() calls).
        encoder_hidden_states = debug_load("pt_encoder_hidden_states")
        stacked_prompt_layers = debug_load("pt_prompt_layers")
        text_ids = debug_load("pt_text_ids")
        latent_image_ids = debug_load("pt_latent_image_ids")
        attention_mask = debug_load("pt_attention_mask")

        # Shape assumptions (matching the PyTorch pipeline):
        # - encoder_hidden_states: (2 * B, seq_ctx, joint_attention_dim)
        # - stacked_prompt_layers: (L, 2 * B, seq_txt, text_encoder_dim)
        # - text_ids: (2 * B, seq_txt, 3)
        # - latent_image_ids: (seq_img, 3)
        # - attention_mask: (2 * B, 1, total_seq, total_seq)
        batch_twice = encoder_hidden_states.shape[0]
        if batch_twice % 2 != 0:
            raise ValueError(
                f"Expected encoder_hidden_states batch dimension to be even (guidance uncond+cond), got {batch_twice}"
            )
        batch_size = batch_twice // 2

        # Convert stacked prompt layers to the list expected by the transformer
        prompt_layers: list[mx.array] = [stacked_prompt_layers[i] for i in range(stacked_prompt_layers.shape[0])]

        # ---------------------------------------------------------------------
        # 2. Prepare scheduler and initial latents (FlowMatch-style)
        # ---------------------------------------------------------------------
        runtime_config = config
        scheduler = runtime_config.scheduler
        num_steps = runtime_config.num_inference_steps
        height = runtime_config.height
        width = runtime_config.width

        # FIBO uses a VAE scale factor of 16 (see BriaFiboPipeline / VAE config)
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        channels = self.transformer.in_channels

        # Create initial Gaussian latents in (B, C, H', W')
        key = mx.random.key(seed)
        latents_4d = mx.random.normal(
            shape=(batch_size, channels, latent_height, latent_width),
            key=key,
        )

        # Pack to (B, seq, C) to match transformer expectations
        latents = mx.transpose(latents_4d, (0, 2, 3, 1))
        latents = mx.reshape(latents, (batch_size, latent_height * latent_width, channels))

        guidance_scale = runtime_config.guidance

        # ---------------------------------------------------------------------
        # 3. Denoising loop in MLX (mirrors PyTorch BriaFiboPipeline)
        # ---------------------------------------------------------------------
        print(f"[FIBO-MLX] Starting denoising loop with {num_steps} steps...", flush=True)
        for step_index in range(num_steps):
            # Scheduler timesteps are pre-computed inside FlowMatchEulerDiscreteScheduler
            timestep_value = scheduler.timesteps[step_index]
            print(f"[FIBO-MLX] Step {step_index + 1}/{num_steps} (t={timestep_value:.4f})", flush=True)

            # Classifier-free guidance: duplicate latents if guidance > 1
            if guidance_scale > 1.0:
                latent_model_input = mx.concatenate([latents, latents], axis=0)
            else:
                latent_model_input = latents

            # Build timestep array matching the latent batch size
            timestep = mx.full(
                (latent_model_input.shape[0],),
                timestep_value,
                dtype=latent_model_input.dtype,
            )

            # One denoising step with the MLX transformer
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                text_encoder_layers=prompt_layers,
                timestep=timestep,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                attention_mask=attention_mask,
            )

            # Perform classifier-free guidance if enabled
            if guidance_scale > 1.0:
                # noise_pred: (2 * B, seq, C)
                half = noise_pred.shape[0] // 2
                noise_uncond = noise_pred[:half]
                noise_text = noise_pred[half:]
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler update: x_t -> x_{t-1}
            latents = scheduler.step(
                model_output=noise_pred,
                timestep=step_index,
                sample=latents,
            )

            # Force computation to keep the loop responsive under MLX lazy eval
            mx.eval(latents)

        # ---------------------------------------------------------------------
        # 4. Unpack latents and decode with the MLX FIBO VAE
        # ---------------------------------------------------------------------

        # LOAD HERE!
        # For cross-framework validation, load the exact VAE input latents from PyTorch.
        # This ensures the VAE decode path matches exactly.
        latents_for_vae = debug_load("vae_input_latents")

        # PyTorch VAE expects (B, C, 1, H, W) - add temporal dimension if needed
        if latents_for_vae.ndim == 4:
            latents_for_vae = mx.reshape(
                latents_for_vae,
                (
                    latents_for_vae.shape[0],
                    latents_for_vae.shape[1],
                    1,
                    latents_for_vae.shape[2],
                    latents_for_vae.shape[3],
                ),
            )

        # Decode using VAE with the exact tensor from diffusers
        decoded = self.vae.decode(latents_for_vae)

        # ---------------------------------------------------------------------
        # 5. Convert decoded latents to a GeneratedImage (same helper as Flux)
        # ---------------------------------------------------------------------
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
