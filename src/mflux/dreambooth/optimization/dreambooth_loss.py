import random

import mlx.core as mx

from mflux import Config, Flux1
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.dataset.batch import Batch
from mflux.dreambooth.dataset.dataset import Example
from mflux.dreambooth.optimization.depth_guided_loss import DepthGuidedLoss
from mflux.latent_creator.latent_creator import LatentCreator


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux: Flux1, config: RuntimeConfig, batch: Batch) -> mx.float16:
        losses = [
            DreamBoothLoss._single_example_loss(flux, config, example, batch.rng)
            for example in batch.examples
        ]  # fmt: off
        mean_loss = mx.mean(mx.array(losses))
        return mean_loss

    @staticmethod
    def _single_example_loss(flux: Flux1, config: RuntimeConfig, example: Example, rng: random.Random) -> mx.float16:
        # Must be a better way to handle the randomness than this, but we already
        # save/restore the random state via the iterator so this is a continent shortcut.
        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        # Draw a random timestep t from [0, num_inference_steps]
        t = int(
            mx.random.randint(
                low=0,
                high=config.num_inference_steps,
                shape=[],
                key=mx.random.key(time_seed),
            ),
        )

        # Get the clean image latent
        clean_image = example.clean_latents

        # Generate pure noise
        pure_noise = mx.random.normal(
            shape=clean_image.shape,
            dtype=Config.precision,
            key=mx.random.key(noise_seed),
        )

        # By linear interpolation between the clean image and pure noise, construct a latent at time t
        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=config.sigmas[t],
        )

        # Predict the noise from timestep t
        predicted_noise = flux.transformer(
            t=t,
            config=config,
            hidden_states=latents_t,
            prompt_embeds=example.prompt_embeds,
            pooled_prompt_embeds=example.pooled_prompt_embeds,
        )

        # Construct the loss (derivation in src/mflux/dreambooth/optimization/_loss_derivation)
        pixel_losses = (clean_image + predicted_noise - pure_noise).square()

        # Use depth-guided loss if available
        has_depth = example.raw_depth_map is not None

        if has_depth:
            depth_guided_loss = DepthGuidedLoss()
            # Get the configuration from the example (defaults provided if not available)
            emphasis_mode = getattr(example, "depth_emphasis_mode", "foreground")
            emphasis_strength = getattr(example, "depth_emphasis_strength", 2.0)

            return depth_guided_loss.apply_depth_emphasis_to_loss(
                loss_tensor=pixel_losses,
                raw_depth_map=example.raw_depth_map,
                emphasis_mode=emphasis_mode,
                emphasis_strength=emphasis_strength,
            )
        else:
            return pixel_losses.mean()
