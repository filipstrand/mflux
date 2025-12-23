import random

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.flux.variants.dreambooth.dataset.batch import Batch, Example
from mflux.models.flux.variants.txt2img.flux import Flux1


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux: Flux1, config: Config, batch: Batch) -> mx.float16:
        losses = [
            DreamBoothLoss._single_example_loss(flux, config, example, batch.rng)
            for example in batch.examples
        ]  # fmt: off
        return mx.mean(mx.array(losses))

    @staticmethod
    def _single_example_loss(flux: Flux1, config: Config, example: Example, rng: random.Random) -> mx.float16:
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
            dtype=ModelConfig.precision,
            key=mx.random.key(noise_seed),
        )

        # By linear interpolation between the clean image and pure noise, construct a latent at time t
        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=config.scheduler.sigmas[t],
        )

        # Predict the noise from timestep t
        predicted_noise = flux.transformer(
            t=t,
            config=config,
            hidden_states=latents_t,
            prompt_embeds=example.prompt_embeds,
            pooled_prompt_embeds=example.pooled_prompt_embeds,
        )

        # Construct the loss (derivation in src/mflux/models/flux/variants/dreambooth/optimization/_loss_derivation)
        return (clean_image + predicted_noise - pure_noise).square().mean()
