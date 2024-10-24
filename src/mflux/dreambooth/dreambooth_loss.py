import mlx.core as mx

from mflux import Config, Flux1
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.finetuning_dataset import Batch, Example


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux: Flux1, config: RuntimeConfig, batch: Batch) -> mx.float16:
        loss = 0
        for example in batch.examples:
            single_example_loss = DreamBoothLoss._per_example_loss(flux, config, example)
            loss += single_example_loss
        return loss / len(batch.examples)

    @staticmethod
    def _per_example_loss(flux: Flux1, config: RuntimeConfig, example: Example) -> mx.float16:
        # draw a random timestep t from [0, num_inference_steps]
        t = int(
            mx.random.randint(
                low=0,
                high=config.num_inference_steps,
                shape=[],
            )
        )  # fmt: off

        # generate pure noise
        pure_noise = mx.random.normal(
            shape=example.encoded_image_latents.shape,
            dtype=Config.precision,
        )  # fmt: off

        # Via linear interpolation, produce two latent arrays at time t and t+1
        latents_t = (1 - config.sigmas[t]) * example.encoded_image_latents + config.sigmas[t] * pure_noise
        latents_t_1 = (1 - config.sigmas[t + 1]) * example.encoded_image_latents + config.sigmas[t + 1] * pure_noise

        # Predict the noise from timestep t
        noise = flux.transformer.predict(
            t=t,
            prompt_embeds=example.prompt_embeds,
            pooled_prompt_embeds=example.pooled_prompt_embeds,
            hidden_states=latents_t,
            config=config,
        )

        # Take one denoise step to arrive at the predicted latent[t+1]
        dt = config.sigmas[t + 1] - config.sigmas[t]
        latents_t_1_predicted = latents_t + noise * dt

        # The loss is the squared difference between our 'reference' latents_t_1 and the predicted one
        return (latents_t_1_predicted - latents_t_1).square().mean()
