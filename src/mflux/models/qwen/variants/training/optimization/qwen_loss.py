"""
Qwen-Image Training Loss Functions.

Implements flow matching loss for Qwen-Image diffusion model training.
Supports both single-example and batched computation.

Flow Matching Loss Derivation:
- At timestep t, latent is interpolated: z_t = (1 - sigma_t) * clean + sigma_t * noise
- Transformer predicts velocity: v = d(z_t)/dt
- For linear interpolation: v_target = noise - clean
- Loss: ||v_predicted - v_target||^2

Equivalent formulation (used here):
- Loss = (clean + predicted_noise - pure_noise).square().mean()
"""

import random
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator


@dataclass
class QwenTrainingExample:
    """Single training example for Qwen-Image training."""

    prompt: str
    image_path: str
    clean_latents: mx.array  # VAE-encoded image
    prompt_embeds: mx.array  # Text encoder output
    prompt_mask: mx.array  # Attention mask for text


@dataclass
class QwenTrainingBatch:
    """Batch of training examples."""

    examples: list[QwenTrainingExample]
    rng: random.Random

    @property
    def batch_size(self) -> int:
        return len(self.examples)

    def get_stacked_clean_latents(self) -> mx.array:
        """Stack clean latents into batch."""
        return mx.stack([ex.clean_latents for ex in self.examples], axis=0)

    def get_stacked_prompt_embeds(self) -> mx.array:
        """Stack prompt embeddings into batch (with padding)."""
        max_seq_len = max(ex.prompt_embeds.shape[-2] for ex in self.examples)
        padded = []
        for ex in self.examples:
            embed = ex.prompt_embeds
            # Remove batch dim if present (shape should be [seq, hidden] or [1, seq, hidden])
            if embed.ndim == 3 and embed.shape[0] == 1:
                embed = embed.squeeze(0)
            if embed.shape[0] < max_seq_len:
                pad_len = max_seq_len - embed.shape[0]
                embed = mx.pad(embed, [(0, pad_len), (0, 0)])
            padded.append(embed)
        # Use stack to create new batch dimension: (batch, seq, hidden)
        return mx.stack(padded, axis=0)

    def get_stacked_prompt_masks(self) -> mx.array:
        """Stack prompt masks into batch (with padding)."""
        max_seq_len = max(ex.prompt_mask.shape[-1] for ex in self.examples)
        padded = []
        for ex in self.examples:
            mask = ex.prompt_mask
            # Remove batch dim if present (shape should be [seq] or [1, seq])
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if mask.shape[0] < max_seq_len:
                pad_len = max_seq_len - mask.shape[0]
                mask = mx.pad(mask, [(0, pad_len)])
            padded.append(mask)
        # Use stack to create new batch dimension: (batch, seq)
        return mx.stack(padded, axis=0)


class QwenLoss:
    """
    Loss computation for Qwen-Image training.

    Implements flow matching loss using the equivalent formulation:
    loss = (clean_latents + predicted_noise - pure_noise).square().mean()
    """

    @staticmethod
    def compute_loss(
        qwen: Any,  # QwenImage model
        config: Config,
        batch: QwenTrainingBatch,
    ) -> mx.array:
        """
        Compute training loss for a batch using vectorized operations.

        This is more efficient than computing losses for each example
        individually and averaging.

        Args:
            qwen: QwenImage model
            config: Training configuration
            batch: Batch of training examples

        Returns:
            Scalar loss value
        """
        batch_size = batch.batch_size

        # Generate random timesteps for each example in batch
        time_seed = batch.rng.randint(0, 2**32 - 1)
        timesteps = mx.random.randint(
            low=0,
            high=config.num_inference_steps,
            shape=(batch_size,),
            key=mx.random.key(time_seed),
        )
        # Ensure timesteps are int32 for indexing
        timesteps = timesteps.astype(mx.int32)

        # Stack clean latents: (batch, seq, hidden)
        clean_latents = batch.get_stacked_clean_latents()

        # Generate pure noise for each example
        noise_seed = batch.rng.randint(0, 2**32 - 1)
        pure_noise = mx.random.normal(
            shape=clean_latents.shape,
            dtype=ModelConfig.precision,
            key=mx.random.key(noise_seed),
        )

        # Stack embeddings and masks: (batch, seq, hidden) and (batch, seq)
        prompt_embeds = batch.get_stacked_prompt_embeds()
        prompt_masks = batch.get_stacked_prompt_masks()

        # Create noisy latents at each timestep
        # For batched operation, we need to handle per-example sigma
        # Convert sigmas list to array for efficient indexing
        sigmas_array = mx.array(config.scheduler.sigmas)
        sigmas = sigmas_array[timesteps]
        sigmas = sigmas.reshape(-1, 1, 1)  # Broadcast over seq and hidden dims

        latents_t = QwenLoss._add_noise_batch(clean_latents, pure_noise, sigmas)

        # Predict noise for all examples in batch
        # Note: Qwen transformer handles batched input
        predicted_noise = qwen.transformer(
            t=timesteps,  # Batched timesteps
            config=config,
            hidden_states=latents_t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_masks,
        )

        # Flow matching loss: (clean + predicted - noise)^2
        residual = clean_latents + predicted_noise - pure_noise
        loss = residual.square().mean()

        return loss

    @staticmethod
    def compute_loss_single(
        qwen: Any,  # QwenImage model
        config: Config,
        example: QwenTrainingExample,
        rng: random.Random,
    ) -> mx.array:
        """
        Compute training loss for a single example.

        Useful for debugging or when batch size must be 1.

        Args:
            qwen: QwenImage model
            config: Training configuration
            example: Single training example
            rng: Random number generator

        Returns:
            Scalar loss value
        """
        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        # Random timestep
        t = int(
            mx.random.randint(
                low=0,
                high=config.num_inference_steps,
                shape=[],
                key=mx.random.key(time_seed),
            )
        )

        # Pure noise
        pure_noise = mx.random.normal(
            shape=example.clean_latents.shape,
            dtype=ModelConfig.precision,
            key=mx.random.key(noise_seed),
        )

        # Noisy latent at timestep t
        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=example.clean_latents,
            noise=pure_noise,
            sigma=config.scheduler.sigmas[t],
        )

        # Predict noise
        predicted_noise = qwen.transformer(
            t=t,
            config=config,
            hidden_states=latents_t,
            encoder_hidden_states=example.prompt_embeds,
            encoder_hidden_states_mask=example.prompt_mask,
        )

        # Flow matching loss
        return (example.clean_latents + predicted_noise - pure_noise).square().mean()

    @staticmethod
    def _add_noise_batch(
        clean: mx.array,
        noise: mx.array,
        sigmas: mx.array,
    ) -> mx.array:
        """
        Add noise to clean latents using linear interpolation (batched).

        z_t = (1 - sigma) * clean + sigma * noise

        Args:
            clean: Clean latents (batch, seq, hidden)
            noise: Pure noise (batch, seq, hidden)
            sigmas: Sigma values (batch, 1, 1)

        Returns:
            Noisy latents at timestep t
        """
        return (1.0 - sigmas) * clean + sigmas * noise


class QwenLossWithRegularization(QwenLoss):
    """
    Qwen loss with optional regularization terms.

    Adds:
    - LoRA weight regularization (prevents rank collapse)
    - Prior preservation loss (optional, for DreamBooth-style training)
    """

    @staticmethod
    def compute_loss_with_regularization(
        qwen: Any,
        config: Config,
        batch: QwenTrainingBatch,
        lora_weight_decay: float = 0.0,
        prior_batch: QwenTrainingBatch | None = None,
        prior_weight: float = 1.0,
    ) -> mx.array:
        """
        Compute loss with optional regularization.

        Args:
            qwen: QwenImage model
            config: Training configuration
            batch: Training batch
            lora_weight_decay: L2 regularization on LoRA weights
            prior_batch: Optional prior preservation batch
            prior_weight: Weight for prior preservation loss

        Returns:
            Total loss (base + regularization)
        """
        # Base loss
        loss = QwenLoss.compute_loss(qwen, config, batch)

        # LoRA weight regularization
        if lora_weight_decay > 0:
            lora_reg = QwenLossWithRegularization._compute_lora_regularization(qwen)
            loss = loss + lora_weight_decay * lora_reg

        # Prior preservation loss (DreamBooth-style)
        if prior_batch is not None:
            prior_loss = QwenLoss.compute_loss(qwen, config, prior_batch)
            loss = loss + prior_weight * prior_loss

        return loss

    @staticmethod
    def _compute_lora_regularization(qwen: Any) -> mx.array:
        """
        Compute L2 regularization on LoRA weights.

        Helps prevent rank collapse and keeps LoRA updates bounded.

        Note: We collect all LoRA parameters first, then compute the
        sum in a single vectorized operation to avoid lazy graph explosion.
        """
        # Collect all LoRA weight arrays
        lora_weights = []

        params = qwen.parameters()
        for name, param in _flatten_params(params):
            if "lora_A" in name or "lora_B" in name:
                lora_weights.append(param.flatten())

        if not lora_weights:
            return mx.array(0.0)

        # Concatenate all weights and compute L2 norm in single operation
        # This avoids unbounded lazy evaluation from iterative addition
        all_weights = mx.concatenate(lora_weights)
        reg_sum = mx.sum(all_weights**2)

        # Force evaluation to prevent graph explosion
        mx.eval(reg_sum)

        return reg_sum / all_weights.size


def _flatten_params(params: dict, prefix: str = "") -> list[tuple[str, mx.array]]:
    """Flatten nested parameter dictionary."""
    items = []
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_params(value, name))
        elif isinstance(value, mx.array):
            items.append((name, value))
    return items
