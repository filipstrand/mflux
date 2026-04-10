from __future__ import annotations

import gc
import random
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map, tree_unflatten
from PIL import Image as PILImage
from tqdm import tqdm

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.dataset.batch import Batch, DataItem
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.training_state import TrainingState
from mflux.models.common.training.statistics.plotter import Plotter
from mflux.models.common.training.utils import TrainingUtil


class TrainingTrainer:
    @staticmethod
    def compute_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        batch: Batch,
    ) -> mx.float16:
        losses = [
            TrainingTrainer._single_example_loss(adapter, training_spec, base_config, item, batch.rng)
            for item in batch.data
        ]
        return mx.mean(mx.array(losses))

    @staticmethod
    def _single_example_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        item: DataItem,
        rng: random.Random,
    ) -> mx.float16:
        # Create a config matching this item's spatial dimensions.
        # Flux uses config.width/height for rotary embeddings, so this must match the latent layout.
        config = adapter.create_config(training_spec, width=item.width, height=item.height)

        # Reuse the base scheduler only when compatible with the item's dimensions.
        # Some schedulers depend on image seq len when sigma shift is enabled.
        if not config.model_config.requires_sigma_shift or config.image_seq_len == base_config.image_seq_len:
            config._scheduler = base_config.scheduler  # type: ignore[attr-defined]
        else:
            _ = config.scheduler

        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        low = int(training_spec.training_loop.timestep_low)
        high = int(
            config.num_inference_steps
            if training_spec.training_loop.timestep_high is None
            else training_spec.training_loop.timestep_high
        )

        t = int(
            mx.random.randint(
                low=low,
                high=high,
                shape=[],
                key=mx.random.key(time_seed),
            )
        )

        clean_image = item.clean_latents
        pure_noise = mx.random.normal(
            shape=clean_image.shape,
            dtype=config.precision,
            key=mx.random.key(noise_seed),
        )

        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=config.scheduler.sigmas[t],
        )

        predicted_noise = adapter.predict_noise(
            t=t,
            latents_t=latents_t,
            sigmas=config.scheduler.sigmas,
            cond=item.cond,
            config=config,
        )

        error = (clean_image + predicted_noise - pure_noise).square()
        return error.mean()

    @staticmethod
    def train(
        *,
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        first_preview = None
        if training_spec.monitoring is not None and training_spec.monitoring.preview_images:
            first_preview = training_spec.monitoring.preview_images[0]
        preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec, preview_image=first_preview)
        base_config = adapter.create_config(training_spec, width=preview_width, height=preview_height)
        # Ensure scheduler is initialized once and can be reused in per-item configs.
        _ = base_config.scheduler

        # Freeze base weights and unfreeze LoRA weights
        adapter.freeze_base()
        TrainingTrainer._unfreeze_lora_layers(adapter.transformer())

        train_step_function = nn.value_and_grad(
            model=adapter.model(),
            fn=lambda b: TrainingTrainer.compute_loss(adapter, training_spec, base_config, b),
        )

        if training_spec.monitoring is not None and training_state.iterator.num_iterations == 0:
            TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)
            validation_batch = training_state.iterator.get_validation_batch()
            validation_loss = TrainingTrainer.compute_loss(adapter, training_spec, base_config, validation_batch)
            training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=float(validation_loss))  # fmt: off
            Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
            del validation_loss
            training_state.save(adapter, training_spec)

        grad_acc_steps = training_spec.training_loop.gradient_accumulation_steps
        accum_grads = None
        micro_step = 0
        # Accumulate training loss so we can plot it for free (no separate
        # validation forward pass needed — the loss is already computed
        # together with the gradients and was previously discarded).
        train_loss_accum = 0.0
        loss_accum_count = 0

        batches = tqdm(
            training_state.iterator,
            total=training_state.iterator.total_number_of_steps(),
            initial=training_state.iterator.num_iterations,
        )

        for batch in batches:
            loss, grads = train_step_function(batch)
            # Evaluate loss and grads immediately so the backward pass runs
            # and all intermediate activation tensors are freed before we
            # accumulate. Without this, mx.eval(accum_grads) at step 2+ must
            # hold both the previous accumulated grads AND the new backprop
            # activations in memory simultaneously, causing swap at 768px.
            mx.eval(loss, grads)
            train_loss_accum += float(loss)
            loss_accum_count += 1
            del loss
            accum_grads = grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
            del grads
            mx.eval(accum_grads)
            micro_step += 1

            if micro_step == grad_acc_steps:
                if grad_acc_steps > 1:
                    accum_grads = tree_map(lambda g: g / grad_acc_steps, accum_grads)
                training_state.optimizer.optimizer.update(model=adapter.model(), gradients=accum_grads)
                # Only eval the trainable LoRA params and optimizer state —
                # evaluating all model params (including frozen base weights)
                # is unnecessary and adds memory pressure every step.
                mx.eval(TrainingTrainer._get_trainable_params(adapter), training_state.optimizer.optimizer.state)
                del accum_grads
                accum_grads = None
                micro_step = 0
                # Free optimizer/gradient memory immediately after the update so
                # that monitoring (image generation, save) runs with a clean slate.
                if training_spec.low_ram:
                    gc.collect()
                    mx.clear_cache()

            if training_state.should_plot_loss(training_spec):
                avg_loss = train_loss_accum / loss_accum_count if loss_accum_count > 0 else 0.0
                training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=avg_loss)  # fmt: off
                Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
                train_loss_accum = 0.0
                loss_accum_count = 0

            if training_state.should_generate_image(training_spec):
                TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

            if training_state.should_save(training_spec):
                training_state.save(adapter, training_spec)

            if training_spec.low_ram:
                mx.clear_cache()

        training_state.save(adapter, training_spec)

    @staticmethod
    def _get_trainable_params(adapter: TrainingAdapter) -> list:
        """Return only the trainable LoRA parameter arrays (lora_A, lora_B).

        Used to scope mx.eval after optimizer updates — evaluating the full
        model (including frozen base weights) is wasteful and adds pressure.
        """
        params = []
        for _, child in adapter.transformer().named_modules():
            if isinstance(child, LoRALinear):
                if getattr(child, "_mflux_lora_role", None) == "train":
                    params.extend([child.lora_A, child.lora_B])
            elif isinstance(child, FusedLoRALinear):
                for lora in child.loras:
                    if getattr(lora, "_mflux_lora_role", None) == "train":
                        params.extend([lora.lora_A, lora.lora_B])
        return params

    @staticmethod
    def _unfreeze_lora_layers(module: nn.Module) -> None:
        for _, child in module.named_modules():
            if isinstance(child, LoRALinear):
                if getattr(child, "_mflux_lora_role", None) == "train":
                    child.unfreeze(keys=["lora_A", "lora_B"], strict=False)
            elif isinstance(child, FusedLoRALinear):
                for lora in child.loras:
                    if getattr(lora, "_mflux_lora_role", None) == "train":
                        lora.unfreeze(keys=["lora_A", "lora_B"], strict=False)

    @staticmethod
    def _preview_dimensions(training_spec: TrainingSpec, *, preview_image: Path | None = None) -> tuple[int, int]:
        if training_spec.monitoring is None:
            return 1024, 1024
        if preview_image is not None:
            with PILImage.open(preview_image) as img:
                width, height = img.size
        else:
            width = int(training_spec.monitoring.preview_width)
            height = int(training_spec.monitoring.preview_height)

        return TrainingUtil.resolve_dimensions(
            width=width,
            height=height,
            max_resolution=None,
            error_template=(
                f"Preview image too small for training (needs >=16px). Got {{width}}x{{height}} from {preview_image}"
            ),
        )

    @staticmethod
    def _generate_previews(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        if training_spec.monitoring is None:
            return
        preview_prompts = training_spec.monitoring.preview_prompts
        preview_names = training_spec.monitoring.preview_prompt_names
        preview_images = training_spec.monitoring.preview_images
        for idx, prompt in enumerate(preview_prompts):
            image_paths = None
            if training_spec.is_edit:
                if not preview_images or idx >= len(preview_images):
                    raise ValueError("Edit training requires data/preview.* for each preview prompt.")
                image_paths = [preview_images[idx]]
                preview_width, preview_height = TrainingTrainer._preview_dimensions(
                    training_spec, preview_image=preview_images[idx]
                )
            else:
                preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec)
            image = adapter.generate_preview_image(
                seed=training_spec.seed,
                prompt=prompt,
                width=preview_width,
                height=preview_height,
                steps=training_spec.steps,
                image_paths=image_paths,
            )
            preview_name = preview_names[idx] if idx < len(preview_names) else None
            image.save(
                training_state.get_current_preview_image_path(
                    training_spec,
                    preview_index=idx,
                    preview_name=preview_name,
                )
            )
            del image

    @staticmethod
    def _generate_previews_with_optimizer_offload(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        optimizer = training_state.optimizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            offload_path = Path(tmp_dir) / "optimizer_offload.safetensors"
            optimizer.save(offload_path)
            optimizer.optimizer.state = []

            gc.collect()
            mx.clear_cache()
            try:
                TrainingTrainer._generate_previews(adapter, training_spec, training_state)
            finally:
                restored_state = tree_unflatten(list(mx.load(str(offload_path)).items()))
                optimizer.optimizer.state = restored_state
                gc.collect()
                mx.clear_cache()
