import math

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.transformer.ada_layer_norm_continuous import (
    AdaLayerNormContinuous,
)
from mflux.models.transformer.embed_nd import EmbedND
from mflux.models.transformer.joint_transformer_block import (
    JointTransformerBlock,
)
from mflux.models.transformer.single_transformer_block import (
    SingleTransformerBlock,
)
from mflux.models.transformer.time_text_embed import TimeTextEmbed


class Transformer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(64, 3072)
        self.time_text_embed = TimeTextEmbed(model_config=model_config)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(19)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(38)]
        self.norm_out = AdaLayerNormContinuous(3072, 3072)
        self.proj_out = nn.Linear(3072, 64)

        self.enable_teacache = True
        self.cnt = 0
        self.rel_l1_thresh = (
            0.6  # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
        )
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None

    def predict(
        self,
        t: int,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        hidden_states: mx.array,
        config: RuntimeConfig,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
    ) -> mx.array:
        time_step = config.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        hidden_states = self.x_embedder(hidden_states)
        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        text_embeddings = self.time_text_embed(time_step, pooled_prompt_embeds, guidance)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        txt_ids = Transformer.prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = Transformer.prepare_latent_image_ids(config.height, config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed(ids)

        # =================
        # fmt: off
        self.num_steps = config.num_inference_steps
        inp = mx.array(hidden_states)
        temb_ = mx.array(text_embeddings)
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, text_embeddings=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
        # fmt: off
        # =================

        if not should_calc:
            hidden_states += self.previous_residual
        else:
            hidden_states = self.regular_forward_pass(
                controlnet_block_samples,
                controlnet_single_block_samples,
                encoder_hidden_states,
                hidden_states,
                image_rotary_emb,
                text_embeddings
            )

        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        noise = hidden_states
        return noise

    def regular_forward_pass(
        self,
        controlnet_block_samples,
        controlnet_single_block_samples,
        encoder_hidden_states,
        hidden_states,
        image_rotary_emb,
        text_embeddings,
    ):
        ori_hidden_states = mx.array(hidden_states)

        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            if controlnet_block_samples is not None and len(controlnet_block_samples) > 0:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(math.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[idx // interval_control]

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            if controlnet_single_block_samples is not None and len(controlnet_single_block_samples) > 0:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(math.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[idx // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        self.previous_residual = hidden_states - ori_hidden_states
        return hidden_states

    @staticmethod
    def prepare_latent_image_ids(height: int, width: int) -> mx.array:
        latent_width = width // 16
        latent_height = height // 16
        latent_image_ids = mx.zeros((latent_height, latent_width, 3))
        latent_image_ids = latent_image_ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
        latent_image_ids = latent_image_ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
        latent_image_ids = mx.repeat(latent_image_ids[None, :], 1, axis=0)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_width * latent_height, 3))
        return latent_image_ids

    @staticmethod
    def prepare_text_ids(seq_len: mx.array) -> mx.array:
        return mx.zeros((1, seq_len, 3))
