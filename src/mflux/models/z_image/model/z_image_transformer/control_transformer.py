import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_transformer.control_transformer_block import ZImageControlTransformerBlock
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock


class ZImageControlTransformer(ZImageTransformer):
    """Z-Image S3-DiT transformer with a VACE-style ControlNet branch (sc-2257).

    Strict pose conditioning for Z-Image-Turbo via
    ``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`` (Apache-2.0), ported
    from VideoX-Fun ``ZImageControlTransformer2DModel`` (v2.1 config:
    ``add_control_noise_refiner=True`` / ``add_control_noise_refiner_correctly=True``).

    On top of the base transformer this adds:
      - ``control_all_x_embedder``: a 33ch->dim patch embedder for the VAE-encoded
        control context (control latent 16ch + mask 1ch + inpaint latent 16ch).
      - ``control_noise_refiner`` (2 blocks): a parallel control refiner whose hints
        are injected into the base ``noise_refiner`` (image-length stage).
      - ``control_layers`` (15 blocks at base places 0,2,...,28): the main control
        stack whose hints are injected into the matching base ``layers`` (unified
        image+caption stage).

    The control branch reuses the base image position ids / RoPE / padding (the
    control context shares the image's spatial dims), so no separate alignment is
    needed. When ``control_context`` is ``None`` the forward is bit-identical to the
    base transformer; with ``control_context_scale == 0`` the hints contribute zero,
    so it also reproduces the base output exactly (used as the parity gate).
    """

    CONTROL_IN_DIM = 33
    CONTROL_LAYERS_PLACES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    CONTROL_REFINER_PLACES = [0, 1]

    def __init__(
        self,
        patch_size: int = 2,
        f_patch_size: int = 1,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: list[int] | None = None,
        axes_lens: list[int] | None = None,
    ):
        super().__init__(
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

        key = f"{patch_size}-{f_patch_size}"
        control_embed_dim = f_patch_size * patch_size * patch_size * ZImageControlTransformer.CONTROL_IN_DIM
        self.control_all_x_embedder = {key: nn.Linear(control_embed_dim, dim, bias=True)}

        self.control_layers_mapping = {place: n for n, place in enumerate(ZImageControlTransformer.CONTROL_LAYERS_PLACES)}  # fmt: off
        self.control_refiner_mapping = {place: n for n, place in enumerate(ZImageControlTransformer.CONTROL_REFINER_PLACES)}  # fmt: off

        self.control_layers = [
            ZImageControlTransformerBlock(dim, n_heads, norm_eps, qk_norm, has_before_proj=(i == 0))
            for i in range(len(ZImageControlTransformer.CONTROL_LAYERS_PLACES))
        ]
        self.control_noise_refiner = [
            ZImageControlTransformerBlock(dim, n_heads, norm_eps, qk_norm, has_before_proj=(i == 0))
            for i in range(len(ZImageControlTransformer.CONTROL_REFINER_PLACES))
        ]

    def __call__(
        self,
        x: mx.array,
        timestep: mx.array | float | int,
        sigmas: mx.array,
        cap_feats: mx.array,
        control_context: mx.array | None = None,
        control_context_scale: float = 1.0,
    ) -> mx.array:
        key = f"{self.patch_size}-{self.f_patch_size}"

        # Time embedding
        if not isinstance(timestep, mx.array):
            if isinstance(timestep, int):
                sigma_t = sigmas[timestep].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
            else:
                timestep = mx.array(timestep, dtype=mx.float32)
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,))
        t_emb = self.t_embedder(timestep.astype(mx.float32) * self.t_scale)

        # Patchify image and caption
        x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
            image=x,
            cap_feats=cap_feats,
            patch_size=self.patch_size,
            f_patch_size=self.f_patch_size,
        )

        # Image embedding
        x_emb = self.all_x_embedder[key](x_emb)
        x_emb = mx.where(x_pad_mask[:, None], self.x_pad_token, x_emb)
        x_freqs_cis = self.rope_embedder(x_pos_ids)
        x_attn_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
        x_emb = mx.expand_dims(x_emb, axis=0)

        # Control refiner pass: build the control context embedding (reusing the
        # image's pad mask / RoPE), run the parallel control refiner, and collect
        # the per-block hints + the threaded control state for the main stack.
        refiner_hints = None
        threaded_control = None
        if control_context is not None:
            c_tokens = ZImageControlTransformer._patchify_control(control_context, self.patch_size, self.f_patch_size)
            c_emb = self.control_all_x_embedder[key](c_tokens)
            c_emb = mx.where(x_pad_mask[:, None], self.x_pad_token, c_emb)
            c_emb = mx.expand_dims(c_emb, axis=0)
            refiner_hints, threaded_control = ZImageControlTransformer._run_control_blocks(
                blocks=self.control_noise_refiner,
                c=c_emb,
                x_base=x_emb,
                attn_mask=x_attn_mask,
                freqs_cis=x_freqs_cis,
                t_emb=t_emb,
            )

        # Noise refiner (with control hints when present)
        for i, layer in enumerate(self.noise_refiner):
            x_emb = layer(x=x_emb, attn_mask=x_attn_mask, freqs_cis=x_freqs_cis, t_emb=t_emb)
            if refiner_hints is not None and i in self.control_refiner_mapping:
                x_emb = x_emb + refiner_hints[self.control_refiner_mapping[i]] * control_context_scale

        # Caption embedding
        cap_emb = self.cap_embedder[1](self.cap_embedder[0](cap_emb))
        cap_emb = mx.where(cap_pad_mask[:, None], self.cap_pad_token, cap_emb)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)
        cap_attn_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
        cap_emb = mx.expand_dims(cap_emb, axis=0)

        # Context refiner
        for layer in self.context_refiner:
            cap_emb = layer(x=cap_emb, attn_mask=cap_attn_mask, freqs_cis=cap_freqs_cis)

        # Unify image + caption
        x_len = x_emb.shape[1]
        unified = mx.concatenate([x_emb, cap_emb], axis=1)
        unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
        unified_attn_mask = mx.ones((1, unified.shape[1]), dtype=mx.bool_)

        # Main control pass: thread the (refined) control state + caption through
        # the 15 control layers to produce the hints for the unified main loop.
        main_hints = None
        if control_context is not None:
            control_unified = mx.concatenate([threaded_control, cap_emb], axis=1)
            main_hints, _ = ZImageControlTransformer._run_control_blocks(
                blocks=self.control_layers,
                c=control_unified,
                x_base=unified,
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_cis,
                t_emb=t_emb,
            )

        # Main layers (with control hints when present)
        for i, layer in enumerate(self.layers):
            unified = layer(x=unified, attn_mask=unified_attn_mask, freqs_cis=unified_freqs_cis, t_emb=t_emb)
            if main_hints is not None and i in self.control_layers_mapping:
                unified = unified + main_hints[self.control_layers_mapping[i]] * control_context_scale

        # Final layer and unpatchify
        unified = self.all_final_layer[key](unified, t_emb)
        output = ZImageTransformer._unpatchify(
            x=unified[0, :x_len],
            size=x_size,
            patch_size=self.patch_size,
            f_patch_size=self.f_patch_size,
            out_channels=self.out_channels,
        )
        return -output

    @staticmethod
    def _run_control_blocks(
        blocks: list[ZImageControlTransformerBlock],
        c: mx.array,
        x_base: mx.array,
        attn_mask: mx.array,
        freqs_cis: mx.array,
        t_emb: mx.array,
    ) -> tuple[list[mx.array], mx.array]:
        """Run a parallel control stack, returning (hints, threaded_control).

        Mirrors the VACE stack threading: block 0 seeds the control branch via
        ``before_proj(c) + x_base``; each block runs the base transformer forward
        and emits ``after_proj(c)`` as its hint, passing the running control state
        ``c`` to the next block.
        """
        hints: list[mx.array] = []
        for i, block in enumerate(blocks):
            if i == 0:
                c = block.before_proj(c) + x_base
            c = ZImageTransformerBlock.__call__(block, c, attn_mask, freqs_cis, t_emb)
            hints.append(block.after_proj(c))
        return hints, c

    @staticmethod
    def _patchify_control(control_context: mx.array, patch_size: int, f_patch_size: int) -> mx.array:
        """Patchify the (C=33, F, H, W) control context into (seq, patch*patch*33)
        tokens, padded to a multiple of 32 exactly like the base image patchify so
        the control sequence aligns 1:1 with the image tokens (shared RoPE / mask).
        """
        pH = pW = patch_size
        pF = f_patch_size
        C, F, H, W = control_context.shape
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

        c = control_context.reshape(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        c = mx.transpose(c, axes=(1, 3, 5, 2, 4, 6, 0))
        c = c.reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

        ori_len = c.shape[0]
        padding_len = (-ori_len) % 32
        if padding_len > 0:
            c = mx.concatenate([c, mx.repeat(c[-1:], padding_len, axis=0)], axis=0)
        return c
