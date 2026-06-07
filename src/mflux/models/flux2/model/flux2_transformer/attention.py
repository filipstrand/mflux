import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


class Flux2Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=1e-5)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=1e-5)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_layer_idx: int | None = None,
    ):
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.heads,
            head_dim=self.dim_head,
        )

        enc_query = enc_key = enc_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            enc_query, enc_key, enc_value = AttentionUtils.process_qkv(
                hidden_states=encoder_hidden_states,
                to_q=self.add_q_proj,
                to_k=self.add_k_proj,
                to_v=self.add_v_proj,
                norm_q=self.norm_added_q,
                norm_k=self.norm_added_k,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
            query = mx.concatenate([enc_query, query], axis=2)
            key = mx.concatenate([enc_key, key], axis=2)
            value = mx.concatenate([enc_value, value], axis=2)

        # Apply RoPE to the fresh Q/K. In mflux, the post-concat token layout is
        # `[txt, target, ref]` in extract mode and `[txt, target]` in cached mode.
        # The rotary embedding tensor is sized to match the current input layout,
        # so we slice it accordingly in cached mode (handled by the caller).
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        if kv_cache is not None and kv_cache.mode == "extract":
            # We have the full `[txt, target, ref]` input; the trailing
            # `num_ref_tokens` slice along the sequence dim is the static
            # reference K/V we want to cache.
            ref_count = kv_cache.num_ref_tokens
            if ref_count > 0:
                ref_k = key[:, :, -ref_count:, :]
                ref_v = value[:, :, -ref_count:, :]
                kv_cache.store("double", kv_cache_layer_idx, ref_k, ref_v)

        if kv_cache is not None and kv_cache.mode == "cached":
            # Input is `[txt, target]` (no ref). Splice cached ref K/V at the
            # end so attention sees the full `[txt, target, ref]` K/V layout.
            cached_k, cached_v = kv_cache.load("double", kv_cache_layer_idx)
            key = mx.concatenate([key, cached_k], axis=2)
            value = mx.concatenate([value, cached_v], axis=2)

        if kv_cache is not None and kv_cache.mode == "extract" and kv_cache.num_ref_tokens > 0:
            hidden_states = Flux2KVCache.compute_extract_attention(
                query=query,
                key=key,
                value=value,
                num_ref_tokens=kv_cache.num_ref_tokens,
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
        else:
            hidden_states = AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=hidden_states.shape[0],
                num_heads=self.heads,
                head_dim=self.dim_head,
            )

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out(hidden_states)
        return hidden_states, encoder_hidden_states
