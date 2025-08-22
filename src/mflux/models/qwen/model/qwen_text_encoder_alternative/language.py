from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

import numpy as np

from .config import ModelConfig, TextConfig
class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False
    
class QuantizedKVCache(_BaseCache):
    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.step, self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.step, self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

class KVCache(_BaseCache):
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache



def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out

def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if isinstance(cache, QuantizedKVCache):
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
    
def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    lengths: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds <= rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask & (rinds < lengths)
    return mask

def create_attention_mask(
    h: mx.array, cache: Optional[Any] = None, return_array: bool = False
):
    T = h.shape[1]
    if T > 1:
        offset = 0
        window_size = None
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.offset
            if hasattr(c, "max_size"):
                window_size = c.max_size
                offset = min(window_size, offset)
                return_array = return_array or offset + T > window_size
        if return_array:
            return create_causal_mask(T, offset, window_size=window_size)
        else:
            return "causal"
    else:
        mask = None
    return mask






class Qwen2RotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = mx.arange(self.max_seq_len_cached).astype(mx.float32)

        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self, x, seq_len=None):

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return (
            self.cos_cached[:seq_len].astype(x.dtype),
            self.sin_cached[:seq_len].astype(x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    Args:
        q (mx.array): The query tensor.
        k (mx.array): The key tensor.
        cos (mx.array): The cosine part of the rotary embedding.
        sin (mx.array): The sine part of the rotary embedding.
        mrope_section (List[int]): Multimodal rope section for channel dimension of temporal, height and width.
        unsqueeze_dim (int, optional): Dimension to unsqueeze. Defaults to 1.
    Returns:
        tuple(mx.array): The rotated query and key tensors.
    """

    mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
    cos = cos[position_ids]
    sin = sin[position_ids]

    cos = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))], axis=-1
    )[
        :, None, :, :
    ]  # unsqueeze dim 1
    sin = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(sin, mrope_section, axis=-1))], axis=-1
    )[:, None, :, :]

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope_scaling = args.rope_scaling

        self.rotary_emb = Qwen2RotaryEmbedding(
            head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            kv_seq_len += cache.offset + 1
            position_ids = mx.arange(cache.offset, cache.offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        cos, sin = self.rotary_emb(values, kv_seq_len)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]
        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, position_ids, self.rope_scaling["mrope_section"]
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen2Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen2VLDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen2Model(args)
        self.rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        # Calculate RoPE index for image/video tokens
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(
                    attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids)
                )
                image_nums, video_nums = 0, 0
                vision_start_indices = mx.sum(
                    mx.where(
                        input_ids == vision_start_token_id,
                        mx.arange(input_ids.shape[0]),
                        mx.zeros_like(input_ids),
                    )
                )
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(
                        llm_grid_t, 1
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )  # Equivalent to expand()
                    t_index = t_index.flatten()  # Flattens to 1D

                    h_index = mx.arange(llm_grid_h).reshape(
                        1, llm_grid_h, 1
                    )  # Equivalent to .view(1, -1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    h_index = h_index.flatten()  # Flattens to 1D

                    w_index = mx.arange(llm_grid_w).reshape(
                        1, 1, llm_grid_w
                    )  # Equivalent to .view(1, -1)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    w_index = w_index.flatten()  # Flattens to 1D

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(
                        1, text_len
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (3, text_len)
                    )  # Equivalent to expand(3, -1)

                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(
                    -1, keepdims=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        # reset rope_deltas when processing a new image/video
        if pixel_values is not None:
            self.rope_deltas = None

        if position_ids is None and (mask is None or mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache is not None and cache[0] is not None and cache[0].offset == 0)
                or self.rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self.rope_deltas = rope_deltas
            else:
                # Use the prev pre-calculated rope-deltas to get the correct position ids
                batch_size, seq_length = inputs.shape
                delta = cache[-1].offset + self.rope_deltas if cache is not None else 0
                delta = delta[None][None]
                position_ids = mx.arange(seq_length).reshape(1, seq_length)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                if cache is not None:
                    # Repeat delta for each batch
                    delta = mx.repeat(delta, batch_size // delta.shape[0], axis=0)
                position_ids = mx.add(position_ids, delta).reshape(position_ids.shape)
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs, cache=cache, inputs_embeds=inputs_embeds, position_ids=position_ids
        )
        
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
