import mlx.core as mx
import numpy as np

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder


class Qwen3VLUtil:
    @staticmethod
    def sample_top_p(
        logits: mx.array,
        top_p: float,
        temperature: float = 1.0,
    ) -> mx.array:
        if temperature != 1.0:
            logits = logits / temperature
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        probs_np = np.array(probs)
        sorted_indices_np = np.argsort(probs_np)[::-1]
        sorted_probs_np = probs_np[sorted_indices_np]
        cumulative_probs_np = np.cumsum(sorted_probs_np)
        sorted_indices_to_remove_np = cumulative_probs_np > top_p
        sorted_indices_to_remove_np[0] = False
        indices_to_remove_np = sorted_indices_np[sorted_indices_to_remove_np]
        probs_np[indices_to_remove_np] = 0.0
        probs_np = probs_np / np.sum(probs_np)
        token_id = np.random.choice(len(probs_np), p=probs_np)
        return mx.array(token_id, dtype=mx.int32)

    @staticmethod
    def generate_text(
        decoder: Qwen3VLDecoder,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        max_new_tokens: int = 4096,
        top_p: float = 0.9,
        temperature: float = 0.2,
        stop_token_sequences: list[list[int]] | None = None,
        eos_token_id: int | None = None,
        seed: int | None = None,
    ) -> mx.array:
        if stop_token_sequences is None:
            stop_token_sequences = []
        if eos_token_id is not None:
            stop_token_sequences.append([eos_token_id])

        batch_size = input_ids.shape[0]

        # Avoid unnecessary conversion if already mx.array
        if isinstance(input_ids, mx.array):
            generated_ids = input_ids
        else:
            generated_ids = mx.array(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids).astype(mx.int32)

        # Set random seed for deterministic generation if provided
        if seed is not None:
            np.random.seed(seed)

        # Pre-compute max stop sequence length (constant, don't recompute every iteration)
        max_stop_len = 0
        if stop_token_sequences:
            max_stop_len = max(len(seq) for seq in stop_token_sequences)

        # Pre-allocate reusable arrays for efficiency
        ones_1x1 = mx.ones((batch_size, 1), dtype=mx.int32)

        # Initialize KV cache
        past_key_values = None
        iteration_count = 0

        for iteration in range(max_new_tokens):
            # On first iteration: process full sequence
            # On subsequent iterations: only process new token (use cache)
            if past_key_values is None:
                # First iteration: process full input sequence
                decoder_input_ids = generated_ids
                decoder_attention_mask = attention_mask
            else:
                # Subsequent iterations: only process the new token
                decoder_input_ids = generated_ids[:, -1:]  # Only last token
                decoder_attention_mask = ones_1x1  # Reuse pre-allocated array

            # Forward pass with KV cache
            # Only pass pixel_values and image_grid_thw on first iteration (they're part of the input sequence)
            decoder_kwargs = {
                "input_ids": decoder_input_ids,
                "attention_mask": decoder_attention_mask,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            if past_key_values is None:
                # First iteration: include image data if present
                if pixel_values is not None and image_grid_thw is not None:
                    decoder_kwargs["pixel_values"] = pixel_values
                    decoder_kwargs["image_grid_thw"] = image_grid_thw

            decoder_output = decoder(**decoder_kwargs)

            if isinstance(decoder_output, tuple):
                logits, past_key_values = decoder_output
                # Evaluate KV cache to ensure it's computed and not accumulating
                if past_key_values is not None:
                    for kv_tuple in past_key_values:
                        mx.eval(kv_tuple[0], kv_tuple[1])
            else:
                logits = decoder_output
                past_key_values = None

            # Evaluate logits before sampling
            mx.eval(logits)

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            next_tokens = []
            for i in range(batch_size):
                token_id = Qwen3VLUtil.sample_top_p(next_token_logits[i], top_p, temperature)
                next_tokens.append(token_id)

            next_tokens = mx.stack(next_tokens, axis=0)
            next_tokens = mx.expand_dims(next_tokens, axis=1)

            generated_ids = mx.concatenate([generated_ids, next_tokens], axis=1)
            attention_mask = mx.concatenate([attention_mask, ones_1x1], axis=1)  # Reuse pre-allocated array
            iteration_count += 1

            # Check for stop token sequences (only check last max_stop_len tokens for efficiency)
            should_stop = False
            matched_stop_sequence = None
            if stop_token_sequences and max_stop_len > 0:
                # Only convert the last max_stop_len tokens to numpy (much more efficient)
                if generated_ids.shape[1] >= max_stop_len:
                    last_tokens_mx = generated_ids[0, -max_stop_len:]
                    last_tokens_np = np.array(last_tokens_mx)
                    for stop_sequence in stop_token_sequences:
                        seq_len = len(stop_sequence)
                        if len(last_tokens_np) >= seq_len:
                            # Check if the last seq_len tokens match this stop sequence
                            tokens_to_check = last_tokens_np[-seq_len:].tolist()
                            if tokens_to_check == stop_sequence:
                                should_stop = True
                                matched_stop_sequence = stop_sequence
                                break

            if should_stop:
                # Exclude the stop sequence from the output (match PyTorch behavior)
                if matched_stop_sequence is not None:
                    seq_len = len(matched_stop_sequence)
                    generated_ids = generated_ids[:, :-seq_len]
                break

        return generated_ids
