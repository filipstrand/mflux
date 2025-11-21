import textwrap
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.fibo_vlm.fibo_vlm_initializer import FIBOVLMInitializer
from mflux.models.fibo_vlm.model.qwen3_vl_util import Qwen3VLUtil


class FiboVLM:
    def __init__(
        self,
        model_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ):
        FIBOVLMInitializer.init(
            vlm_model=self,
            model_id=model_id,
            local_path=local_path,
        )

    def generate(
        self,
        task: str,
        *,
        prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        refine_image: Optional[Image.Image] = None,
        structured_prompt: Optional[str] = None,
        editing_instructions: Optional[str] = None,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        # Build messages from task and prompt
        messages = FiboVLM._build_messages(
            task=task,
            image=image,
            refine_image=refine_image,
            prompt=prompt,
            structured_prompt=structured_prompt,
            editing_instructions=editing_instructions,
        )

        # Check if images are present in messages (for processor return_tensors selection)
        has_images = image is not None or refine_image is not None

        # Match diffusers approach: use tokenizer.apply_chat_template with tokenize=False,
        # then call processor directly with text and images separately
        # This ensures consistent image preprocessing between MLX and PyTorch
        if has_images:
            # Extract images from messages (matching diffusers _collect_images)
            images_list = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            img = item.get("image")
                            if img is not None:
                                images_list.append(img)

            # Get text from chat template (without tokenization)
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Call processor directly with text and images
            processor_inputs = {
                "text": [prompt_text],
                "padding": True,
                "return_tensors": "np",  # Return numpy arrays (will convert to MLX)
            }
            if images_list:
                processor_inputs["images"] = images_list

            formatted = self.processor(**processor_inputs)
        else:
            # Text-only: use apply_chat_template directly
            formatted = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="np",
                return_dict=True,
            )

        # Extract input_ids and attention_mask (before MLX conversion)
        # BatchFeature is a UserDict, so it supports .get() but isn't a dict instance
        input_ids_np = None
        attention_mask_np = None
        pixel_values_pt = None
        image_grid_thw_pt = None

        if hasattr(formatted, "get"):
            # BatchFeature or dict-like object
            input_ids_np = formatted.get("input_ids")
            attention_mask_np = formatted.get("attention_mask")
            pixel_values_pt = formatted.get("pixel_values")
            image_grid_thw_pt = formatted.get("image_grid_thw")
            # If attention_mask is None, try accessing the data dict directly (BatchFeature stores data in .data)
            if attention_mask_np is None and hasattr(formatted, "data"):
                attention_mask_np = formatted.data.get("attention_mask") if hasattr(formatted.data, "get") else None
        elif isinstance(formatted, dict):
            input_ids_np = formatted.get("input_ids")
            attention_mask_np = formatted.get("attention_mask")
            pixel_values_pt = formatted.get("pixel_values")
            image_grid_thw_pt = formatted.get("image_grid_thw")
        elif hasattr(formatted, "input_ids"):
            # Access via attributes
            input_ids_np = formatted.input_ids
            attention_mask_np = getattr(formatted, "attention_mask", None)
            pixel_values_pt = getattr(formatted, "pixel_values", None)
            image_grid_thw_pt = getattr(formatted, "image_grid_thw", None)
        else:
            # If processor returns a list or tensor directly
            input_ids_np = formatted
            attention_mask_np = None

        # Convert to numpy if needed (handle both PyTorch tensors and numpy arrays)
        if input_ids_np is not None and hasattr(input_ids_np, "numpy"):
            input_ids_np = input_ids_np.numpy()
        elif isinstance(input_ids_np, np.ndarray):
            pass  # Already numpy
        if attention_mask_np is not None and hasattr(attention_mask_np, "numpy"):
            attention_mask_np = attention_mask_np.numpy()
        elif isinstance(attention_mask_np, np.ndarray):
            pass  # Already numpy

        if pixel_values_pt is not None:
            if hasattr(pixel_values_pt, "numpy"):
                pixel_values_np = pixel_values_pt.numpy()
            elif isinstance(pixel_values_pt, np.ndarray):
                pixel_values_np = pixel_values_pt
            else:
                pixel_values_np = None
        else:
            pixel_values_np = None

        if image_grid_thw_pt is not None:
            if hasattr(image_grid_thw_pt, "numpy"):
                image_grid_thw_np = image_grid_thw_pt.numpy()
            elif isinstance(image_grid_thw_pt, np.ndarray):
                image_grid_thw_np = image_grid_thw_pt
            else:
                image_grid_thw_np = None
        else:
            image_grid_thw_np = None

        # Convert to MLX arrays
        input_ids = mx.array(input_ids_np)
        attention_mask = mx.array(attention_mask_np) if attention_mask_np is not None else None
        pixel_values = mx.array(pixel_values_np) if pixel_values_np is not None else None
        image_grid_thw = mx.array(image_grid_thw_np) if image_grid_thw_np is not None else None

        # Convert stop strings to token ID sequences (preserve multi-token sequences)
        stop_token_sequences = []
        if stop:
            for stop_str in stop:
                stop_sequence = self.processor.tokenizer.encode(stop_str, add_special_tokens=False)
                stop_token_sequences.append(stop_sequence)

        # Generate text
        generated_ids = Qwen3VLUtil.generate_text(
            decoder=self.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stop_token_sequences=stop_token_sequences if stop_token_sequences else None,
            eos_token_id=self.eos_token_id,
            seed=seed,
        )

        input_len = input_ids.shape[1]
        generated_tokens = generated_ids[:, input_len:]
        generated_tokens_np = np.array(generated_tokens[0])  # Take first batch item

        # Decode using tokenizer (match diffusers: skip_special_tokens=True)
        generated_text = self.processor.tokenizer.decode(generated_tokens_np, skip_special_tokens=True)

        return generated_text

    @staticmethod
    def _build_messages(
        task: str,
        *,
        image: Optional[Image.Image] = None,
        refine_image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        structured_prompt: Optional[str] = None,
        editing_instructions: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = []

        if task == "inspire":
            user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": "<inspire>"})
        elif task == "generate":
            text_value = (prompt or "").strip()
            formatted = f"<generate>\n{text_value}"
            user_content.append({"type": "text", "text": formatted})
        else:  # refine
            if refine_image is None:
                base_prompt = (structured_prompt or "").strip()
                edits = (editing_instructions or "").strip()
                formatted = textwrap.dedent(
                    f"""<refine>
Input:
{base_prompt}
Editing instructions:
{edits}"""
                ).strip()
                user_content.append({"type": "text", "text": formatted})
            else:
                user_content.append({"type": "image", "image": refine_image})
                edits = (editing_instructions or "").strip()
                formatted = textwrap.dedent(
                    f"""<refine>
Editing instructions:
{edits}"""
                ).strip()
                user_content.append({"type": "text", "text": formatted})

        messages = [{"role": "user", "content": user_content}]
        return messages
