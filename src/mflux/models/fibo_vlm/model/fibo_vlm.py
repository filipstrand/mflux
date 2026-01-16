import json
import textwrap
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil
from mflux.models.fibo_vlm.fibo_vlm_initializer import FiboVLMInitializer
from mflux.models.fibo_vlm.tokenizer.qwen2vl_processor import Qwen2VLProcessor


class FiboVLM:
    decoder = Qwen3VLDecoder

    def __init__(
        self,
        model_id: str = "briaai/FIBO-vlm",
        model_path: str | None = None,
        quantize: int | None = None,
    ):
        FiboVLMInitializer.init(
            model=self,
            model_path=model_path if model_path else model_id,
            quantize=quantize,
        )

    @property
    def processor(self) -> Qwen2VLProcessor:
        return self.tokenizers["fibo_vlm"].processor

    def generate(
        self,
        prompt: str,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="generate",
            prompt=prompt,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def refine(
        self,
        structured_prompt: str,
        editing_instructions: str,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="refine",
            structured_prompt=structured_prompt,
            editing_instructions=editing_instructions,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def inspire(
        self,
        image: Image.Image,
        prompt: str | None = None,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="inspire",
            image=image,
            prompt=prompt,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def _generate_internal(
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
        stop = stop or ["<|im_end|>", "<|end_of_text|>"]
        messages = FiboVLM._build_messages(
            task=task,
            image=image,
            prompt=prompt,
            refine_image=refine_image,
            structured_prompt=FiboVLM._normalize_json(structured_prompt),
            editing_instructions=editing_instructions,
        )
        formatted = FiboVLM._process_messages(self.processor, messages, image, refine_image)
        input_ids = mx.array(formatted["input_ids"])
        attention_mask = mx.array(formatted["attention_mask"]) if formatted.get("attention_mask") is not None else None
        pixel_values = mx.array(formatted["pixel_values"]) if formatted.get("pixel_values") is not None else None
        image_grid_thw = mx.array(formatted["image_grid_thw"]) if formatted.get("image_grid_thw") is not None else None
        stop_token_sequences = [self.processor.tokenizer.encode(s, add_special_tokens=False) for s in stop]
        generated_ids = Qwen3VLUtil.generate_text(
            seed=seed,
            top_p=top_p,
            decoder=self.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_token_sequences=stop_token_sequences,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_tokens = generated_ids[:, input_ids.shape[1] :]
        generated_text = self.processor.tokenizer.decode(np.array(generated_tokens[0]), skip_special_tokens=True)
        return FiboVLM._format_json_output(generated_text)

    @staticmethod
    def _normalize_json(structured_prompt: Optional[str]) -> Optional[str]:
        if structured_prompt is None:
            return None
        try:
            return json.dumps(json.loads(structured_prompt.strip()), separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            return structured_prompt

    @staticmethod
    def _process_messages(
        processor: Qwen2VLProcessor,
        messages: List[Dict[str, Any]],
        image: Optional[Image.Image],
        refine_image: Optional[Image.Image],
    ) -> Dict[str, Any]:
        has_images = image is not None or refine_image is not None

        if has_images:
            images_list = FiboVLM._extract_images(messages)
            prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = {"text": [prompt_text], "padding": True, "return_tensors": "np"}
            if images_list:
                inputs["images"] = images_list
            return processor(**inputs)

        return processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="np", return_dict=True
        )

    @staticmethod
    def _extract_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        img = item.get("image")
                        if img is not None:
                            images.append(img)
        return images

    @staticmethod
    def _format_json_output(text: str) -> str:
        try:
            return f"\n{json.dumps(json.loads(text), indent=2, ensure_ascii=False)}\n"
        except json.JSONDecodeError:
            return text

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
        if task == "inspire":
            inspire_text = "<inspire>"
            if prompt:
                inspire_text = f"<inspire>\n{(prompt or '').strip()}"
            content = [{"type": "image", "image": image}, {"type": "text", "text": inspire_text}]
        elif task == "generate":
            content = [{"type": "text", "text": f"<generate>\n{(prompt or '').strip()}"}]
        else:
            content = FiboVLM._build_refine_content(refine_image, structured_prompt, editing_instructions)

        return [{"role": "user", "content": content}]

    @staticmethod
    def _build_refine_content(
        refine_image: Optional[Image.Image],
        structured_prompt: Optional[str],
        editing_instructions: Optional[str],
    ) -> List[Dict[str, Any]]:
        edits = (editing_instructions or "").strip()

        if refine_image is None:
            base_prompt = (structured_prompt or "").strip()
            text = textwrap.dedent(
                f"""<refine>
Input:
{base_prompt}
Editing instructions:
{edits}"""
            ).strip()
            return [{"type": "text", "text": text}]

        text = textwrap.dedent(
            f"""<refine>
Editing instructions:
{edits}"""
        ).strip()
        return [{"type": "image", "image": refine_image}, {"type": "text", "text": text}]
