from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizer

from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput


@runtime_checkable
class Tokenizer(Protocol):
    tokenizer: PreTrainedTokenizer

    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput: ...


class BaseTokenizer(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @abstractmethod
    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput: ...


class LanguageTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        return_attention_mask: bool = True,
        template: str | None = None,
        use_chat_template: bool = False,
        chat_template_kwargs: dict | None = None,
        add_special_tokens: bool = True,
    ):
        super().__init__(tokenizer, max_length)
        self.padding = padding
        self.return_attention_mask = return_attention_mask
        self.template = template
        self.use_chat_template = use_chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.add_special_tokens = add_special_tokens

    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput:
        max_length = max_length or self.max_length

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        prompts = [p if p is not None else "" for p in prompts]
        if all(p == "" for p in prompts):
            batch_size = len(prompts)
            input_ids = mx.array(np.empty((batch_size, 0), dtype=np.int32))
            attention_mask = mx.array(np.empty((batch_size, 0), dtype=np.int32))
            return TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)

        if self.template or self.use_chat_template:
            formatted_prompts = []
            for p in prompts:
                if self.template:
                    formatted = self.template.format(p)
                elif self.use_chat_template:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                        **self.chat_template_kwargs,
                    )
                else:
                    formatted = p
                formatted_prompts.append(formatted)
            prompts = formatted_prompts

        tokens = self.tokenizer(
            prompts,
            padding=self.padding,
            max_length=max_length,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="np",
        )

        input_ids = mx.array(tokens["input_ids"])
        if self.return_attention_mask:
            attention_mask = mx.array(tokens["attention_mask"])
        else:
            attention_mask = mx.ones_like(input_ids)

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class VisionLanguageTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor,
        max_length: int = 1024,
        template: str | None = None,
        image_token: str = "<|image_pad|>",
    ):
        super().__init__(tokenizer, max_length)
        self.processor = processor
        self.template = template
        self.image_token = image_token

    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput:
        max_length = max_length or self.max_length

        if isinstance(prompt, str):
            prompt = [prompt]

        if self.template and images:
            img_prompt = ""
            for i in range(len(images)):
                img_prompt += f"Picture {i + 1}: <|vision_start|>{self.image_token}<|vision_end|>"
            formatted_text = self.template.format(img_prompt + prompt[0])
        elif self.template:
            formatted_text = self.template.format(prompt[0])
        else:
            formatted_text = prompt[0]

        pixel_values = None
        image_grid_thw = None

        if images:
            model_inputs = self.processor(
                text=[formatted_text],
                images=images,
                padding=True,
                return_tensors=None,
            )

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            pixel_values = mx.array(model_inputs["pixel_values"])
            image_grid_thw = mx.array(model_inputs["image_grid_thw"])
        else:
            tokens = self.tokenizer(
                [formatted_text],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            input_ids = mx.array(tokens["input_ids"])
            attention_mask = mx.array(tokens["attention_mask"])

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
