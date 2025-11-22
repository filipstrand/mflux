from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor


class Qwen2VLImageProcessor(QwenImageProcessor):
    def __init__(self):
        super().__init__(
            min_pixels=256 * 28 * 28,  # 200704
            max_pixels=1024 * 28 * 28,  # 802816
            patch_size=16,  # Qwen2VL uses 16, Qwen uses 14
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],  # Qwen2VL uses [0.5, 0.5, 0.5], Qwen uses OpenAI CLIP
            image_std=[0.5, 0.5, 0.5],  # Qwen2VL uses [0.5, 0.5, 0.5], Qwen uses OpenAI CLIP
        )
