from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor


class Qwen2VLImageProcessor(QwenImageProcessor):
    def __init__(self):
        super().__init__(
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
