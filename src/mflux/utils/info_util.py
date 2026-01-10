from datetime import datetime
from pathlib import Path


class InfoUtil:
    @staticmethod
    def format_metadata(metadata: dict) -> str:
        exif = metadata.get("exif", {})
        if not exif:
            return "No metadata found"

        lines = ["=" * 60, "MFLUX Image Information", "=" * 60]

        def _has_any(keys: list[str]) -> bool:
            return any(exif.get(k) is not None for k in keys)

        # Prompt
        if prompt := exif.get("prompt"):
            lines.append(f"\nPrompt: {prompt}")

        if negative_prompt := exif.get("negative_prompt"):
            lines.append(f"Negative Prompt: {negative_prompt}")

        # Model information
        lines.append("")
        if model := exif.get("model"):
            if (orig_model := exif.get("original_model")) and orig_model != model:
                lines.append(f"Model: {model} (Original: {orig_model})")
            else:
                lines.append(f"Model: {model}")

        # Image dimensions
        if width := exif.get("width"):
            if (orig_w := exif.get("original_width")) and orig_w != width:
                lines.append(f"Width: {width} (Original: {orig_w})")
            else:
                lines.append(f"Width: {width}")
        if height := exif.get("height"):
            if (orig_h := exif.get("original_height")) and orig_h != height:
                lines.append(f"Height: {height} (Original: {orig_h})")
            else:
                lines.append(f"Height: {height}")

        # Generation parameters
        lines.append("")
        if seed := exif.get("seed"):
            if (orig_seed := exif.get("original_seed")) and orig_seed != seed:
                lines.append(f"Seed: {seed} (Original: {orig_seed})")
            else:
                lines.append(f"Seed: {seed}")
        if steps := exif.get("steps"):
            if (orig_steps := exif.get("original_steps")) and orig_steps != steps:
                lines.append(f"Steps: {steps} (Original: {orig_steps})")
            else:
                lines.append(f"Steps: {steps}")
        if guidance := exif.get("guidance"):
            if (orig_guidance := exif.get("original_guidance")) and orig_guidance != guidance:
                lines.append(f"Guidance: {guidance} (Original: {orig_guidance})")
            else:
                lines.append(f"Guidance: {guidance}")

        # Technical settings
        if quantize := exif.get("quantize"):
            lines.append(f"Quantization: {quantize}-bit")
        if precision := exif.get("precision"):
            lines.append(f"Precision: {precision}")

        # Explicit "Original generation" section for derived images (upscales, edits, etc.)
        original_keys = [
            "original_model",
            "original_width",
            "original_height",
            "original_seed",
            "original_steps",
            "original_guidance",
            "original_quantize",
            "original_lora_paths",
            "original_lora_scales",
        ]
        if _has_any(original_keys):
            lines.append("")
            lines.append("Original Generation:")

            if orig_model := exif.get("original_model"):
                lines.append(f"  - Model: {orig_model}")
            if exif.get("original_width") is not None and exif.get("original_height") is not None:
                lines.append(f"  - Size: {exif.get('original_width')}x{exif.get('original_height')}")
            if orig_seed := exif.get("original_seed"):
                lines.append(f"  - Seed: {orig_seed}")
            if orig_steps := exif.get("original_steps"):
                lines.append(f"  - Steps: {orig_steps}")
            if (orig_guidance := exif.get("original_guidance")) is not None:
                lines.append(f"  - Guidance: {orig_guidance}")
            if (orig_quantize := exif.get("original_quantize")) is not None:
                lines.append(f"  - Quantization: {orig_quantize}-bit")

            if orig_lora_paths := exif.get("original_lora_paths"):
                lines.append(f"  - LoRAs ({len(orig_lora_paths)}):")
                orig_lora_scales = exif.get("original_lora_scales") or []
                for i, lora in enumerate(orig_lora_paths):
                    scale = orig_lora_scales[i] if i < len(orig_lora_scales) else 1.0
                    lora_name = Path(lora).name
                    lines.append(f"    - {lora_name} (scale: {scale})")

        # LoRA information
        if lora_paths := exif.get("lora_paths"):
            lines.append("")
            lines.append(f"LoRAs ({len(lora_paths)}):")
            lora_scales = exif.get("lora_scales") or []
            for i, lora in enumerate(lora_paths):
                scale = lora_scales[i] if i < len(lora_scales) else 1.0
                lora_name = Path(lora).name
                lines.append(f"  - {lora_name} (scale: {scale})")

        # Image-to-image parameters
        if image_path := exif.get("image_path"):
            lines.append("")
            lines.append(f"Source Image: {Path(image_path).name}")
            if image_strength := exif.get("image_strength"):
                lines.append(f"Image Strength: {image_strength}")

        # ControlNet parameters
        if controlnet_path := exif.get("controlnet_image_path"):
            lines.append("")
            lines.append(f"ControlNet Image: {Path(controlnet_path).name}")
            if controlnet_strength := exif.get("controlnet_strength"):
                lines.append(f"ControlNet Strength: {controlnet_strength}")

        # Generation metadata
        lines.append("")
        if gen_time := exif.get("generation_time_seconds"):
            lines.append(f"Generation Time: {gen_time:.2f}s")

        if created_at := exif.get("created_at"):
            try:
                dt = datetime.fromisoformat(created_at)
                lines.append(f"Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except (ValueError, AttributeError):
                lines.append(f"Created: {created_at}")

        if version := exif.get("mflux_version"):
            lines.append(f"MFLUX Version: {version}")

        lines.append("=" * 60)
        return "\n".join(lines)
