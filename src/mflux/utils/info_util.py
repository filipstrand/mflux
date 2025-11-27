from datetime import datetime
from pathlib import Path


class InfoUtil:
    @staticmethod
    def format_metadata(metadata: dict) -> str:
        exif = metadata.get("exif", {})
        if not exif:
            return "No metadata found"

        lines = ["=" * 60, "MFLUX Image Information", "=" * 60]

        # Prompt
        if prompt := exif.get("prompt"):
            lines.append(f"\nPrompt: {prompt}")

        if negative_prompt := exif.get("negative_prompt"):
            lines.append(f"Negative Prompt: {negative_prompt}")

        # Model information
        lines.append("")
        if model := exif.get("model"):
            lines.append(f"Model: {model}")

        # Image dimensions
        if width := exif.get("width"):
            lines.append(f"Width: {width}")
        if height := exif.get("height"):
            lines.append(f"Height: {height}")

        # Generation parameters
        lines.append("")
        if seed := exif.get("seed"):
            lines.append(f"Seed: {seed}")
        if steps := exif.get("steps"):
            lines.append(f"Steps: {steps}")
        if guidance := exif.get("guidance"):
            lines.append(f"Guidance: {guidance}")

        # Technical settings
        if quantize := exif.get("quantize"):
            lines.append(f"Quantization: {quantize}-bit")
        if precision := exif.get("precision"):
            lines.append(f"Precision: {precision}")

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
