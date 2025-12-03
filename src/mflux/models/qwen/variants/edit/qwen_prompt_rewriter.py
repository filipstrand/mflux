from typing import Optional


class QwenPromptRewriter:
    @staticmethod
    def detect_language(prompt: str) -> str:
        chinese_ranges = [
            ("\u4e00", "\u9fff"),
        ]
        for char in prompt:
            if any(start <= char <= end for start, end in chinese_ranges):
                return "zh"
        return "en"

    @staticmethod
    def enhance_prompt_en(original_prompt: str) -> str:
        # Remove extra whitespace and normalize
        prompt = original_prompt.strip()

        # Add style and quality enhancers if not present
        quality_terms = ["ultra hd", "4k", "cinematic", "high quality", "detailed"]
        has_quality = any(term in prompt.lower() for term in quality_terms)

        if not has_quality:
            prompt += ", ultra HD, 4K, cinematic composition, high quality, detailed"

        # Enhance editing-specific instructions
        if any(word in prompt.lower() for word in ["change", "replace", "modify", "edit"]):
            if "maintain" not in prompt.lower() and "keep" not in prompt.lower():
                prompt += ", maintain original composition and lighting"

        # Add style consistency for artistic transformations
        art_styles = ["watercolor", "oil painting", "sketch", "digital art", "anime", "cartoon"]
        if any(style in prompt.lower() for style in art_styles):
            prompt += ", consistent artistic style throughout"

        return prompt

    @staticmethod
    def enhance_prompt_zh(original_prompt: str) -> str:
        # Remove extra whitespace and normalize
        prompt = original_prompt.strip()

        # Add quality enhancers for Chinese prompts
        quality_terms = ["超清", "4k", "高质量", "精细", "电影级"]
        has_quality = any(term in prompt for term in quality_terms)

        if not has_quality:
            prompt += "，超清，4K，电影级构图，高质量，精细"

        # Enhance editing-specific instructions
        edit_terms = ["改变", "替换", "修改", "编辑", "变成"]
        if any(term in prompt for term in edit_terms):
            if "保持" not in prompt and "维持" not in prompt:
                prompt += "，保持原有构图和光线"

        return prompt

    @staticmethod
    def enhance_edit_prompt(prompt: str, context: Optional[str] = None) -> str:
        language = QwenPromptRewriter.detect_language(prompt)

        if language == "zh":
            enhanced = QwenPromptRewriter.enhance_prompt_zh(prompt)
        else:
            enhanced = QwenPromptRewriter.enhance_prompt_en(prompt)

        # Add edit-specific stability improvements
        edit_keywords = {
            "en": ["transform", "change", "modify", "edit", "replace", "add", "remove"],
            "zh": ["转换", "改变", "修改", "编辑", "替换", "添加", "移除", "变成"],
        }

        current_keywords = edit_keywords[language]
        has_edit_instruction = any(keyword in enhanced.lower() for keyword in current_keywords)

        if has_edit_instruction:
            if language == "zh":
                if "保持细节" not in enhanced:
                    enhanced += "，保持细节准确"
            else:
                if "preserve details" not in enhanced.lower():
                    enhanced += ", preserve fine details"

        return enhanced

    @staticmethod
    def create_edit_template(
        action: str, target: str, replacement: Optional[str] = None, style: Optional[str] = None, language: str = "en"
    ) -> str:
        if language == "zh":
            templates = {
                "replace": f"将{target}替换为{replacement}" if replacement else f"替换{target}",
                "add": f"在图像中添加{target}",
                "remove": f"从图像中移除{target}",
                "transform": f"将图像转换为{target}风格",
            }
            quality_suffix = "，保持原有构图，超清，4K，精细细节"
        else:
            templates = {
                "replace": f"Replace {target} with {replacement}" if replacement else f"Replace {target}",
                "add": f"Add {target} to the image",
                "remove": f"Remove {target} from the image",
                "transform": f"Transform the image into {target} style",
            }
            quality_suffix = ", maintain original composition, ultra HD, 4K, fine details"

        base_prompt = templates.get(action, target)

        if style:
            if language == "zh":
                base_prompt += f"，{style}风格"
            else:
                base_prompt += f", {style} style"

        return base_prompt + quality_suffix
