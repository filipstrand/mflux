from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping


class TestFlux2LoRAMapping:
    def test_matches_diffusion_model_double_block_keys(self):
        keys = [
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_B.weight",
            "diffusion_model.double_blocks.0.img_mlp.0.lora_A.weight",
            "diffusion_model.double_blocks.0.img_mlp.0.lora_B.weight",
            "diffusion_model.double_blocks.0.img_mlp.2.lora_A.weight",
            "diffusion_model.double_blocks.0.img_mlp.2.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_mlp.0.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_mlp.0.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_mlp.2.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_mlp.2.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_single_transformer_default_weight_keys(self):
        keys = [
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default.weight",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.default.weight",
            "single_transformer_blocks.0.attn.to_out.lora_A.default.weight",
            "single_transformer_blocks.0.attn.to_out.lora_B.default.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_transformer_block_default_weight_keys(self):
        keys = [
            "transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "transformer_blocks.0.attn.to_q.lora_B.default.weight",
            "transformer_blocks.0.attn.to_out.0.lora_A.default.weight",
            "transformer_blocks.0.attn.to_out.0.lora_B.default.weight",
            "transformer_blocks.0.ff.linear_in.lora_A.default.weight",
            "transformer_blocks.0.ff.linear_in.lora_B.default.weight",
            "transformer_blocks.0.ff_context.linear_out.lora_A.default.weight",
            "transformer_blocks.0.ff_context.linear_out.lora_B.default.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusion_model_single_block_keys(self):
        keys = [
            "diffusion_model.single_blocks.0.linear1.lora_A.weight",
            "diffusion_model.single_blocks.0.linear1.lora_B.weight",
            "diffusion_model.single_blocks.0.linear2.lora_A.weight",
            "diffusion_model.single_blocks.0.linear2.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_preserves_existing_base_model_double_block_matches(self):
        keys = [
            "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "base_model.model.double_blocks.0.txt_attn.proj.lora_A.weight",
            "base_model.model.double_blocks.0.txt_attn.proj.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        matched_keys: set[str] = set()

        for key in keys:
            for target in Flux2LoRAMapping.get_mapping():
                pattern_groups = (
                    target.possible_up_patterns,
                    target.possible_down_patterns,
                    target.possible_alpha_patterns,
                )
                for patterns in pattern_groups:
                    if any(LoRALoader._match_pattern(key, pattern) is not None for pattern in patterns):
                        matched_keys.add(key)
                        break
                if key in matched_keys:
                    break

        return matched_keys
