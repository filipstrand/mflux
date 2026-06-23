import mlx.core as mx
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping


class TestKrea2LoRAMapping:
    @pytest.mark.fast
    def test_matches_diffusers_and_official_krea_keys(self):
        keys = [
            "transformer.transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_B.default.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_B.weight",
            "transformer.transformer_blocks.0.attn.to_k.lora.down.weight",
            "transformer.transformer_blocks.0.attn.to_k.lora.up.weight",
            "blocks.0.attn.wq.lora_A.weight",
            "blocks.0.attn.wq.lora_B.weight",
            "base_model.model.blocks.0.mlp.gate.lora_down.weight",
            "base_model.model.blocks.0.mlp.gate.lora_up.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.lora_B.weight",
            "txtfusion.refiner_blocks.1.mlp.down.lora_A.weight",
            "txtfusion.refiner_blocks.1.mlp.down.lora_B.weight",
            "tmlp.0.lora_A.weight",
            "tmlp.0.lora_B.weight",
            "tproj.1.lora_A.weight",
            "tproj.1.lora_B.weight",
            "lora_unet_blocks_0_attn_wq.lora_down.weight",
            "lora_unet_blocks_0_attn_wq.lora_up.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    @pytest.mark.fast
    def test_applies_official_block_lora_to_transformer(self, tmp_path):
        transformer = Krea2Transformer(
            in_channels=4,
            num_layers=1,
            attention_head_dim=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=24,
            timestep_embed_dim=4,
            text_hidden_dim=12,
            num_text_layers=2,
            text_num_attention_heads=2,
            text_num_key_value_heads=1,
            text_intermediate_size=24,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
        )
        lora_path = tmp_path / "krea2_lora.safetensors"
        mx.save_safetensors(
            str(lora_path),
            {
                "blocks.0.attn.wq.lora_A.weight": mx.ones((2, 12)),
                "blocks.0.attn.wq.lora_B.weight": mx.ones((12, 2)),
            },
        )

        lora_paths, lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(lora_path)],
            lora_scales=[0.7],
        )

        target = transformer.transformer_blocks[0].attn.to_q
        assert lora_paths == [str(lora_path)]
        assert lora_scales == [pytest.approx(0.7)]
        assert isinstance(target, LoRALinear)
        assert target.scale == pytest.approx(0.7)
        assert target.lora_A.shape == (12, 2)
        assert target.lora_B.shape == (2, 12)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        matched_keys: set[str] = set()

        for key in keys:
            for target in Krea2LoRAMapping.get_mapping():
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
