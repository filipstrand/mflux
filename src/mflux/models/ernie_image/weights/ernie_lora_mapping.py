from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class ErnieLoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return [
            # self_attention.to_q
            LoRATarget(
                model_path="layers.{block}.self_attention.to_q",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_q.lora_B.weight",
                    "layers.{block}.self_attention.to_q.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_q.lora_A.weight",
                    "layers.{block}.self_attention.to_q.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_q.alpha",
                    "layers.{block}.self_attention.to_q.alpha",
                ],
            ),
            # self_attention.to_k
            LoRATarget(
                model_path="layers.{block}.self_attention.to_k",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_k.lora_B.weight",
                    "layers.{block}.self_attention.to_k.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_k.lora_A.weight",
                    "layers.{block}.self_attention.to_k.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_k.alpha",
                    "layers.{block}.self_attention.to_k.alpha",
                ],
            ),
            # self_attention.to_v
            LoRATarget(
                model_path="layers.{block}.self_attention.to_v",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_v.lora_B.weight",
                    "layers.{block}.self_attention.to_v.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_v.lora_A.weight",
                    "layers.{block}.self_attention.to_v.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_v.alpha",
                    "layers.{block}.self_attention.to_v.alpha",
                ],
            ),
            # self_attention.to_out.0
            LoRATarget(
                model_path="layers.{block}.self_attention.to_out.0",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_out.0.lora_B.weight",
                    "layers.{block}.self_attention.to_out.0.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_out.0.lora_A.weight",
                    "layers.{block}.self_attention.to_out.0.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.self_attention.to_out.0.alpha",
                    "layers.{block}.self_attention.to_out.0.alpha",
                ],
            ),
            # mlp.gate_proj
            LoRATarget(
                model_path="layers.{block}.mlp.gate_proj",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.mlp.gate_proj.lora_B.weight",
                    "layers.{block}.mlp.gate_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.mlp.gate_proj.lora_A.weight",
                    "layers.{block}.mlp.gate_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.mlp.gate_proj.alpha",
                    "layers.{block}.mlp.gate_proj.alpha",
                ],
            ),
            # mlp.up_proj
            LoRATarget(
                model_path="layers.{block}.mlp.up_proj",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.mlp.up_proj.lora_B.weight",
                    "layers.{block}.mlp.up_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.mlp.up_proj.lora_A.weight",
                    "layers.{block}.mlp.up_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.mlp.up_proj.alpha",
                    "layers.{block}.mlp.up_proj.alpha",
                ],
            ),
            # mlp.linear_fc2
            LoRATarget(
                model_path="layers.{block}.mlp.linear_fc2",
                possible_up_patterns=[
                    "diffusion_model.layers.{block}.mlp.linear_fc2.lora_B.weight",
                    "layers.{block}.mlp.linear_fc2.lora_B.weight",
                ],
                possible_down_patterns=[
                    "diffusion_model.layers.{block}.mlp.linear_fc2.lora_A.weight",
                    "layers.{block}.mlp.linear_fc2.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "diffusion_model.layers.{block}.mlp.linear_fc2.alpha",
                    "layers.{block}.mlp.linear_fc2.alpha",
                ],
            ),
        ]
