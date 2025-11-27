from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms


class DepthProWeightMapping(WeightMapping):
    @staticmethod
    def get_mapping() -> List[WeightTarget]:
        return (
            DepthProWeightMapping._get_dino_encoder_mapping("patch_encoder")
            + DepthProWeightMapping._get_dino_encoder_mapping("image_encoder")
            + DepthProWeightMapping._get_upsample_block_mapping("upsample_latent0", num_layers=4)
            + DepthProWeightMapping._get_upsample_block_mapping("upsample_latent1", num_layers=3)
            + DepthProWeightMapping._get_upsample_block_mapping("upsample0", num_layers=2)
            + DepthProWeightMapping._get_upsample_block_mapping("upsample1", num_layers=2)
            + DepthProWeightMapping._get_upsample_block_mapping("upsample2", num_layers=2)
            + DepthProWeightMapping._get_encoder_conv_mapping()
            + DepthProWeightMapping._get_decoder_mapping()
            + DepthProWeightMapping._get_head_mapping()
        )

    @staticmethod
    def _get_dino_encoder_mapping(encoder_name: str) -> List[WeightTarget]:
        prefix = f"encoder.{encoder_name}"
        return [
            WeightTarget(
                to_pattern=f"{prefix}.cls_token",
                from_pattern=[f"{prefix}.cls_token"],
            ),
            WeightTarget(
                to_pattern=f"{prefix}.pos_embed",
                from_pattern=[f"{prefix}.pos_embed"],
            ),
            WeightTarget(
                to_pattern=f"{prefix}.patch_embed.proj.weight",
                from_pattern=[f"{prefix}.patch_embed.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.patch_embed.proj.bias",
                from_pattern=[f"{prefix}.patch_embed.proj.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix}.norm.weight",
                from_pattern=[f"{prefix}.norm.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix}.norm.bias",
                from_pattern=[f"{prefix}.norm.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.norm1.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.norm1.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.norm1.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.norm1.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.attn.qkv.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.attn.qkv.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.attn.qkv.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.attn.qkv.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.attn.proj.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.attn.proj.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.attn.proj.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.attn.proj.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.ls1.gamma",
                from_pattern=[f"{prefix}.blocks.{{block}}.ls1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.norm2.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.norm2.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.norm2.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.norm2.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.mlp.fc1.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.mlp.fc1.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.mlp.fc1.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.mlp.fc1.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.mlp.fc2.weight",
                from_pattern=[f"{prefix}.blocks.{{block}}.mlp.fc2.weight"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.mlp.fc2.bias",
                from_pattern=[f"{prefix}.blocks.{{block}}.mlp.fc2.bias"],
                max_blocks=24,
            ),
            WeightTarget(
                to_pattern=f"{prefix}.blocks.{{block}}.ls2.gamma",
                from_pattern=[f"{prefix}.blocks.{{block}}.ls2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
                max_blocks=24,
            ),
        ]

    @staticmethod
    def _get_upsample_block_mapping(block_name: str, num_layers: int) -> List[WeightTarget]:
        prefix = f"encoder.{block_name}"
        targets = []

        targets.append(
            WeightTarget(
                to_pattern=f"{prefix}.layers.0.weight",
                from_pattern=[f"{prefix}.0.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            )
        )

        targets.extend(
            WeightTarget(
                to_pattern=f"{prefix}.layers.{layer}.weight",
                from_pattern=[f"{prefix}.{layer}.weight"],
                transform=WeightTransforms.transpose_conv_transpose2d_weight,
            )
            for layer in range(1, num_layers)
        )

        return targets

    @staticmethod
    def _get_encoder_conv_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="encoder.upsample_lowres.weight",
                from_pattern=["encoder.upsample_lowres.weight"],
                transform=WeightTransforms.transpose_conv_transpose2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.upsample_lowres.bias",
                from_pattern=["encoder.upsample_lowres.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.fuse_lowres.weight",
                from_pattern=["encoder.fuse_lowres.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.fuse_lowres.bias",
                from_pattern=["encoder.fuse_lowres.bias"],
            ),
        ]

    @staticmethod
    def _get_decoder_mapping() -> List[WeightTarget]:
        return (
            DepthProWeightMapping._get_decoder_convs_mapping()
            + DepthProWeightMapping._get_fusion_block_mapping(0)
            + DepthProWeightMapping._get_fusion_block_mapping(1)
            + DepthProWeightMapping._get_fusion_block_mapping(2)
            + DepthProWeightMapping._get_fusion_block_mapping(3)
            + DepthProWeightMapping._get_fusion_block_mapping(4)
        )

    @staticmethod
    def _get_decoder_convs_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="decoder.convs.1.weight",
                from_pattern=["decoder.convs.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.convs.2.weight",
                from_pattern=["decoder.convs.2.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.convs.3.weight",
                from_pattern=["decoder.convs.3.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.convs.4.weight",
                from_pattern=["decoder.convs.4.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
        ]

    @staticmethod
    def _get_fusion_block_mapping(i: int) -> List[WeightTarget]:
        targets = [
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet1.residual.1.weight",
                from_pattern=[f"decoder.fusions.{i}.resnet1.residual.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet1.residual.1.bias",
                from_pattern=[f"decoder.fusions.{i}.resnet1.residual.1.bias"],
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet1.residual.3.weight",
                from_pattern=[f"decoder.fusions.{i}.resnet1.residual.3.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet1.residual.3.bias",
                from_pattern=[f"decoder.fusions.{i}.resnet1.residual.3.bias"],
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet2.residual.1.weight",
                from_pattern=[f"decoder.fusions.{i}.resnet2.residual.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet2.residual.1.bias",
                from_pattern=[f"decoder.fusions.{i}.resnet2.residual.1.bias"],
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet2.residual.3.weight",
                from_pattern=[f"decoder.fusions.{i}.resnet2.residual.3.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.resnet2.residual.3.bias",
                from_pattern=[f"decoder.fusions.{i}.resnet2.residual.3.bias"],
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.out_conv.weight",
                from_pattern=[f"decoder.fusions.{i}.out_conv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern=f"decoder.fusions.{i}.out_conv.bias",
                from_pattern=[f"decoder.fusions.{i}.out_conv.bias"],
            ),
        ]
        if i > 0:
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.fusions.{i}.deconv.weight",
                    from_pattern=[f"decoder.fusions.{i}.deconv.weight"],
                    transform=WeightTransforms.transpose_conv_transpose2d_weight,
                )
            )
        return targets

    @staticmethod
    def _get_head_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="head.convs.0.weight",
                from_pattern=["head.0.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="head.convs.0.bias",
                from_pattern=["head.0.bias"],
            ),
            WeightTarget(
                to_pattern="head.convs.1.weight",
                from_pattern=["head.1.weight"],
                transform=WeightTransforms.transpose_conv_transpose2d_weight,
            ),
            WeightTarget(
                to_pattern="head.convs.1.bias",
                from_pattern=["head.1.bias"],
            ),
            WeightTarget(
                to_pattern="head.convs.2.weight",
                from_pattern=["head.2.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="head.convs.2.bias",
                from_pattern=["head.2.bias"],
            ),
            WeightTarget(
                to_pattern="head.convs.4.weight",
                from_pattern=["head.4.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="head.convs.4.bias",
                from_pattern=["head.4.bias"],
            ),
        ]
