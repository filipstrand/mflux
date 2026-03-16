"""Z-Image Image-to-LoRA (i2L) pipeline.

Encodes style images into LoRA weights that can be applied to the Z-Image
transformer for style transfer during generation.

Flow:
    images → SigLIP2 encoder → style embeddings (1536d)
    images → DINOv3 encoder → visual embeddings (4096d)
    concat(style, visual) → i2L decoder → LoRA weights → .safetensors
"""

import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_vision_transformer import DINOv3VisionTransformer
from mflux.models.z_image.model.z_image_i2l.i2l_decoder.i2l_decoder import ZImageI2LDecoder
from mflux.models.z_image.model.z_image_i2l.i2l_image_preprocessor import preprocess_for_dinov3, preprocess_for_siglip2
from mflux.models.z_image.model.z_image_i2l.i2l_weight_loader import (
    load_dinov3_weights,
    load_i2l_decoder_weights,
    load_siglip2_weights,
)
from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_vision_transformer import Siglip2VisionTransformer

logger = logging.getLogger(__name__)


class ZImageI2LPipeline:
    """Image-to-LoRA pipeline for Z-Image style transfer.

    Loads three models (SigLIP2, DINOv3, i2L decoder) and encodes
    style reference images into LoRA weights compatible with Z-Image.
    """

    def __init__(
        self,
        siglip2: Siglip2VisionTransformer,
        dinov3: DINOv3VisionTransformer,
        i2l_decoder: ZImageI2LDecoder,
    ):
        self.siglip2 = siglip2
        self.dinov3 = dinov3
        self.i2l_decoder = i2l_decoder

    @classmethod
    def from_pretrained(cls, precision: mx.Dtype = mx.bfloat16) -> "ZImageI2LPipeline":
        """Download weights and create the full i2L pipeline."""
        total_start = time.time()

        # Load SigLIP2
        print("Loading SigLIP2-G384...")
        t0 = time.time()
        siglip2 = Siglip2VisionTransformer()
        siglip2_weights = load_siglip2_weights(precision=precision)
        siglip2.update(siglip2_weights, strict=False)
        mx.eval(siglip2.parameters())
        print(f"  SigLIP2 loaded in {time.time() - t0:.1f}s")

        # Load DINOv3
        print("Loading DINOv3-7B...")
        t0 = time.time()
        dinov3 = DINOv3VisionTransformer()
        dinov3_weights = load_dinov3_weights(precision=precision)
        dinov3.update(dinov3_weights, strict=False)
        mx.eval(dinov3.parameters())
        print(f"  DINOv3 loaded in {time.time() - t0:.1f}s")

        # Load i2L decoder
        print("Loading i2L decoder...")
        t0 = time.time()
        i2l_decoder = ZImageI2LDecoder()
        i2l_weights = load_i2l_decoder_weights(precision=precision)
        i2l_decoder.update(i2l_weights, strict=False)
        mx.eval(i2l_decoder.parameters())
        print(f"  i2L decoder loaded in {time.time() - t0:.1f}s")

        print(f"All models loaded in {time.time() - total_start:.1f}s")
        return cls(siglip2=siglip2, dinov3=dinov3, i2l_decoder=i2l_decoder)

    def encode_images(self, images: list[Image.Image]) -> mx.array:
        """Encode images into concatenated SigLIP2 + DINOv3 embeddings.

        Args:
            images: List of PIL Images to encode.

        Returns:
            Stacked embeddings of shape (N, 5632) where N = len(images).
        """
        siglip2_embs = []
        dinov3_embs = []

        for image in images:
            # SigLIP2 encoding
            pixel_values = preprocess_for_siglip2(image)
            emb = self.siglip2(pixel_values)
            mx.eval(emb)
            siglip2_embs.append(emb)

            # DINOv3 encoding
            pixel_values = preprocess_for_dinov3(image)
            emb = self.dinov3(pixel_values)
            mx.eval(emb)
            dinov3_embs.append(emb)

        # Stack and concatenate: (N, 1536) + (N, 4096) -> (N, 5632)
        siglip2_stack = mx.concatenate(siglip2_embs, axis=0)  # (N, 1536)
        dinov3_stack = mx.concatenate(dinov3_embs, axis=0)  # (N, 4096)
        return mx.concatenate([siglip2_stack, dinov3_stack], axis=-1)  # (N, 5632)

    def decode_to_lora(self, embeddings: mx.array) -> dict[str, mx.array]:
        """Decode embeddings into LoRA weights, concatenating across images.

        Following the DiffSynth-Studio merge strategy: lora_A matrices are
        concatenated along axis 0 (increasing rank), lora_B along axis 1,
        and lora_A is scaled by alpha=1/N.

        With N images and base rank 4, the output has effective rank 4*N.

        Args:
            embeddings: (N, 5632) concatenated image embeddings.

        Returns:
            Dictionary of merged LoRA weight tensors.
        """
        num_images = embeddings.shape[0]
        all_loras = []

        for i in range(num_images):
            lora = self.i2l_decoder(embeddings[i])
            mx.eval(lora)
            all_loras.append(lora)

        if len(all_loras) == 1:
            return all_loras[0]

        # Merge by concatenation (DiffSynth-Studio strategy)
        alpha = 1.0 / num_images
        merged = {}
        lora_a_keys = [k for k in all_loras[0].keys() if ".lora_A." in k]

        for key_a in lora_a_keys:
            key_b = key_a.replace(".lora_A.", ".lora_B.")
            # Concat lora_A along axis 0: [rank, dim] x N -> [rank*N, dim]
            merged[key_a] = mx.concatenate([lora[key_a] for lora in all_loras], axis=0) * alpha
            # Concat lora_B along axis 1: [dim, rank] x N -> [dim, rank*N]
            merged[key_b] = mx.concatenate([lora[key_b] for lora in all_loras], axis=1)

        return merged

    def generate_lora(
        self,
        images: list[Image.Image],
        output_path: str | Path = "lora.safetensors",
    ) -> Path:
        """Full pipeline: encode images and save LoRA weights.

        Each image contributes rank 4 to the output LoRA. With N images,
        the output has effective rank 4*N (merged by concatenation).

        Args:
            images: List of style reference images.
            output_path: Where to save the generated .safetensors file.

        Returns:
            Path to the saved LoRA file.
        """
        output_path = Path(output_path)
        total_start = time.time()

        # Encode
        print(f"Encoding {len(images)} image(s)...")
        t0 = time.time()
        embeddings = self.encode_images(images)
        encode_time = time.time() - t0
        print(f"  Encoding done in {encode_time:.1f}s")
        print(f"  Output LoRA rank: {4 * len(images)}")

        # Decode to LoRA
        print("Generating LoRA weights...")
        t0 = time.time()
        lora = self.decode_to_lora(embeddings)
        decode_time = time.time() - t0
        print(f"  Decoding done in {decode_time:.1f}s")
        print(f"  Generated {len(lora)} LoRA weight tensors")

        # Save as safetensors (convert MLX bfloat16 -> float32 -> torch bfloat16)
        print(f"Saving to {output_path}...")
        lora_torch = {}
        for key, value in lora.items():
            np_array = np.array(value.astype(mx.float32))
            lora_torch[key] = torch.from_numpy(np_array).to(torch.bfloat16)

        save_file(lora_torch, str(output_path))

        total_time = time.time() - total_start
        print(f"✅ LoRA saved to {output_path} ({total_time:.1f}s total)")
        return output_path
