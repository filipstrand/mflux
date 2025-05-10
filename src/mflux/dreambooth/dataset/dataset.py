from pathlib import Path

import mlx.core as mx
import numpy as np  # local import to avoid adding global dependency
import PIL.Image
from mlx import nn
from PIL import Image as PilImage
from tqdm import tqdm

from mflux import Flux1, ImageUtil
from mflux.dreambooth.dataset.batch import Example
from mflux.dreambooth.dataset.dreambooth_preprocessing import DreamBoothPreProcessing
from mflux.dreambooth.state.training_spec import ExampleSpec
from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.post_processing.array_util import ArrayUtil


class Dataset:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    @staticmethod
    def prepare_dataset(
        flux: Flux1,
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
    ) -> "Dataset":
        # Encode the original examples (image and text)
        examples = Dataset._create_examples(flux, raw_data, width=width, height=height)

        # Expend the original dataset to get more training data with variations
        augmented_examples = []
        for example in examples:
            [augmented_examples.append(variation) for variation in DreamBoothPreProcessing.augment(example)]

        # Dataset is now prepared
        return Dataset(augmented_examples)

    def size(self) -> int:
        return len(self.examples)

    @staticmethod
    def _create_examples(
        flux: Flux1,
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
    ) -> list[Example]:
        examples = []
        depth_pro = DepthPro()

        for i, entry in enumerate(tqdm(raw_data, desc="Encoding original dataset")):
            # Encode the image
            encoded_image = Dataset._encode_image_from_path(flux.vae, entry.image, width=width, height=height)

            # Encode the prompt
            prompt_embeds = flux.t5_text_encoder(flux.t5_tokenizer.tokenize(entry.prompt))
            pooled_prompt_embeds = flux.clip_text_encoder(flux.clip_tokenizer.tokenize(entry.prompt))

            # Create depth-based weighting map (optional)
            depth_map = Dataset._create_depth_weight_map(
                example_spec=entry,
                depth_pro=depth_pro,
                height=height,
                width=width,
            )

            # Create the example object
            example = Example(
                example_id=i,
                prompt=entry.prompt,
                image_path=entry.image,
                encoded_image=encoded_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                depth_map=depth_map,
            )
            examples.append(example)

            # Evaluate to enable progress tracking
            mx.eval(encoded_image, prompt_embeds, pooled_prompt_embeds)
            if depth_map is not None:
                mx.eval(depth_map)

        return examples

    @staticmethod
    def _create_depth_weight_map(
        example_spec: ExampleSpec,
        depth_pro: DepthPro,
        height: int,
        width: int,
    ) -> mx.array | None:
        """Create a per-patch weighting map from the depth image.

        The returned tensor is shaped like the model latents `(1, N, 64)` where
        `N = (width // 16) * (height // 16)`.  Each of the 64 feature channels
        for a patch receives the same scalar weight (broadcast) that is
        proportional to the *foregroundness* (white = 1.0, black = 0.0).
        """
        if not example_spec.use_depth:
            return None

        # 1. Generate and (optionally) transform the depth map
        depth_result = depth_pro.create_depth_map(image_path=example_spec.image)
        transformed = depth_result.apply_transformation(
            transform_type="sigmoid",
            strength=2.0,
        )

        depth_np = np.array(transformed.depth_image).astype("float32") / 255.0  # (H, W)

        # 3. Down-sample to one weight per 16×16 patch to match latent resolution

        patch_h, patch_w = height // 16, width // 16
        depth_small = PilImage.fromarray((depth_np * 255).astype("uint8")).resize(
            (patch_w, patch_h),
            resample=PilImage.BILINEAR,
        )
        depth_small_np = np.array(depth_small).astype("float32") / 255.0  # (patch_h, patch_w)

        # 4. Flatten and broadcast over the latent feature dimension (64)
        depth_flat = depth_small_np.flatten()  # (N,)
        weight_per_patch = mx.array(depth_flat, dtype=mx.float16).reshape(1, -1, 1)  # (1, N, 1)
        weight_per_patch = mx.repeat(weight_per_patch, 64, axis=2)  # (1, N, 64)

        return weight_per_patch

    @staticmethod
    def _encode_image_from_path(vae: nn.Module, image_path: Path, width: int, height: int) -> mx.array:
        image = PIL.Image.open(image_path.resolve()).convert("RGB")
        return Dataset._encode_image(vae=vae, image=image, width=width, height=height)

    @staticmethod
    def _encode_image(vae: nn.Module, image: PIL.Image, width: int, height: int) -> mx.array:
        scaled_user_image = ImageUtil.scale_to_dimensions(image, target_width=width, target_height=height)
        encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
        latents = ArrayUtil.pack_latents(encoded, width=width, height=height)
        return latents
