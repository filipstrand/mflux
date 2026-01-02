from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.utils.image_util import ImageUtil

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedControls:
    control_latents: list[mx.array]
    strengths: list[float]
    types: list[ControlType]
    images: list[PIL.Image.Image]


class ZImageControlnetUtil:
    @staticmethod
    def encode_controls(
        *,
        vae: VAE,
        width: int,
        height: int,
        controls: list[ControlSpec],
    ) -> EncodedControls:
        if len(controls) == 0:
            raise ValueError("At least one control must be provided.")

        control_latents: list[mx.array] = []
        strengths: list[float] = []
        types: list[ControlType] = []
        images: list[PIL.Image.Image] = []

        for control in controls:
            img = ImageUtil.load_image(control.image_path)
            img = ZImageControlnetUtil._scale_image(img=img, width=width, height=height)
            img = ZImageControlnetUtil._preprocess(img, control.type)

            arr = ImageUtil.to_array(img)
            latent = vae.encode(arr)  # (1, 16, 1, H/8, W/8)
            latent = ZImageLatentCreator.pack_latents(latent, height=height, width=width)  # (16, 1, H/8, W/8)

            # Diffusers ZImageControlNetPipeline behavior:
            # If base latents are 16ch but ControlNet expects 33ch, pad the control latents with zeros to 33ch.
            if latent.shape[0] < 33:
                padding = mx.zeros((33 - latent.shape[0], *latent.shape[1:]), dtype=latent.dtype)
                latent = mx.concatenate([latent, padding], axis=0)

            control_latents.append(latent)
            strengths.append(float(control.strength))
            types.append(control.type)
            images.append(img)

        return EncodedControls(control_latents=control_latents, strengths=strengths, types=types, images=images)

    @staticmethod
    def _preprocess(img: PIL.Image.Image, control_type: ControlType) -> PIL.Image.Image:
        # For Union checkpoints, many modalities can be supplied as *already-preprocessed* control images.
        # We only implement light-weight preprocessing here (Canny); others are pass-through.
        if control_type == ControlType.canny:
            # OpenCV Canny expects an 8-bit single-channel image.
            gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
            edges_u8 = cv2.Canny(gray_u8, 100, 200)
            edges_rgb = np.repeat(edges_u8[:, :, None], 3, axis=2)
            return PIL.Image.fromarray(edges_rgb)

        return img

    @staticmethod
    def _scale_image(*, img: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        if height != img.height or width != img.width:
            log.warning(
                f"Control image {img.width}x{img.height} has different dimensions than requested. Resizing to {width}x{height}"
            )
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img
