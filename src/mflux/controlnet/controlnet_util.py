import logging
import os

import cv2
import numpy as np
import PIL.Image

log = logging.getLogger(__name__)


class ControlnetUtil:
    @staticmethod
    def preprocess_canny(img: PIL.Image) -> PIL.Image:
        image_to_canny = np.array(img)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = np.array(image_to_canny[:, :, None])
        image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
        return PIL.Image.fromarray(image_to_canny)

    @staticmethod
    def scale_image(height: int, width: int, img: PIL.Image) -> PIL.Image:
        if height != img.height or width != img.width:
            log.warning(f"Control image has different dimensions than the model. Resizing to {width}x{height}")
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img

    @staticmethod
    def save_canny_image(control_image: PIL.Image, path: str):
        from mflux import ImageUtil

        base, ext = os.path.splitext(path)
        new_filename = f"{base}_controlnet_canny{ext}"
        ImageUtil.save_image(control_image, new_filename)
