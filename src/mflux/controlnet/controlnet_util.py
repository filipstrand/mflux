import cv2
import numpy as np
import PIL


class ControlnetUtil:

    @staticmethod
    def preprocess_canny(img: PIL.Image) -> PIL.Image:
        image_to_canny = np.array(img)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = np.array(image_to_canny[:, :, None])
        image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
        return PIL.Image.fromarray(image_to_canny)
