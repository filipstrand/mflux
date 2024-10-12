CONTROLNET_STRENGTH = 0.4
GUIDANCE_SCALE = 3.5
HEIGHT, WIDTH = 1024, 1024
INIT_IMAGE_STRENGTH = 0.4  # for image-to-image init_image
MODEL_CHOICES = ["dev", "schnell"]
MODEL_INFERENCE_STEPS = {
    "dev": 14,
    "schnell": 4,
}
QUANTIZE_CHOICES = [4, 8]
