CONTROLNET_STRENGTH = 0.4
GUIDANCE_SCALE = 3.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = ["dev", "dev-fill", "schnell"]
MODEL_INFERENCE_STEPS = {
    "dev": 14,
    "dev-fill": 14,
    "dev-depth": 14,
    "dev-redux": 14,
    "schnell": 4,
}
QUANTIZE_CHOICES = [3, 4, 6, 8]
