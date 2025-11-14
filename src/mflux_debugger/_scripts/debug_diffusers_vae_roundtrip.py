import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from PIL import Image

from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir
from mflux_debugger.tensor_debug import debug_save

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


OWL_PATH = Path("/Users/filipstrand/Desktop/owl.png")


def load_image_as_tensor() -> torch.Tensor:
    if not OWL_PATH.exists():
        raise FileNotFoundError(f"Input image not found at {OWL_PATH}")

    config = TXT2IMG_DEBUG_CONFIG

    image = Image.open(OWL_PATH).convert("RGB")
    image = image.resize((config.width, config.height), Image.BICUBIC)

    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.unsqueeze(2)
    return tensor


def main():
    torch.set_grad_enabled(False)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    vae = AutoencoderKLWan.from_pretrained(
        "briaai/FIBO",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae = vae.to(device)
    vae.eval()

    x = load_image_as_tensor().to(device)
    x = x.to(next(vae.parameters()).dtype)

    with torch.no_grad():
        enc_out = vae.encode(x, return_dict=True)
        posterior = enc_out.latent_dist
        z = posterior.mean

        debug_save(z, "vae_roundtrip_latents")

        dec_out = vae.decode(z, return_dict=True)
        recon_5d = dec_out.sample

    recon = recon_5d[:, :, 0, :, :]
    recon = torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0)

    recon_np = recon.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    recon_img = Image.fromarray((recon_np * 255.0).round().astype("uint8"))

    archive_images("pytorch")

    images_dir = get_images_latest_framework_dir("pytorch")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_diffusers_vae_roundtrip_{timestamp}.png"
    recon_img.save(str(output_path))
    print(f"Saved PyTorch VAE roundtrip image: {output_path}")


if __name__ == "__main__":
    main()
