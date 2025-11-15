import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
from diffusers import BriaFiboPipeline

from mflux_debugger.tensor_debug import debug_save


def load_pt_tensor(name: str) -> torch.Tensor:
    """
    Minimal loader for tensors saved via debug_save on the PyTorch side.
    """
    # NOTE: For debugging we use an absolute path to the shared tensor directory.
    base = Path("/Users/filipstrand/Desktop/mflux/mflux_debugger/tensors/latest")
    path = base / f"{name}.npy"
    arr = np.load(path)
    return torch.from_numpy(arr)


def main():
    torch.set_grad_enabled(False)

    # Load FIBO pipeline just to get the transformer weights.
    pipe = BriaFiboPipeline.from_pretrained("briaai/FIBO", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    device = pipe._execution_device
    transformer = pipe.transformer

    # Load pre-saved transformer inputs from debug_save
    hidden_states = load_pt_tensor("pt_hidden_states").to(device=device, dtype=transformer.dtype)
    timestep = load_pt_tensor("pt_timestep").to(device=device, dtype=transformer.dtype)
    encoder_hidden_states = load_pt_tensor("pt_encoder_hidden_states").to(device=device, dtype=transformer.dtype)

    stacked_prompt_layers = load_pt_tensor("pt_prompt_layers")  # (L, B, S, D)
    prompt_layers = [
        stacked_prompt_layers[i].to(device=device, dtype=transformer.dtype)
        for i in range(stacked_prompt_layers.shape[0])
    ]

    text_ids = load_pt_tensor("pt_text_ids").to(device=device)
    latent_image_ids = load_pt_tensor("pt_latent_image_ids").to(device=device)
    attention_mask = load_pt_tensor("pt_attention_mask").to(device=device, dtype=transformer.dtype)

    joint_attention_kwargs = {"attention_mask": attention_mask}

    # Direct transformer call from saved inputs
    with torch.no_grad():
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_encoder_layers=prompt_layers,
            timestep=timestep,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=None,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

    debug_save(out, "pytorch_transformer_output")
    print("Saved 'pytorch_transformer_output' from direct transformer call.")


if __name__ == "__main__":
    main()
