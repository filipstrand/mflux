from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

MODEL_PATH_ENV = "MFLUX_IDEOGRAM4_MODEL_PATH"
REFERENCE_REPO_ENV = "MFLUX_IDEOGRAM4_REFERENCE_REPO"
DEFAULT_PROMPT = "A crisp poster for a jazz trio, clean typography"


def main() -> None:
    args = _parse_args()
    reference_repo = Path(args.reference_repo).expanduser()
    sys.path.insert(0, str(reference_repo))

    from mlx_vlm.models.ideogram4.pipeline import Ideogram4Image
    from mlx_vlm.models.ideogram4.weights import load_vae

    model_path = Path(args.model_path).expanduser()
    pipeline = Ideogram4Image.from_pretrained(
        model_path=model_path,
        evict_text_encoder=False,
        evict_transformers=False,
    )
    inputs = pipeline._build_inputs([args.prompt], height=args.height, width=args.width)
    llm_features = pipeline._encode_text(
        inputs["token_ids"],
        inputs["text_position_ids"],
        inputs["indicator"],
    )

    preset = _get_reference_preset(args.preset)
    num_steps = preset.num_steps if args.use_preset_steps else args.steps
    guidance_values = preset.guidance_schedule if args.use_preset_steps else (float(args.guidance),) * num_steps
    t_values, s_values = _make_reference_timesteps(
        num_steps=num_steps,
        height=args.height,
        width=args.width,
        mu=preset.mu,
        std=preset.std,
    )

    batch_size = 1
    latent_dim = _latent_dim(model_path)
    mx.random.seed(args.seed)
    z = mx.random.normal(
        (batch_size, int(inputs["num_image_tokens"]), latent_dim),
        dtype=mx.float32,
    )
    text_z_padding = mx.zeros(
        (batch_size, int(inputs["max_text_tokens"]), latent_dim),
        dtype=mx.float32,
    )

    arrays = {
        "token_ids": inputs["token_ids"],
        "text_position_ids": inputs["text_position_ids"],
        "position_ids": inputs["position_ids"],
        "segment_ids": inputs["segment_ids"],
        "indicator": inputs["indicator"],
        "llm_features": llm_features,
        "initial_latents": z,
        "t_values": mx.array(t_values, dtype=mx.float32),
        "s_values": mx.array(s_values, dtype=mx.float32),
        "guidance_values": mx.array(guidance_values, dtype=mx.float32),
    }

    denoise_steps = min(args.denoise_steps, num_steps)
    if denoise_steps > 0:
        pipeline._ensure_transformers_and_vae()
        negative_inputs = _negative_inputs(inputs, llm_features)
        for step_index in range(denoise_steps):
            schedule_index = num_steps - 1 - step_index
            t = mx.full((batch_size,), float(t_values[schedule_index]), dtype=mx.float32)
            pos_z = mx.concatenate([text_z_padding, z], axis=1)
            pos_out = pipeline.conditional_transformer(
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )
            pos_v = pos_out[:, int(inputs["max_text_tokens"]) :, :]
            neg_v = pipeline.unconditional_transformer(
                llm_features=negative_inputs["llm_features"],
                x=z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                segment_ids=negative_inputs["segment_ids"],
                indicator=negative_inputs["indicator"],
            )
            v = float(guidance_values[schedule_index]) * pos_v
            v = v + (1.0 - float(guidance_values[schedule_index])) * neg_v
            z = z + v * (float(s_values[schedule_index]) - float(t_values[schedule_index]))
            mx.eval(z)
            arrays[f"latents_after_step_{step_index}"] = z

    if args.decode:
        if pipeline.vae is None:
            pipeline.vae = load_vae(model_path)
        arrays["decoded_image"] = pipeline._decode(
            z,
            grid_h=int(inputs["grid_h"]),
            grid_w=int(inputs["grid_w"]),
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(*arrays.values())
    mx.save_safetensors(
        str(output),
        arrays,
        metadata={
            "source": "mlx-vlm",
            "reference_repo": str(reference_repo),
            "model_path": str(model_path),
            "prompt": args.prompt,
            "seed": str(args.seed),
            "steps": str(num_steps),
            "denoise_steps": str(denoise_steps),
            "width": str(args.width),
            "height": str(args.height),
            "preset": args.preset,
            "guidance": str(args.guidance),
            "use_preset_steps": str(args.use_preset_steps),
        },
    )
    print(f"Wrote reference artifact: {output}")


def _negative_inputs(inputs: dict, llm_features: mx.array) -> dict[str, mx.array]:
    max_text_tokens = int(inputs["max_text_tokens"])
    num_image_tokens = int(inputs["num_image_tokens"])
    return {
        "position_ids": inputs["position_ids"][:, max_text_tokens:, :],
        "segment_ids": inputs["segment_ids"][:, max_text_tokens:],
        "indicator": inputs["indicator"][:, max_text_tokens:],
        "llm_features": mx.zeros((1, num_image_tokens, llm_features.shape[-1]), dtype=llm_features.dtype),
    }


def _get_reference_preset(name: str):
    from mlx_vlm.models.ideogram4.scheduler import get_preset

    return get_preset(name)


def _make_reference_timesteps(*, num_steps: int, height: int, width: int, mu: float, std: float):
    from mlx_vlm.models.ideogram4.scheduler import make_timesteps

    return make_timesteps(num_steps=num_steps, height=height, width=width, mu=mu, std=std)


def _latent_dim(model_path: Path) -> int:
    config = json.loads((model_path / "transformer" / "config.json").read_text())
    return int(config.get("in_channels", 128))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic Ideogram 4 tensors from the mlx-vlm reference.")
    parser.add_argument("--reference-repo", default=os.environ.get(REFERENCE_REPO_ENV))
    parser.add_argument("--model-path", default=os.environ.get(MODEL_PATH_ENV))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--preset", default="V4_DEFAULT_20")
    parser.add_argument("--use-preset-steps", action="store_true")
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.reference_repo is None:
        parser.error(f"--reference-repo is required or set {REFERENCE_REPO_ENV}")
    if args.model_path is None:
        parser.error(f"--model-path is required or set {MODEL_PATH_ENV}")
    return args


if __name__ == "__main__":
    main()
