from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from mflux.models.ideogram4 import Ideogram4
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.scheduler import Ideogram4Scheduler

MODEL_PATH_ENV = "MFLUX_IDEOGRAM4_MODEL_PATH"


def main() -> None:
    args = _parse_args()
    artifact, metadata = mx.load(str(Path(args.artifact).expanduser()), return_metadata=True)
    prompt = args.prompt or metadata["prompt"]
    seed = int(args.seed if args.seed is not None else metadata["seed"])
    width = int(args.width if args.width is not None else metadata["width"])
    height = int(args.height if args.height is not None else metadata["height"])
    steps = int(args.steps if args.steps is not None else metadata["steps"])
    guidance = float(args.guidance if args.guidance is not None else metadata["guidance"])
    preset_name = args.preset or metadata["preset"]
    denoise_steps = int(metadata.get("denoise_steps", "0"))

    model = Ideogram4(model_path=args.model_path)
    inputs = model._build_inputs([prompt], height=height, width=width)
    _assert_tensor("token_ids", inputs["token_ids"], artifact["token_ids"], args.atol, args.rtol)
    _assert_tensor(
        "text_position_ids", inputs["text_position_ids"], artifact["text_position_ids"], args.atol, args.rtol
    )
    _assert_tensor("position_ids", inputs["position_ids"], artifact["position_ids"], args.atol, args.rtol)
    _assert_tensor("segment_ids", inputs["segment_ids"], artifact["segment_ids"], args.atol, args.rtol)
    _assert_tensor("indicator", inputs["indicator"], artifact["indicator"], args.atol, args.rtol)

    llm_features = model._encode_prompt(
        prompt=prompt,
        width=width,
        height=height,
        inputs=inputs,
    )
    _assert_tensor("llm_features", llm_features, artifact["llm_features"], args.atol, args.rtol)

    initial_latents = Ideogram4LatentCreator.create_noise(
        seed=seed,
        width=width,
        height=height,
        latent_dim=model.conditional_transformer.config.in_channels,
    )
    _assert_tensor("initial_latents", initial_latents, artifact["initial_latents"], args.atol, args.rtol)

    sampler = Ideogram4Scheduler.get_preset(preset_name)
    t_values, s_values = Ideogram4Scheduler.make_timesteps(
        num_steps=steps,
        height=height,
        width=width,
        mu=sampler.mu,
        std=sampler.std,
    )
    _assert_tensor("t_values", mx.array(t_values, dtype=mx.float32), artifact["t_values"], args.atol, args.rtol)
    _assert_tensor("s_values", mx.array(s_values, dtype=mx.float32), artifact["s_values"], args.atol, args.rtol)
    guidance_values = _guidance_values(metadata, sampler.guidance_schedule, steps, guidance)
    _assert_tensor(
        "guidance_values",
        mx.array(guidance_values, dtype=mx.float32),
        artifact["guidance_values"],
        args.atol,
        args.rtol,
    )

    z = artifact["initial_latents"]
    text_z_padding = mx.zeros(
        (1, int(inputs["max_text_tokens"]), model.conditional_transformer.config.in_channels),
        dtype=mx.float32,
    )
    negative_inputs = model._negative_inputs(inputs, llm_features)
    for step_index in range(denoise_steps):
        schedule_index = steps - 1 - step_index
        z = model._denoise_step(
            z=z,
            t_value=float(t_values[schedule_index]),
            s_value=float(s_values[schedule_index]),
            guidance_value=float(guidance_values[schedule_index]),
            text_z_padding=text_z_padding,
            llm_features=llm_features,
            inputs=inputs,
            negative_inputs=negative_inputs,
        )
        mx.eval(z)
        _assert_tensor(
            f"latents_after_step_{step_index}", z, artifact[f"latents_after_step_{step_index}"], args.atol, args.rtol
        )

    if "decoded_image" in artifact:
        decoded = model.vae.decode(Ideogram4LatentCreator.unpack_latents(z, height, width))
        image = _to_image_array(decoded)
        _assert_tensor("decoded_image", image, artifact["decoded_image"], args.image_atol, 0.0)

    print("Ideogram 4 mflux parity checks passed.")


def _guidance_values(
    metadata: dict, preset_schedule: tuple[float, ...], steps: int, guidance: float
) -> tuple[float, ...]:
    use_preset_steps = metadata.get("use_preset_steps", "False") == "True"
    if use_preset_steps:
        return preset_schedule
    return (guidance,) * steps


def _to_image_array(decoded: mx.array) -> mx.array:
    image = mx.clip(decoded, -1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().astype(mx.uint8)
    image = image.transpose(0, 2, 3, 1)[0]
    mx.eval(image)
    return image


def _assert_tensor(name: str, actual: mx.array, expected: mx.array, atol: float, rtol: float) -> None:
    mx.eval(actual, expected)
    if _is_floating(actual) or _is_floating(expected):
        actual_np = np.array(actual.astype(mx.float32))
        expected_np = np.array(expected.astype(mx.float32))
        if not np.allclose(actual_np, expected_np, atol=atol, rtol=rtol):
            max_abs = float(np.max(np.abs(actual_np - expected_np)))
            raise AssertionError(f"{name} mismatch: max_abs={max_abs}, atol={atol}, rtol={rtol}")
    else:
        actual_np = np.array(actual)
        expected_np = np.array(expected)
        if not np.array_equal(actual_np, expected_np):
            raise AssertionError(f"{name} mismatch: expected exact equality")
    print(f"ok {name}: shape={actual.shape} dtype={actual.dtype}")


def _is_floating(value: mx.array) -> bool:
    return value.dtype in (mx.float16, mx.bfloat16, mx.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mflux Ideogram 4 tensors against an exported reference artifact."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-path", default=os.environ.get(MODEL_PATH_ENV))
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--preset")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--image-atol", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
