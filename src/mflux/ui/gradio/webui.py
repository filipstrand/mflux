import gradio as gr
import time
import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet

LORA_DIR = os.environ.get("MFLUX_LORA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "lora"))

flux_cache = {}

def get_or_create_flux(model, quantize, path, lora_paths, lora_scales, is_controlnet=False):
    key = (model, quantize, path, tuple(lora_paths), tuple(lora_scales), is_controlnet)
    if key not in flux_cache:
        FluxClass = Flux1Controlnet if is_controlnet else Flux1
        flux_cache[key] = FluxClass(
            model_config=ModelConfig.from_alias(model),
            quantize=quantize,
            local_path=path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
    return flux_cache[key]

def get_available_lora_files(lora_dir):
    return list(Path(lora_dir).rglob("*.safetensors"))

def generate_image_gradio(
    prompt, model, seed, height, width, steps, guidance, quantize, path, lora_files, lora_scales_str, metadata
):
    lora_paths = lora_files
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(",")] if lora_scales_str else None

    seed = None if seed == "" else int(seed)
    quantize = None if quantize == "None" else int(quantize)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, quantize, path if path else None, lora_paths, lora_scales)

    if steps is None:
        steps = 4 if model == "schnell" else 14

    image = flux.generate_image(
        seed=int(time.time()) if seed is None else seed,
        prompt=prompt,
        config=Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        ),
    )

    return image.image

def generate_image_controlnet_gradio(
    prompt,
    control_image,
    model,
    seed,
    height,
    width,
    steps,
    guidance,
    controlnet_strength,
    quantize,
    path,
    lora_files,
    lora_scales_str,
    metadata,
    save_canny,
):
    lora_paths = lora_files
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(",")] if lora_scales_str else None

    seed = None if seed == "" else int(seed)
    quantize = None if quantize == "None" else int(quantize)
    steps = None if steps == "" else int(steps)

    flux = get_or_create_flux(model, quantize, path if path else None, lora_paths, lora_scales, is_controlnet=True)

    if steps is None:
        steps = 4 if model == "schnell" else 14

    try:
        image = flux.generate_image(
            seed=int(time.time()) if seed is None else seed,
            prompt=prompt,
            output="temp_output.png",  # Tijdelijke output voor save_canny
            control_image=control_image,
            controlnet_save_canny=save_canny,
            config=ConfigControlnet(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                controlnet_strength=controlnet_strength,
            ),
        )
        return image.image, "Image generated successfully!"
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

def save_quantized_model_gradio(model, quantize, save_path):
    quantize = int(quantize)
    flux = Flux1(
        model_config=ModelConfig.from_alias(model),
        quantize=quantize,
    )

    flux.save_model(save_path)

    return f"Model saved at {save_path}"

def simple_generate_image(prompt, model, height, width, lora_files, lora_scales_str):
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(",")] if lora_scales_str else None
    steps = 20 if model == "dev (slow quality)" else 4
    model_alias = "dev" if model == "dev (slow quality)" else "schnell"

    flux = get_or_create_flux(model_alias, 4, None, lora_files, lora_scales)

    image = flux.generate_image(
        seed=int(time.time()),
        prompt=prompt,
        config=Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=7.5,
        ),
    )

    return image.image

def create_ui(lora_dir):
    with gr.Blocks() as demo:
        gr.Image("https://raw.githubusercontent.com/CharafChnioune/mflux/main/src/mflux/assets/logo.png", height=100)

        with gr.Tabs():
            with gr.TabItem("MFLUX Easy", id=0):
                with gr.Row():
                    with gr.Column():
                        prompt_simple = gr.Textbox(label="Prompt", lines=2)
                        model_simple = gr.Radio(
                            choices=["schnell (fast lower quality)", "dev (slow quality)"],
                            label="Model",
                            value="schnell (fast lower quality)",
                        )
                        height_simple = gr.Number(label="Height", value=512, precision=0)
                        width_simple = gr.Number(label="Width", value=512, precision=0)
                        lora_files_simple = gr.Dropdown(
                            choices=get_available_lora_files(LORA_DIR), label="Select LoRA Files", multiselect=True
                        )
                        lora_scales_simple = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        generate_button_simple = gr.Button("Generate Image")
                    with gr.Column():
                        output_image_simple = gr.Image(label="Generated Image")
                generate_button_simple.click(
                    fn=simple_generate_image,
                    inputs=[
                        prompt_simple,
                        model_simple,
                        height_simple,
                        width_simple,
                        lora_files_simple,
                        lora_scales_simple,
                    ],
                    outputs=output_image_simple,
                )

            with gr.TabItem("Advanced Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", lines=2)
                        model = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        seed = gr.Textbox(label="Seed (optional)", value="")
                        height = gr.Number(label="Height", value=1024, precision=0)
                        width = gr.Number(label="Width", value=1024, precision=0)
                        steps = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance = gr.Number(label="Guidance Scale", value=3.5)
                        quantize = gr.Radio(choices=["None", "4", "8"], label="Quantize", value="None")
                        path = gr.Textbox(label="Model Local Path (optional)")
                        lora_files = gr.Dropdown(
                            choices=get_available_lora_files(LORA_DIR), label="Select LoRA Files", multiselect=True
                        )
                        lora_scales = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        metadata = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button = gr.Button("Generate Image")
                    with gr.Column():
                        output_image = gr.Image(label="Generated Image")
                generate_button.click(
                    fn=generate_image_gradio,
                    inputs=[
                        prompt,
                        model,
                        seed,
                        height,
                        width,
                        steps,
                        guidance,
                        quantize,
                        path,
                        lora_files,
                        lora_scales,
                        metadata,
                    ],
                    outputs=output_image,
                )

            with gr.TabItem("ControlNet"):
                with gr.Row():
                    with gr.Column():
                        prompt_cn = gr.Textbox(label="Prompt", lines=2)
                        control_image = gr.Image(label="Control Image", type="pil")
                        model_cn = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        seed_cn = gr.Textbox(label="Seed (optional)", value="")
                        height_cn = gr.Number(label="Height", value=1024, precision=0)
                        width_cn = gr.Number(label="Width", value=1024, precision=0)
                        steps_cn = gr.Textbox(label="Inference Steps (optional)", value="")
                        guidance_cn = gr.Number(label="Guidance Scale", value=3.5)
                        controlnet_strength = gr.Number(label="ControlNet Strength", value=0.7)
                        quantize_cn = gr.Radio(choices=["None", "4", "8"], label="Quantize", value="None")
                        path_cn = gr.Textbox(label="Model Local Path (optional)")
                        lora_files_cn = gr.Dropdown(
                            choices=get_available_lora_files(LORA_DIR), label="Select LoRA Files", multiselect=True
                        )
                        lora_scales_cn = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        metadata_cn = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        save_canny = gr.Checkbox(label="Save Canny Edge Detection Image", value=False)
                        generate_button_cn = gr.Button("Generate Image")
                    with gr.Column():
                        output_image_cn = gr.Image(label="Generated Image")
                        output_message_cn = gr.Textbox(label="Status")
                generate_button_cn.click(
                    fn=generate_image_controlnet_gradio,
                    inputs=[
                        prompt_cn,
                        control_image,
                        model_cn,
                        seed_cn,
                        height_cn,
                        width_cn,
                        steps_cn,
                        guidance_cn,
                        controlnet_strength,
                        quantize_cn,
                        path_cn,
                        lora_files_cn,
                        lora_scales_cn,
                        metadata_cn,
                        save_canny,
                    ],
                    outputs=[output_image_cn, output_message_cn],
                )

                gr.Markdown("""
                ⚠️ Note: Controlnet requires an additional one-time download of ~3.58GB of weights from Huggingface. This happens automatically the first time you run the generate command.
                At the moment, the Controlnet used is [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny), which was trained for the `dev` model. 
                It can work well with `schnell`, but performance is not guaranteed.

                ⚠️ Note: The output can be highly sensitive to the controlnet strength and is very much dependent on the reference image. 
                Too high settings will corrupt the image. A recommended starting point is a value like 0.4. Experiment with different strengths to find the best result.
                """)

            with gr.TabItem("Quantize Model"):
                with gr.Row():
                    with gr.Column():
                        model_quant = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        quantize_level = gr.Radio(choices=["4", "8"], label="Quantize Level", value="8")
                        save_path = gr.Textbox(
                            label="Save Path", placeholder="Enter the path to save the quantized model"
                        )
                        save_button = gr.Button("Save Quantized Model")
                    with gr.Column():
                        save_output = gr.Textbox(label="Output")
                save_button.click(
                    fn=save_quantized_model_gradio, inputs=[model_quant, quantize_level, save_path], outputs=save_output
                )

        gr.Markdown("**Note:** Ensure all paths and files are correct and that the models are accessible.")

    return demo

def main():
    parser = argparse.ArgumentParser(description="MFlux Gradio Web UI")
    parser.add_argument("--lora_dir", type=str, default=LORA_DIR, help="Directory containing LoRA files")
    args = parser.parse_args()

    demo = create_ui(args.lora_dir)
    demo.launch()

if __name__ == "__main__":
    main()
