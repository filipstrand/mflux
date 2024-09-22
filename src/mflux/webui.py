import gradio as gr
import time
import sys
import os
import argparse

# Voeg dit toe aan het begin van je script, na de imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config, ConfigControlnet
from mflux.flux.flux import Flux1
from mflux.controlnet.flux_controlnet import Flux1Controlnet
from mflux.post_processing.image_util import ImageUtil

# Voeg deze regel toe aan het begin van je script, na de imports
LORA_DIR = os.path.join(os.path.dirname(__file__), 'lora')

def get_available_lora_files(lora_dir):
    return [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]

def generate_image_gradio(prompt, model, seed, height, width, steps, guidance, quantize, path, lora_files, lora_scales_str, metadata):
    lora_paths = [os.path.join(LORA_DIR, f) for f in lora_files] if lora_files else None
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(',')] if lora_scales_str else None

    seed = None if seed == '' else int(seed)
    quantize = None if quantize == 'None' else int(quantize)
    steps = None if steps == '' else int(steps)

    flux = Flux1(
        model_config=ModelConfig.from_alias(model),
        quantize=quantize,
        local_path=path if path else None,
        lora_paths=lora_paths,
        lora_scales=lora_scales
    )

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
        )
    )

    return image.image

def generate_image_controlnet_gradio(prompt, control_image, model, seed, height, width, steps, guidance, controlnet_strength, quantize, path, lora_files, lora_scales_str, metadata):
    lora_paths = [os.path.join(LORA_DIR, f) for f in lora_files] if lora_files else None
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(',')] if lora_scales_str else None

    seed = None if seed == '' else int(seed)
    quantize = None if quantize == 'None' else int(quantize)
    steps = None if steps == '' else int(steps)

    flux = Flux1Controlnet(
        model_config=ModelConfig.from_alias(model),
        quantize=quantize,
        local_path=path if path else None,
        lora_paths=lora_paths,
        lora_scales=lora_scales
    )

    if steps is None:
        steps = 4 if model == "schnell" else 14

    control_image_pil = control_image
    image = flux.generate_image(
        seed=int(time.time()) if seed is None else seed,
        prompt=prompt,
        control_image=control_image_pil,
        config=ConfigControlnet(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            controlnet_strength=controlnet_strength
        )
    )

    return image.image

def save_quantized_model_gradio(model, quantize, save_path):
    quantize = int(quantize)
    flux = Flux1(
        model_config=ModelConfig.from_alias(model),
        quantize=quantize,
    )

    flux.save_model(save_path)

    return f"Model saved at {save_path}"

def simple_generate_image(prompt, height, width, lora_files, lora_scales_str):
    lora_paths = [os.path.join(LORA_DIR, f) for f in lora_files] if lora_files else None
    lora_scales = [float(s.strip()) for s in lora_scales_str.split(',')] if lora_scales_str else None

    flux = Flux1(
        model_config=ModelConfig.from_alias("dev"),  # We gebruiken het 'dev' model voor deze simpele generatie
        lora_paths=lora_paths,
        lora_scales=lora_scales
    )

    image = flux.generate_image(
        seed=int(time.time()),
        prompt=prompt,
        config=Config(
            num_inference_steps=20,  # Vast aantal steps
            height=height,
            width=width,
            guidance=7.5,  # Standaard guidance scale
        )
    )

    return image.image

def create_ui(lora_dir):
    with gr.Blocks() as demo:
        gr.Image("https://raw.githubusercontent.com/CharafChnioune/mflux/main/src/mflux/assets/logo.png")

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", lines=2)
                        model = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        seed = gr.Textbox(label="Seed (optional)", value='')
                        height = gr.Number(label="Height", value=1024, precision=0)
                        width = gr.Number(label="Width", value=1024, precision=0)
                        steps = gr.Textbox(label="Inference Steps (optional)", value='')
                        guidance = gr.Number(label="Guidance Scale", value=3.5)
                        quantize = gr.Radio(choices=["None", "4", "8"], label="Quantize", value="None")
                        path = gr.Textbox(label="Model Local Path (optional)")
                        lora_files = gr.Dropdown(
                            choices=get_available_lora_files(lora_dir),
                            label="Select LoRA Files",
                            multiselect=True
                        )
                        lora_scales = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        metadata = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button = gr.Button("Generate Image")
                    with gr.Column():
                        output_image = gr.Image(label="Generated Image")
                generate_button.click(
                    fn=generate_image_gradio,
                    inputs=[prompt, model, seed, height, width, steps, guidance, quantize, path, lora_files, lora_scales, metadata],
                    outputs=output_image
                )
            with gr.TabItem("ControlNet"):
                with gr.Row():
                    with gr.Column():
                        prompt_cn = gr.Textbox(label="Prompt", lines=2)
                        control_image = gr.Image(label="Control Image", type="pil")
                        model_cn = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        seed_cn = gr.Textbox(label="Seed (optional)", value='')
                        height_cn = gr.Number(label="Height", value=1024, precision=0)
                        width_cn = gr.Number(label="Width", value=1024, precision=0)
                        steps_cn = gr.Textbox(label="Inference Steps (optional)", value='')
                        guidance_cn = gr.Number(label="Guidance Scale", value=3.5)
                        controlnet_strength = gr.Number(label="ControlNet Strength", value=0.7)
                        quantize_cn = gr.Radio(choices=["None", "4", "8"], label="Quantize", value="None")
                        path_cn = gr.Textbox(label="Model Local Path (optional)")
                        lora_files_cn = gr.Dropdown(
                            choices=get_available_lora_files(lora_dir),
                            label="Select LoRA Files",
                            multiselect=True
                        )
                        lora_scales_cn = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        metadata_cn = gr.Checkbox(label="Export Metadata as JSON", value=False)
                        generate_button_cn = gr.Button("Generate Image")
                    with gr.Column():
                        output_image_cn = gr.Image(label="Generated Image")
                generate_button_cn.click(
                    fn=generate_image_controlnet_gradio,
                    inputs=[prompt_cn, control_image, model_cn, seed_cn, height_cn, width_cn, steps_cn, guidance_cn, controlnet_strength, quantize_cn, path_cn, lora_files_cn, lora_scales_cn, metadata_cn],
                    outputs=output_image_cn
                )
            with gr.TabItem("Quantize Model"):
                with gr.Row():
                    with gr.Column():
                        model_quant = gr.Radio(choices=["dev", "schnell"], label="Model", value="dev")
                        quantize_level = gr.Radio(choices=["4", "8"], label="Quantize Level", value="8")
                        save_path = gr.Textbox(label="Save Path", placeholder="Enter the path to save the quantized model")
                        save_button = gr.Button("Save Quantized Model")
                    with gr.Column():
                        save_output = gr.Textbox(label="Output")
                save_button.click(
                    fn=save_quantized_model_gradio,
                    inputs=[model_quant, quantize_level, save_path],
                    outputs=save_output
                )
            with gr.TabItem("Simple Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt_simple = gr.Textbox(label="Prompt", lines=2)
                        height_simple = gr.Number(label="Height", value=512, precision=0)
                        width_simple = gr.Number(label="Width", value=512, precision=0)
                        lora_files_simple = gr.Dropdown(
                            choices=get_available_lora_files(lora_dir),
                            label="Select LoRA Files",
                            multiselect=True
                        )
                        lora_scales_simple = gr.Textbox(label="LoRA Scales (comma-separated, optional)")
                        generate_button_simple = gr.Button("Generate Image")
                    with gr.Column():
                        output_image_simple = gr.Image(label="Generated Image")
                generate_button_simple.click(
                    fn=simple_generate_image,
                    inputs=[prompt_simple, height_simple, width_simple, lora_files_simple, lora_scales_simple],
                    outputs=output_image_simple
                )

        gr.Markdown("**Note:** Ensure all paths and files are correct and that the models are accessible.")

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFlux Gradio Web UI")
    parser.add_argument("--lora_dir", type=str, default=LORA_DIR,
                        help="Directory containing LoRA files")
    args = parser.parse_args()

    demo = create_ui(args.lora_dir)
    demo.launch()
