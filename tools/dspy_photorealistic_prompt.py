#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["dspy", "ollama", "rich"]
# ///

# This is an example script to show you the power of enhancing a naive user prompt
# to fill out details that would be helpful for a image generation model to follow
# the example here is not meant to be a comprehensive workflow that fits all needs
# the intended usage is for you to copy/paste into your own Python-based workflow
# for mflux/Flux/Qwen models and theoretically any future image model

import json
from typing import Literal

import dspy
from rich import box, print
from rich.live import Live
from rich.table import Table

# configure model: https://dspy.ai/learn/programming/language_models/?h=openai+compatible#__tabbed_1_6
MODEL = "ollama/qwen3:14b"
LOCAL_API_BASE = "http://localhost:11434"
LOCAL_LM = dspy.LM(
    model=MODEL,
    api_key="not_required_for_local",
    api_base=LOCAL_API_BASE,
    # enable cache if you want the LM instance to
    # cache responses for the same prompts in a long living app
    cache=True,
)
dspy.configure(lm=LOCAL_LM)


class PhotorealisticPrompt(dspy.Signature):
    user_prompt: str = dspy.InputField(
        desc="a high level description of an image formed by a novice casual user who wants to describe a photorealistic image but does not know the technical terms"
    )
    camera_model: Literal[
        "iPhone", "Canon", "Fujifilm", "Kodak", "Leica", "Nikon", "Olympus", "Polaroid", "Sony", "Hasselblad"
    ] = dspy.OutputField()
    camera_type: Literal["Film", "Digital"] = dspy.OutputField()
    resolution: Literal["lowres", "highres", "4k", "8k"] = dspy.OutputField()
    lens: Literal[
        # Focal Lengths
        "16mm",
        "24mm",
        "35mm",
        "50mm",
        "85mm",
        "135mm",
        "200mm",
        # Lens Types
        "Wide-angle",
        "Prime lens",
        "Telephoto",
        "Macro",
        "Fisheye",
        "Standard",
        "Zoom lens",
    ] = dspy.OutputField()
    shot_angle: Literal[
        "Eye-level shot",
        "Low-angle shot",
        "High-angle shot",
        "Dutch angle",
        "Over-the-shoulder shot",
        "Bird's-eye view",
    ] = dspy.OutputField()
    pose: Literal[
        # Body Position
        "Standing",
        "Sitting",
        "Walking",
        "Running",
        "Jumping",
        "Lying down",
        "Prone",
        "Supine",
        # Shot Type
        "Full body shot",
        "Portrait",
        "Headshot",
        "Close-up",
        "Profile shot",
        # Action/Style
        "Candid",
        "Action shot",
        "Looking at camera",
        "Looking away",
    ] = dspy.OutputField()
    focus: Literal[
        "Foreground focus",
        "Midground focus",
        "Background focus",
        "Deep focus",
        "Shallow focus",
        "Soft focus",
        "Rack focus",
    ] = dspy.OutputField()
    f_stop: Literal["f/1.4", "f/1.8", "f/2.0", "f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11", "f/16", "f/22"] = (
        dspy.OutputField()
    )
    shutter_speed: Literal["1/4000s", "1/1000s", "1/250s", "1/60s", "1/15s", "1s", "5s"] = dspy.OutputField()
    iso: Literal["ISO 50", "ISO 100", "ISO 400", "ISO 800", "ISO 1600", "ISO 3200", "ISO 6400"] = dspy.OutputField()
    lighting: Literal[
        # Quality & Type
        "Soft light",
        "Hard light",
        "Natural light",
        "Studio light",
        "Ambient light",
        "Backlit",
        "Rim lighting",
        # Time & Mood
        "Cinematic lighting",
        "Dramatic lighting",
        "Golden hour",
        "Blue hour",
        # Techniques
        "High key",
        "Low key",
        "Rembrandt lighting",
        "Volumetric lighting",
    ] = dspy.OutputField()
    color_style: Literal[
        "Vibrant",
        "Muted",
        "Monochrome",
        "Black and White",
        "Sepia",
        "High contrast",
        "Low contrast",
        "Cinematic teal and orange",
    ] = dspy.OutputField()
    film_stock: Literal[
        None,
        "Kodak Portra 400",
        "Kodak Ektar 100",
        "Fujifilm Velvia 50",
        "Fujifilm Pro 400H",
        "Ilford HP5 Plus 400",
        "CineStill 800T",
    ] = dspy.OutputField()
    overall_mood: Literal[
        "Serene", "Energetic", "Melancholy", "Joyful", "Mysterious", "Nostalgic", "Futuristic", "Romantic"
    ] = dspy.OutputField()


def enhance_user_prompt(user_prompt: str, trigger_words: str = None):
    prediction = generate_enhanced_prompt(user_prompt=user_prompt)
    enhanced_prompt_dict = prediction.toDict()
    enhanced_prompt_dict.pop("reasoning")
    if trigger_words:
        enhanced_prompt_dict["trigger_words"] = trigger_words
    enhanced_prompt_dict["description"] = user_prompt
    return enhanced_prompt_dict


EXAMPLE_NAIVE_USER_PROMPTS = [
    "iPhone macro photo of a plate of high-end Italian food",
    "DSLR Telephoto of a climber celebrating at the summit of Yosemite's El Capitan",
    "In a dark room, a cat pounces on a laser pointer",
    "Wide angle lens, Birds eye view of a dog lying supine on the lawn with its paws up",
    "Canon DSLR, studio backdrop, High school yearbook portrait of a athletic track and field runner sporting his team jacket",
]

generate_enhanced_prompt = dspy.ChainOfThought(PhotorealisticPrompt)


def run_examples():
    print(f"No prompt provided. This program will now show you {len(EXAMPLE_NAIVE_USER_PROMPTS)} examples.")
    print("Generating speed varies by the capabilities of your hardware.")
    print("You may need to be patient as the results stream in.")
    grid = Table(title="Prompt Enhancement Examples", box=box.MINIMAL_DOUBLE_HEAD)
    grid.add_column("Original Prompt", width=40, justify="right", style="magenta")
    grid.add_column("Enhanced Prompt", width=60, justify="left", style="green")
    with Live(grid, refresh_per_second=1):
        for idx, naive_prompt in enumerate(EXAMPLE_NAIVE_USER_PROMPTS, 1):
            enhanced_prompt_dict = enhance_user_prompt(naive_prompt, trigger_words="dspy demo")
            grid.add_row(naive_prompt, json.dumps(enhanced_prompt_dict, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance a user prompt for photorealistic image generation.")
    parser.add_argument("user_prompt", type=str, nargs="?", default=None, help="The user's high-level prompt.")
    parser.add_argument("--trigger-words", type=str, help="Optional trigger words to include in the prompt.")

    args = parser.parse_args()

    if args.user_prompt:
        enhanced_prompt = enhance_user_prompt(args.user_prompt, args.trigger_words)
        print(json.dumps(enhanced_prompt, indent=4))
    else:
        run_examples()
