#!/usr/bin/env python3
"""Generate training examples JSON from battlemap datasets.

Scans 3 battlemap datasets and creates a unified JSON file with image/prompt pairs:
- battlemap_fantasy_dataset: Uses **FIELD:** markdown format
- battlemap_scifi_dataset: Uses **FIELD:** markdown format
- cze-peku-spaceships: Uses field:_ underscore format (with subdirectories)

Output format matches mflux training spec requirements.
"""

import json
from pathlib import Path

DATASETS = [
    "/Volumes/Seagate1/image-datasets/datasets/battlemap-datasets/battlemap_fantasy_dataset",
    "/Volumes/Seagate1/image-datasets/datasets/battlemap-datasets/battlemap_scifi_dataset",
    "/Volumes/Seagate1/image-datasets/datasets/battlemap-datasets/cze-peku-spaceships",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Fields to extract from markdown format (in priority order for prompt construction)
MARKDOWN_FIELDS = [
    "SUBJECT",
    "ACTION",
    "SETTING",
    "MOOD",
    "STYLE",
    "LIGHTING",
    "COLOR",
    "COMPOSITION",
    "TAGS",
]


def normalize_markdown_caption(text: str) -> str:
    """Convert **FIELD:** markdown format to plain text prompt.

    Example input:
        **TAGS:** battlemap, top_down, gridless...
        **SUBJECT:** fossilized skull monument...
        **ACTION:** rising from the cavern floor...

    Example output:
        fossilized skull monument rising from the cavern floor deep underground cavern...
    """
    fields = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("**") and ":**" in line:
            # Extract field name and content
            field_name = line.split(":**")[0].replace("**", "").strip().upper()
            content = line.split(":**", 1)[-1].strip()
            if content and field_name:
                # Clean up content: remove trailing punctuation, normalize spaces
                content = content.rstrip(".,;:").strip()
                # Replace underscores with spaces (sometimes used in tags)
                content = content.replace("_", " ").replace("  ", " ")
                fields[field_name] = content

    # Build prompt from fields in priority order
    prompt_parts = []

    # Add subject first (the main thing being depicted)
    if "SUBJECT" in fields:
        prompt_parts.append(fields["SUBJECT"])

    # Add action (what's happening)
    if "ACTION" in fields:
        prompt_parts.append(fields["ACTION"])

    # Add setting (where)
    if "SETTING" in fields:
        prompt_parts.append(fields["SETTING"])

    # Add mood/atmosphere
    if "MOOD" in fields:
        prompt_parts.append(f"atmosphere: {fields['MOOD']}")

    # Add style info
    if "STYLE" in fields:
        prompt_parts.append(fields["STYLE"])

    # Add lighting
    if "LIGHTING" in fields:
        prompt_parts.append(f"lighting: {fields['LIGHTING']}")

    # Add tags as comma-separated suffix
    if "TAGS" in fields:
        # Clean up tags
        tags = fields["TAGS"].replace(",", ", ").replace("  ", " ")
        prompt_parts.append(f"tags: {tags}")

    return ", ".join(prompt_parts) if prompt_parts else ""


def normalize_underscore_caption(text: str) -> str:
    """Convert field:_ underscore format to plain text prompt.

    Example input:
        tags:_battlemap, spaceship, deck_plan...
        subject:_this_top_down_battlemap_depicts...
        setting:_set_in_a_high_tech_sci_fi_universe...

    Example output:
        this top down battlemap depicts a sleek fighter class spaceship...
    """
    fields = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if ":_" in line:
            # Extract field name and content
            parts = line.split(":_", 1)
            if len(parts) == 2:
                field_name = parts[0].strip().lower()
                content = parts[1].replace("_", " ").strip()
                # Clean up: remove trailing punctuation
                content = content.rstrip(".,;:").strip()
                if content:
                    fields[field_name] = content

    # Build prompt from fields
    prompt_parts = []

    # Add subject first
    if "subject" in fields:
        prompt_parts.append(fields["subject"])

    # Add setting
    if "setting" in fields:
        prompt_parts.append(fields["setting"])

    # Add style/perspective
    if "style" in fields:
        prompt_parts.append(fields["style"])
    if "perspective" in fields:
        prompt_parts.append(fields["perspective"])

    # Add tags
    if "tags" in fields:
        tags = fields["tags"].replace(",", ", ").replace("  ", " ")
        prompt_parts.append(f"tags: {tags}")

    return ", ".join(prompt_parts) if prompt_parts else ""


def detect_caption_format(text: str) -> str:
    """Detect whether caption uses markdown (**FIELD:**) or underscore (field:_) format."""
    if "**" in text and ":**" in text:
        return "markdown"
    elif ":_" in text:
        return "underscore"
    else:
        # Fallback: treat as raw text
        return "raw"


def normalize_caption(text: str) -> str:
    """Normalize any caption format to plain text prompt."""
    format_type = detect_caption_format(text)

    if format_type == "markdown":
        return normalize_markdown_caption(text)
    elif format_type == "underscore":
        return normalize_underscore_caption(text)
    else:
        # Raw text: just clean it up
        return text.strip().replace("\n", " ").replace("  ", " ")


def scan_dataset(dataset_path: str) -> list[dict]:
    """Scan a dataset directory for image/caption pairs.

    Handles both flat directories and nested subdirectories.
    """
    examples = []
    dataset = Path(dataset_path)

    if not dataset.exists():
        print(f"  WARNING: Dataset path does not exist: {dataset_path}")
        return examples

    # Use rglob to find all images recursively
    for img_path in dataset.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Skip __pycache__ and hidden directories
        if "__pycache__" in str(img_path) or "/." in str(img_path):
            continue

        # Find corresponding caption file
        caption_path = img_path.with_suffix(".txt")
        if not caption_path.exists():
            continue

        try:
            caption = caption_path.read_text(encoding="utf-8", errors="ignore")
            prompt = normalize_caption(caption)

            if prompt and len(prompt) > 20:  # Minimum prompt length
                examples.append({"image": str(img_path), "prompt": prompt})
        except OSError as e:
            print(f"  Error reading {caption_path}: {e}")

    return examples


def main():
    all_examples = []

    # Base path for all datasets - this is the common parent
    BASE_PATH = "/Volumes/Seagate1/image-datasets/datasets/battlemap-datasets"

    print("Scanning battlemap datasets...")
    print("=" * 60)

    for dataset in DATASETS:
        dataset_name = Path(dataset).name
        print(f"\n[{dataset_name}]")
        print(f"  Path: {dataset}")

        examples = scan_dataset(dataset)
        print(f"  Found: {len(examples)} image/caption pairs")

        if examples:
            # Show sample prompt
            sample = examples[0]
            print(f"  Sample prompt ({Path(sample['image']).name}):")
            print(f"    {sample['prompt'][:150]}...")

        all_examples.extend(examples)

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_examples)} training examples")

    # Convert absolute paths to relative paths from BASE_PATH
    # The training spec security check requires relative paths (no leading /)
    relative_examples = []
    for ex in all_examples:
        abs_path = ex["image"]
        # Make path relative to BASE_PATH
        rel_path = str(Path(abs_path).relative_to(BASE_PATH))
        relative_examples.append({"image": rel_path, "prompt": ex["prompt"]})

    # Save to JSON with the base path
    output = {"path": BASE_PATH, "images": relative_examples}

    output_path = Path(__file__).parent / "battlemap_examples.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Base path: {BASE_PATH}")
    print(f"Sample relative path: {relative_examples[0]['image']}")


if __name__ == "__main__":
    main()
