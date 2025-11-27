import json
from pathlib import Path

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.ui.cli.parsers import CommandLineParser
from mflux.utils.exceptions import PromptFileReadError


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Refine FIBO JSON prompts using VLM.")
    # fmt: off
    parser.add_argument("--prompt-file", type=Path, required=True, help="Path to JSON prompt file to refine")
    parser.add_argument("--instructions", type=str, required=True, help="Text instructions for how to refine the prompt (e.g., 'make the dragon blue')")
    parser.add_argument("--output", type=Path, default=Path("refined.json"), help="Output path for refined JSON prompt (default: refined.json)")
    parser.add_argument("--path", type=str, default=None, help="Local path for loading the VLM model from disk")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling for VLM (default: 0.9)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for VLM (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for VLM generation (default: 4096)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for VLM generation")
    # fmt: on
    args = parser.parse_args()

    try:
        # 1. Refine the JSON prompt
        vlm = FiboVLM(local_path=args.path)

        refined_json = vlm.refine(
            structured_prompt=_get_structured_prompt(args),
            editing_instructions=args.instructions,
            top_p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        # 2. Parse and save refined JSON prompt
        _save_prompt(args, refined_json)
    except (PromptFileReadError, ValueError) as exc:
        print(exc)


def _save_prompt(args, refined_json):
    try:
        refined_json_parsed = json.loads(refined_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"VLM did not return valid JSON: {e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(refined_json_parsed, f, indent=2, ensure_ascii=False)


def _get_structured_prompt(args):
    if not args.prompt_file.exists():
        raise PromptFileReadError(f"Prompt file does not exist: {args.prompt_file}")
    with open(args.prompt_file, "rt") as f:
        structured_prompt = f.read().strip()
    if not structured_prompt:
        raise PromptFileReadError(f"Prompt file is empty: {args.prompt_file}")
    try:
        json.loads(structured_prompt)
    except json.JSONDecodeError as e:
        raise PromptFileReadError(f"Prompt file does not contain valid JSON: {e}")
    return structured_prompt


if __name__ == "__main__":
    main()
