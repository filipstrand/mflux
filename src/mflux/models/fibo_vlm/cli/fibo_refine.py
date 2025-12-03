from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil
from mflux.utils.exceptions import PromptFileReadError


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Refine FIBO JSON prompts using VLM.")
    # fmt: off
    parser.add_argument("--prompt-file", type=Path, required=True, help="Path to JSON prompt file to refine")
    parser.add_argument("--instructions", type=str, required=True, help="Text instructions for how to refine the prompt (e.g., 'make the dragon blue')")
    parser.add_argument("--output", type=Path, default=Path("refined.json"), help="Output path for refined JSON prompt (default: refined.json)")
    parser.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8], default=None, help="Quantize the VLM model (3, 4, 5, 6, or 8 bits)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling for VLM (default: 0.9)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for VLM (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for VLM generation (default: 4096)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for VLM generation")
    # fmt: on
    args = parser.parse_args()

    try:
        # 1. Refine the JSON prompt
        vlm = FiboVLM(quantize=args.quantize)
        refined_json = vlm.refine(
            seed=args.seed,
            structured_prompt=FiboVLMUtil.get_structured_prompt(args.prompt_file),
            editing_instructions=args.instructions,
            top_p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # 2. Parse and save refined JSON prompt
        FiboVLMUtil.save_json_prompt(args.output, refined_json)
    except (PromptFileReadError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
