# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow",
#   "keybert",
# ]
# ///

import ast
import json
from pathlib import Path

from keybert import KeyBERT
from PIL import Image, UnidentifiedImageError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXIF_USER_COMMENT_KEY = 37510


class UnsupportedMetadata(Exception):
    pass


def parse_metadata_from_image(image: Image, insecure=False) -> dict:
    try:
        exif_data = image._getexif()
    except AttributeError:
        # when exif data is not available for some image type/format
        return None

    metadata = None
    if exif_data:
        try:
            for tag, value in exif_data.items():
                if tag == EXIF_USER_COMMENT_KEY:
                    metadata = json.loads(value.decode())
        except KeyError:
            metadata = None
        except json.decoder.JSONDecodeError:
            if not insecure:
                raise UnsupportedMetadata(
                    "The metadata is likely stored as a str literal of a Python dict before mflux 0.6.0. "
                    "This tool cannot guarantee safety of running eval(...) on the value. "
                    "However, you can bypass this caution by passing the --insecure flag. "
                    f"Metadata: {value.decode()}"
                )
            try:
                # try to parse the str(dict) data
                obj = ast.literal_eval(value.decode())
                if isinstance(obj, dict):
                    metadata = obj
                else:
                    raise UnsupportedMetadata(
                        f"The metadata is not a dict output recognized by this tool. Metadata: {value.decode()}"
                    )
            except (ValueError, SyntaxError):
                raise UnsupportedMetadata(f"The metadata is not parseable by this tool. Metadata: {value.decode()}")

    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Use mflux metadata to rename images.")
    parser.add_argument("paths", nargs="+", help="mflux image files or directories to process")
    parser.add_argument(
        "--n-keywords", type=int, default=5, help="N number of keywords to extract from each prompt. Default 5."
    )
    parser.add_argument(
        "--yes", action="store_true", default=False, help="Allow renaming without interactive confirmation."
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=False,
        help="At your own risk, allow insecure parsing of literal Python values saved by mflux versions < 0.6.0",
    )
    args = parser.parse_args()

    if not args.yes:
        print(
            "INFO: This tool by default dry runs the file rename. To automatically accept renames, pass the `--yes` flag."
        )

    processed_paths = []
    for path in args.paths:
        p = Path(path)
        if p.is_file():
            processed_paths.append(p)
        elif p.is_dir():
            processed_paths.extend(p.glob("*"))

    keyword_model = KeyBERT()
    unsupported_errors: dict[str, UnsupportedMetadata] = {}
    for p in processed_paths:
        if p.is_dir():
            continue
        try:
            with Image.open(p) as image:
                image.verify()  # Verify that it is an image
        except UnidentifiedImageError:
            # expected when receiving dir paths, ignore all non-images in dirs
            continue

        try:
            mflux_metadata = parse_metadata_from_image(image, insecure=args.insecure)
            if not mflux_metadata:
                unsupported_errors[p.as_posix()] = "metadata not stored in EXIF"
                continue
            prompt_keywords = keyword_model.extract_keywords(mflux_metadata["prompt"], top_n=args.n_keywords)
            proposed_new_stem = f"{'_'.join([word for word, _ in prompt_keywords])}__seed_{mflux_metadata['seed']}"
            new_path = p.with_stem(proposed_new_stem)
            if args.yes:
                old_path = p
                p.rename(new_path)
                print(f"File renamed: {old_path} -> {new_path}")
            else:
                print(f"File rename proposed: {p} -> {new_path}")
        except UnsupportedMetadata as ume:
            unsupported_errors[p.as_posix()] = ume

    for path, error in unsupported_errors.items():
        print(f"{path}: {error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
