import sys
from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.utils.info_util import InfoUtil
from mflux.utils.metadata_reader import MetadataReader


def main():
    # Parse command line arguments
    parser = CommandLineParser(description="Display metadata from MFLUX generated images")
    parser.add_info_arguments()
    args = parser.parse_args()

    # Check if file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Read metadata
    metadata = MetadataReader.read_all_metadata(image_path)

    # Check if metadata was found
    if not metadata or (not metadata.get("exif") and not metadata.get("xmp")):
        print("No metadata found")
        sys.exit(1)

    # Format and display
    print(InfoUtil.format_metadata(metadata))


if __name__ == "__main__":
    main()
