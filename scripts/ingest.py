"""CLI: ingest a directory of markdown / text files into Anchor.

Usage:
    python scripts/ingest.py data/
    python scripts/ingest.py data/ --glob "**/*.txt"
"""

import argparse
import sys
from pathlib import Path

# Allow running as a top-level script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anchor.ingestion.pipeline import IngestionPipeline  # noqa: E402
from anchor.logging_config import configure_logging, get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Anchor.")
    parser.add_argument("directory", type=Path, help="Directory containing source documents.")
    parser.add_argument(
        "--glob",
        default="**/*.md",
        help="Glob pattern for files to ingest (default: **/*.md).",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__)

    if not args.directory.is_dir():
        log.error("not_a_directory", path=str(args.directory))
        return 1

    pipeline = IngestionPipeline()
    total = pipeline.ingest_directory(args.directory, glob=args.glob)
    log.info("ingestion_complete", total_chunks=total, directory=str(args.directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
