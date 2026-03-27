#!/usr/bin/env python3
"""Karaoke app — convert a YouTube music video into a karaoke video."""

import argparse
from pathlib import Path

from karaoke.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Generate a karaoke video from a YouTube music video."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: ./output)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    run(args.url, output_dir)


if __name__ == "__main__":
    main()
