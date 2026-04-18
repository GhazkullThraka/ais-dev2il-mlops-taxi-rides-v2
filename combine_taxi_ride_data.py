from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("data")
OUTPUT_FILE = Path("data/taxi-rides-training-data.parquet")
OVERWRITE = True


def find_input_files(input_dir: Path, dates: list[str] | None = None) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if dates:
        pattern_str = "|".join(re.escape(d) for d in dates)
        filename_pattern = re.compile(rf"(?:{pattern_str})\.taxi-rides\.parquet$")
    else:
        filename_pattern = re.compile(r"\d{4}-\d{2}-\d{2}\.taxi-rides\.parquet$")

    matches = [
        path for path in input_dir.iterdir() if path.is_file() and filename_pattern.fullmatch(path.name)
    ]
    return sorted(matches)


def combine_parquet_files(input_files: list[Path]) -> pd.DataFrame:
    if not input_files:
        raise ValueError("No input parquet files matched the expected filename pattern.")

    frames = [pd.read_parquet(path) for path in input_files]
    return pd.concat(frames, ignore_index=True, sort=False)


def main(dates: list[str] | None = None) -> None:
    if OUTPUT_FILE.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output file already exists: {OUTPUT_FILE}."
        )

    input_files = find_input_files(INPUT_DIR, dates)
    combined = combine_parquet_files(input_files)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(OUTPUT_FILE))

    print(f"Input files: {len(input_files)}")
    print(f"Rows written: {len(combined)}")
    print(f"Output: {OUTPUT_FILE}")

    print(combined.describe(include="all"))


if __name__ == "__main__":
    dates = sys.argv[1:] if len(sys.argv) > 1 else None
    main(dates)
