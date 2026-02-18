"""
Convert old .txt training data to new .csv format for hierarchical training.

Old format: training/data/processed/{category}.txt (one message per line)
New format: training/data/hierarchical/{category}.csv (message,category_type,subcategory)

Usage:
    python -m training.scripts.convert_old_data
    python -m training.scripts.convert_old_data --output-dir training/data/hierarchical
    python -m training.scripts.convert_old_data --mode append  # append to existing CSVs
"""

import argparse
import csv
import os

# Map old .txt filenames to new category names
FILE_MAPPING = {
    "chitchat.txt": "chitchat",
    "off_topic.txt": "off_topic",
    "rag_queries.txt": "rag_query",
}

INPUT_DIR = os.path.join("training", "data", "processed")
DEFAULT_OUTPUT_DIR = os.path.join("training", "data", "hierarchical")


def convert_txt_to_csv(input_dir, output_dir, mode="write"):
    """
    Convert old .txt files to new flat CSV format.

    Args:
        input_dir: Directory containing old .txt files
        output_dir: Directory to write new .csv files
        mode: "write" to overwrite, "append" to add to existing CSVs
    """
    os.makedirs(output_dir, exist_ok=True)

    total_converted = 0

    for txt_file, category in FILE_MAPPING.items():
        txt_path = os.path.join(input_dir, txt_file)

        if not os.path.exists(txt_path):
            print(f"  SKIP: {txt_path} not found")
            continue

        # Read messages from .txt (one per line)
        with open(txt_path, "r", encoding="utf-8") as f:
            messages = [line.strip() for line in f if line.strip()]

        csv_path = os.path.join(output_dir, f"{category}.csv")

        if mode == "append" and os.path.exists(csv_path):
            # Count existing rows
            with open(csv_path, "r", encoding="utf-8") as f:
                existing_count = sum(1 for _ in f) - 1  # minus header

            # Append to existing CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for msg in messages:
                    writer.writerow([msg, category, category])

            print(f"  APPEND: {txt_file} -> {csv_path} (+{len(messages)} messages, total: {existing_count + len(messages)})")
        else:
            # Write new CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["message", "category_type", "subcategory"])
                for msg in messages:
                    writer.writerow([msg, category, category])

            print(f"  WRITE: {txt_file} -> {csv_path} ({len(messages)} messages)")

        total_converted += len(messages)

    return total_converted


def main():
    parser = argparse.ArgumentParser(description="Convert old .txt training data to new .csv format")
    parser.add_argument("--input-dir", default=INPUT_DIR, help=f"Input directory with .txt files (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory for .csv files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--mode", choices=["write", "append"], default="append",
                        help="write=overwrite CSVs, append=add to existing CSVs (default: append)")
    args = parser.parse_args()

    print("=" * 60)
    print("Converting old .txt data to new .csv format")
    print("=" * 60)
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Mode:   {args.mode}")
    print()

    total = convert_txt_to_csv(args.input_dir, args.output_dir, args.mode)

    print(f"\nDone! Converted {total} messages total.")


if __name__ == "__main__":
    main()
