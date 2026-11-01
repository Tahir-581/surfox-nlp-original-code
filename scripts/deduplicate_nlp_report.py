#!/usr/bin/env python3
"""
Report fuzzy duplicate NLP pairs from merge_output.entities in session JSON files.

Usage:
    python deduplicate_nlp_report.py
    python deduplicate_nlp_report.py --input-dir . --output-dir duplicates --threshold 70
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

SKIP_FILES = {"_manifest.json", "_user.json"}
SESSION_FILE_PATTERN = re.compile(r"^\d{2}_\d{8}_\d{6}_.+\.json$")


def normalize_text(text: str) -> str:
    normalized = (text or "").lower().strip()
    normalized = re.sub(r"[-_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def similarity_percent(left: str, right: str) -> float:
    return float(fuzz.token_sort_ratio(normalize_text(left), normalize_text(right)))


def entity_sort_key(entity: dict, index: int) -> tuple:
    return (
        float(entity.get("average_relevance") or 0.0),
        float(entity.get("adjusted_weightage") or 0.0),
        int(entity.get("combined_count") or 0),
        -index,
    )


def pick_keep_remove(
    left: dict,
    left_index: int,
    right: dict,
    right_index: int,
) -> tuple[dict, dict]:
    left_key = entity_sort_key(left, left_index)
    right_key = entity_sort_key(right, right_index)
    if left_key >= right_key:
        return left, right
    return right, left


def find_duplicate_pairs(entities: list[dict], threshold: float) -> list[dict]:
    pairs: list[dict] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            left = entities[i]
            right = entities[j]
            left_text = left.get("text") or ""
            right_text = right.get("text") or ""
            if not left_text or not right_text:
                continue

            score = similarity_percent(left_text, right_text)
            if score < threshold:
                continue

            keep, remove = pick_keep_remove(left, i, right, j)
            pairs.append(
                {
                    "similarity_percent": round(score, 1),
                    "keep": keep.get("text") or "",
                    "remove": remove.get("text") or "",
                }
            )

    pairs.sort(key=lambda item: (-item["similarity_percent"], item["keep"]))
    return pairs


def is_session_json(path: Path) -> bool:
    return path.name not in SKIP_FILES and SESSION_FILE_PATTERN.match(path.name) is not None


def process_file(input_path: Path, output_path: Path, threshold: float) -> dict:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    entities = (data.get("merge_output") or {}).get("entities") or []
    if not entities:
        raise ValueError("missing merge_output.entities")

    duplicates = find_duplicate_pairs(entities, threshold)
    report = {
        "source_file": input_path.name,
        "duplicate_pair_count": len(duplicates),
        "duplicates": duplicates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return report


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Find fuzzy duplicate NLP pairs in merge_output.entities and write reports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Folder containing session JSON files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "duplicates",
        help="Folder for duplicate report JSON files (default: duplicates/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="Minimum fuzzy similarity percent to flag a duplicate pair (default: 70)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    threshold = float(args.threshold)

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    session_files = sorted(path for path in input_dir.glob("*.json") if is_session_json(path))
    if not session_files:
        print(f"No session JSON files found in {input_dir}", file=sys.stderr)
        return 1

    processed = 0
    failed = 0
    total_pairs = 0

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {threshold}%")
    print(f"Files: {len(session_files)}\n")

    for input_path in session_files:
        output_path = output_dir / input_path.name
        try:
            report = process_file(input_path, output_path, threshold)
            processed += 1
            pair_count = report["duplicate_pair_count"]
            total_pairs += pair_count
            print(f"  OK  {input_path.name} -> {output_path.name} ({pair_count} pairs)")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {input_path.name}: {exc}", file=sys.stderr)

    print(
        f"\nDone. Processed {processed}/{len(session_files)} files, "
        f"{total_pairs} duplicate pairs total, {failed} failed."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
