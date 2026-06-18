#!/usr/bin/env python3
"""
Generate router classifier training data from code training.jsonl file(s).

For each record in the input, the route_key is the relative module/file path
stored in record["output"]["file"].  This is the "which file does this question
target" analog of the API-path router's "vendor/api/name" route_key.

Records lacking a 'question' or a 'output.file' field are skipped (counted).

Output files are written to <out-dir>/router_train.jsonl and
<out-dir>/router_test.jsonl using a global seeded shuffle + 80/20 split
(SEED=42, TRAIN_RATIO=0.8), identical to the API-path analog gen_router_data.py.

NOTE: with a global shuffle and a small dataset, a file path (route_key) that
appears in very few records can land entirely in the test split. The downstream
trainer (train_code_router.py) will exit if such unseen labels are found in the
test set. This mirrors the known behaviour of the API analog. For small corpora
consider stratified splitting (or increasing data volume) to avoid this.

Usage:
    # Single file
    python scripts/repo/gen_code_router_data.py --input data/repos/<name>/training.jsonl

    # Directory of training.jsonl files (recurse one level)
    python scripts/repo/gen_code_router_data.py --input-dir data/repos/<name>

    # Both --input and --input-dir can be combined.
    # Output directory (default: data/repos/router)
    python scripts/repo/gen_code_router_data.py --input ... --out-dir data/repos/<name>/router
"""

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).parents[2]

TRAIN_RATIO = 0.8
SEED = 42


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministically shuffle and split router records into (train, test).

    Uses the module constants SEED (=42) and TRAIN_RATIO (=0.8). Operates on a
    copy so the caller's list is never mutated. Same input → identical output
    on every call and across processes. Mirrors gen_router_data.py's faithful
    global shuffle + ratio split.
    """
    shuffled = list(records)
    random.seed(SEED)
    random.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_RATIO)
    return shuffled[:split], shuffled[split:]


def iter_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping malformed line {lineno} in {path}: {exc}", file=sys.stderr)


def collect_records(paths: list[Path]) -> tuple[list[dict], int]:
    """Read all input paths and return (router_records, skip_count).

    A router record is {question, route_key} where route_key = output["file"].
    Records missing 'question' or 'output'/'output.file' are skipped.
    """
    all_records: list[dict] = []
    skip_count = 0

    for path in paths:
        for rec in iter_jsonl(path):
            question = rec.get("question")
            output = rec.get("output") or {}
            file_key = output.get("file") if isinstance(output, dict) else None
            if not question or not file_key:
                skip_count += 1
                continue
            all_records.append({"question": question, "route_key": file_key})

    return all_records, skip_count


def resolve_input_paths(input_file: str | None, input_dir: str | None) -> list[Path]:
    """Return list of Path objects to read from --input and/or --input-dir."""
    paths: list[Path] = []

    if input_file:
        p = Path(input_file)
        if not p.exists():
            sys.exit(f"Input file not found: {p}")
        paths.append(p)

    if input_dir:
        d = Path(input_dir)
        if not d.is_dir():
            sys.exit(f"Input directory not found: {d}")
        found = sorted(d.glob("*.jsonl")) + sorted(d.glob("**/*.jsonl"))
        # Deduplicate while preserving order
        seen: set[Path] = set()
        for p in found:
            if p not in seen:
                seen.add(p)
                paths.append(p)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a single code training.jsonl file.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing training.jsonl file(s) (searched recursively).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for router_train.jsonl / router_test.jsonl "
             "(default: data/repos/router relative to repo root).",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("At least one of --input or --input-dir is required.")

    paths = resolve_input_paths(args.input, args.input_dir)
    if not paths:
        sys.exit("No .jsonl files found in the specified location(s).")

    print(f"Reading {len(paths)} input file(s)...")
    all_records, skip_count = collect_records(paths)

    if skip_count:
        print(f"  Skipped {skip_count} record(s) missing question or output.file")

    if not all_records:
        sys.exit("No valid records found — nothing to write.")

    # Seeded global shuffle then split (mirrors gen_router_data.py exactly)
    train, test = split_records(all_records)

    out_dir = Path(args.out_dir) if args.out_dir else _REPO / "data" / "repos" / "router"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "router_train.jsonl"
    test_path = out_dir / "router_test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec) + "\n")

    # Per-route counts (over entire dataset, like the analog)
    route_counts: dict[str, int] = {}
    for rec in all_records:
        route_counts[rec["route_key"]] = route_counts.get(rec["route_key"], 0) + 1

    print(f"\nPer-route record counts:")
    for route, count in sorted(route_counts.items()):
        print(f"  {route}: {count}")

    print(f"\nTotal: {len(all_records)} records — {len(train)} train / {len(test)} test")
    print(f"Routes: {len(route_counts)}")
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")


if __name__ == "__main__":
    main()
