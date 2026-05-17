#!/usr/bin/env python3
"""
Evol-Instruct question mutation for training data diversification.

Uses Claude API to mutate existing training questions along three axes:
  1. Constraint  — add a realistic constraint ("only the first 10", "sorted by date")
  2. Context     — add domain-specific framing ("for our Q3 analysis")
  3. Complexity  — combine two conditions (multi-condition query)

Model selection:
  - claude-haiku-4-5-20251001   for axis 1 and 2 (simple mutations)
  - claude-sonnet-4-6           for axis 3 (must preserve intent through complexity)

Output records carry all original fields plus:
  "source": "evol"
  "evol_axis": "constraint" | "context" | "complexity"
  "evol_seed": <original question>

Usage:
    python scripts/evolve_questions.py \\
        --input-dir data/<vendor> \\
        --per-record 2 \\
        [--dry-run] [--sample 5]
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import FIELD_QUESTION, FIELD_API_CALL

_REPO = Path(__file__).parents[1]

_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"

_AXIS_MODELS = {
    "constraint": _HAIKU,
    "context": _HAIKU,
    "complexity": _SONNET,
}

_PROMPTS = {
    "constraint": (
        "Rewrite the following API question by adding one realistic constraint "
        "(e.g., a limit, a sort order, a date range). "
        "Keep exactly the same API intent — the same endpoint and parameters must be correct. "
        "Return only the rewritten question, nothing else.\n\n"
        "Original question: {question}\n"
        "API call: {api_call}\n"
        "Rewritten question:"
    ),
    "context": (
        "Rewrite the following API question by adding a brief realistic business context "
        "(e.g., 'for our Q3 report', 'in the dashboard', 'for the media buyer'). "
        "Keep exactly the same API intent — the same endpoint and parameters must be correct. "
        "Return only the rewritten question, nothing else.\n\n"
        "Original question: {question}\n"
        "API call: {api_call}\n"
        "Rewritten question:"
    ),
    "complexity": (
        "Rewrite the following API question in a more complex, natural phrasing that combines "
        "multiple conditions or uses more specific terminology. "
        "Keep exactly the same API intent — the same endpoint and parameters must be correct. "
        "The rewritten question must still be answerable by exactly the same API call. "
        "Return only the rewritten question, nothing else.\n\n"
        "Original question: {question}\n"
        "API call: {api_call}\n"
        "Rewritten question:"
    ),
}


def _call_claude(prompt: str, model: str) -> str:
    """Call the Anthropic Messages API and return the response text."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _evolve_record(record: dict, axis: str) -> dict | None:
    """Return a new record with an evolved question, or None on failure."""
    question = record.get(FIELD_QUESTION, "")
    api_call = record.get(FIELD_API_CALL, {})

    prompt = _PROMPTS[axis].format(
        question=question,
        api_call=json.dumps(api_call),
    )
    model = _AXIS_MODELS[axis]

    try:
        evolved_q = _call_claude(prompt, model)
    except Exception as e:
        print(f"  Claude error ({axis}): {e}")
        return None

    if not evolved_q or evolved_q == question:
        return None

    evolved = {**record, FIELD_QUESTION: evolved_q}
    evolved["source"] = "evol"
    evolved["evol_axis"] = axis
    evolved["evol_seed"] = question
    return evolved


def evolve_file(
    input_path: Path,
    output_path: Path,
    per_record: int = 2,
    dry_run: bool = False,
    sample: int | None = None,
    seed: int = 42,
) -> int:
    """Evolve questions in input_path and append to output_path. Returns count written."""
    records = []
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    if not records:
        return 0

    rng = random.Random(seed)
    if sample:
        records = rng.sample(records, min(sample, len(records)))

    axes = list(_PROMPTS.keys())
    written = 0

    for i, record in enumerate(records):
        # Don't evolve already-evolved records
        if record.get("source") in ("evol", "bootstrap"):
            continue

        selected_axes = rng.sample(axes, min(per_record, len(axes)))
        for axis in selected_axes:
            evolved = _evolve_record(record, axis)
            if evolved is None:
                continue
            if dry_run:
                print(f"  [{axis}] {record[FIELD_QUESTION]!r}")
                print(f"       → {evolved[FIELD_QUESTION]!r}")
            else:
                with open(output_path, "a") as f:
                    f.write(json.dumps(evolved) + "\n")
            written += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(records)} records processed, {written} evolved")

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Vendor data directory (e.g. data/<vendor>)")
    parser.add_argument("--per-record", type=int, default=2,
                        help="Evol mutations per seed record (default: 2, max: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mutations without writing")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only evolve a random sample of N records per file")
    parser.add_argument("--force", action="store_true",
                        help="Re-evolve records that already have source=evol")
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory not found: {args.input_dir}")

    total = 0
    for jsonl in sorted(args.input_dir.glob("*/training.jsonl")):
        print(f"\nProcessing: {jsonl}")
        written = evolve_file(
            input_path=jsonl,
            output_path=jsonl,
            per_record=min(args.per_record, 3),
            dry_run=args.dry_run,
            sample=args.sample,
        )
        total += written
        print(f"  Written: {written}")

    print(f"\nTotal evolved records: {total}")


if __name__ == "__main__":
    main()
