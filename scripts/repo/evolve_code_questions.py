#!/usr/bin/env python3
"""
Deterministic question-evolution tool for **code-unit** training records.

Mirrors scripts/evolve_questions.py (the API-path analog) but replaces the
LLM mutation with deterministic, offline, template-based phrasing along three
code-appropriate axes:

  1. paraphrase   — alternate natural phrasing of the same ask
  2. context      — add file/module and, for methods, class context
  3. task_pattern — reframe as an implementation/usage task

Output records carry all original fields plus:
  "source":    "evol"
  "evol_axis": "paraphrase" | "context" | "task_pattern"
  "evol_seed": <original question>

No LLM, no network, no API keys.  All output is deterministic.

Usage:
    python scripts/repo/evolve_code_questions.py \\
        --input-dir data/repos/<repo-name> \\
        --per-record 2 \\
        [--dry-run] [--sample 5]

    python scripts/repo/evolve_code_questions.py \\
        --input data/repos/<repo-name>/training.jsonl \\
        --per-record 2 \\
        [--dry-run] [--sample 5]
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Axis implementations — pure deterministic functions (record) -> str | None
# ---------------------------------------------------------------------------

_AXES = ("paraphrase", "context", "task_pattern")


def _axis_paraphrase(record: dict) -> str | None:
    """Return an alternate phrasing of the question, or None if unchanged.

    The paraphrase is derived deterministically from the output fields.  We
    produce a stable alternate phrasing that differs from the seed question.
    """
    output = record.get("output", {})
    unit = output.get("unit", "function")
    name = output.get("name", "")
    cls = output.get("class", "")
    seed_q = record.get("question", "")

    if unit == "function":
        candidates = [
            f"What is the signature of `{name}`?",
            f"How do I call `{name}`?",
            f"Show me an example of using `{name}`",
            f"Call the `{name}` function",
        ]
    elif unit == "method":
        if cls:
            candidates = [
                f"How do I invoke `{cls}.{name}`?",
                f"How do I call `{name}` on a `{cls}`?",
                f"Use the `{name}` method of `{cls}`",
                f"Show me how to call `{name}` on `{cls}`",
            ]
        else:
            candidates = [
                f"What is the signature of `{name}`?",
                f"How do I call `{name}`?",
                f"Show me an example of using `{name}`",
            ]
    elif unit == "class":
        candidates = [
            f"Create an instance of `{name}`",
            f"How do I instantiate `{name}`?",
            f"How do I use the `{name}` class?",
            f"Show me how to construct a `{name}` object",
        ]
    elif unit == "api_call":
        candidates = [
            f"Show me how to use `{name}`",
            f"How is the `{name}` API call made?",
            f"Demonstrate the `{name}` API call",
            f"How do I make a `{name}` request?",
        ]
    else:
        candidates = [
            f"What is the signature of `{name}`?",
            f"How do I call `{name}`?",
            f"Show me an example of using `{name}`",
        ]

    # Pick the first candidate that differs from the seed question.
    for candidate in candidates:
        if candidate != seed_q:
            return candidate

    return None


def _axis_context(record: dict) -> str | None:
    """Return a question enriched with file/module and class context, or None.

    Adds deterministic code context drawn from the output fields: the file
    (module path) and, for methods, the class name.  Returns None if the
    produced question would equal the seed.
    """
    output = record.get("output", {})
    unit = output.get("unit", "function")
    name = output.get("name", "")
    cls = output.get("class", "")
    file_path = output.get("file", "")
    seed_q = record.get("question", "")

    # Derive a short module reference from the file path.
    if file_path:
        # Strip leading path separators and use the file stem as module hint.
        p = Path(file_path)
        module = p.stem  # e.g. "generate_from_code" from "scripts/repo/generate_from_code.py"
    else:
        module = ""

    if unit == "method" and cls:
        if module:
            new_q = f"How do I call the `{name}` method on `{cls}` (defined in {module})?"
        else:
            new_q = f"How do I call the `{name}` method on `{cls}`?"
    elif unit == "function":
        if module:
            new_q = f"How do I use `{name}` from {module}?"
        else:
            new_q = f"How do I use `{name}`?"
    elif unit == "class":
        if module:
            new_q = f"How do I instantiate `{name}` from {module}?"
        else:
            new_q = f"How do I instantiate `{name}`?"
    elif unit == "api_call":
        if module:
            new_q = f"How is the `{name}` API call made in {module}?"
        else:
            new_q = f"How is the `{name}` API call made?"
    else:
        if module:
            new_q = f"How do I use `{name}` in {module}?"
        else:
            new_q = f"How do I use `{name}`?"

    if new_q == seed_q:
        return None
    return new_q


def _axis_task_pattern(record: dict) -> str | None:
    """Return a task-framed question using pattern vocabulary, or None.

    Reframes the question as an implementation/usage task, aligning with
    _pattern_implement / _pattern_call_helper framing from generate_from_code.
    """
    output = record.get("output", {})
    unit = output.get("unit", "function")
    name = output.get("name", "")
    cls = output.get("class", "")
    sig = output.get("signature", "")
    seed_q = record.get("question", "")

    call_form = sig if sig else f"{name}(...)"

    if unit == "function":
        new_q = f"Implement `{call_form}`"
    elif unit == "method":
        if cls:
            new_q = f"Call the helper `{name}` on a `{cls}` instance"
        else:
            new_q = f"Implement `{call_form}`"
    elif unit == "class":
        new_q = f"Implement and instantiate `{name}`"
    elif unit == "api_call":
        new_q = f"Make the `{name}` API call"
    else:
        new_q = f"Implement `{call_form}`"

    if new_q == seed_q:
        return None
    return new_q


# ---------------------------------------------------------------------------
# Axis dispatch table
# ---------------------------------------------------------------------------

_AXIS_FNS = {
    "paraphrase": _axis_paraphrase,
    "context": _axis_context,
    "task_pattern": _axis_task_pattern,
}


# ---------------------------------------------------------------------------
# Record evolution
# ---------------------------------------------------------------------------

def _evolve_record(record: dict, axis: str) -> dict | None:
    """Return an evolved record for the given axis, or None.

    Mirrors evolve_questions._evolve_record but uses a deterministic template
    generator instead of the Claude API.
    """
    seed_q = record.get("question", "")
    fn = _AXIS_FNS[axis]
    new_q = fn(record)

    if not new_q or new_q == seed_q:
        return None

    evolved = {**record, "question": new_q}
    evolved["source"] = "evol"
    evolved["evol_axis"] = axis
    evolved["evol_seed"] = seed_q
    return evolved


# ---------------------------------------------------------------------------
# File-level evolution (mirrors evolve_questions.evolve_file)
# ---------------------------------------------------------------------------

def evolve_file(
    input_path: Path,
    output_path: Path,
    per_record: int = 2,
    dry_run: bool = False,
    sample: int | None = None,
    seed: int = 42,
) -> int:
    """Evolve questions in input_path and append to output_path.

    Args:
        input_path:  Path to a JSONL file of code-unit training records.
        output_path: Path to append evolved records to (may equal input_path).
        per_record:  Number of axis mutations to produce per record (1–3).
        dry_run:     If True, print mutations without writing anything.
        sample:      If given, only evolve a random sample of N records.
        seed:        RNG seed for reproducibility (default 42, same as analog).

    Returns:
        Number of evolved records written (or that would have been written).
    """
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

    axes = list(_AXES)
    written = 0

    for i, record in enumerate(records):
        # Don't evolve already-evolved or bootstrap records (mirrors analog).
        if record.get("source") in ("evol", "bootstrap"):
            continue

        selected_axes = rng.sample(axes, min(per_record, len(axes)))
        for axis in selected_axes:
            evolved = _evolve_record(record, axis)
            if evolved is None:
                continue
            if dry_run:
                print(f"  [{axis}] {record['question']!r}")
                print(f"       → {evolved['question']!r}")
            else:
                with open(output_path, "a") as f:
                    f.write(json.dumps(evolved) + "\n")
            written += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(records)} records processed, {written} evolved")

    return written


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def _discover_training_files(input_dir: Path) -> list[Path]:
    """Return the training.jsonl files to evolve under ``input_dir``.

    Supports both documented layouts:
      * a single repo directory — ``input_dir/training.jsonl`` sits directly
        inside (the code-path layout, e.g. ``data/repos/<repo>``); and
      * a parent-of-repos directory — ``input_dir/<repo>/training.jsonl``
        (e.g. ``data/repos``).

    Globbing only ``*/training.jsonl`` (as the API-path analog does) silently
    misses the direct file, so passing a single repo dir would process zero
    records.  Order is stable and duplicates are removed.
    """
    files: list[Path] = []
    direct = input_dir / "training.jsonl"
    if direct.is_file():
        files.append(direct)
    for nested in sorted(input_dir.glob("*/training.jsonl")):
        if nested not in files:
            files.append(nested)
    return files


# ---------------------------------------------------------------------------
# CLI (mirrors evolve_questions.main)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Repo data directory (e.g. data/repos/<repo-name>)",
    )
    group.add_argument(
        "--input",
        type=Path,
        help="Single JSONL input file",
    )
    parser.add_argument(
        "--per-record",
        type=int,
        default=2,
        help="Axis mutations per seed record (default: 2, max: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mutations without writing",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only evolve a random sample of N records per file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evolve records that already have source=evol (not yet implemented — placeholder for parity with analog)",
    )
    args = parser.parse_args()

    if args.input is not None:
        # Single-file mode.
        jsonl = args.input
        if not jsonl.exists():
            sys.exit(f"Input file not found: {jsonl}")
        print(f"\nProcessing: {jsonl}")
        written = evolve_file(
            input_path=jsonl,
            output_path=jsonl,
            per_record=min(args.per_record, 3),
            dry_run=args.dry_run,
            sample=args.sample,
        )
        print(f"  Written: {written}")
        print(f"\nTotal evolved records: {written}")
        return

    # Directory mode.
    input_dir = args.input_dir
    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")

    training_files = _discover_training_files(input_dir)
    if not training_files:
        print(f"No training.jsonl found under {input_dir}")

    total = 0
    for jsonl in training_files:
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
