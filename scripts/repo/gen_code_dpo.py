#!/usr/bin/env python3
"""
DPO (preference pair) generator for code units.

Generates (chosen, rejected) preference pairs from an existing code
training.jsonl produced by generate_from_code.py.  The Phase-3 deterministic
validators in scripts/eval.py serve as the offline reward function — no live
API is required.

For each record the existing correct ``output`` dict is the *chosen* side.
Deliberately corrupted variants are produced deterministically via
``_generate_rejected_code_outputs``.  A candidate survives as a valid *rejected*
negative when it is a strictly-worse answer than the gold output — that is, it
differs from ``chosen`` AND it is EITHER malformed (fails the structural/syntax
verifier ``_output_is_valid``) OR semantically wrong (its ``field_accuracy``
against the gold output is < 1.0).  This keeps semantic corruptions (wrong
name/file/class/unit) as negatives, not just structurally-broken ones, so the
dataset teaches SEMANTIC correctness rather than mere structural validity.  The
first surviving candidate in strategy order is written per record and the
process continues — mirroring ``gen_dpo.gen_dpo``.

Usage:
    python scripts/repo/gen_code_dpo.py --input data/repos/<repo>/training.jsonl
                                         --output data/repos/<repo>/dpo.jsonl
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Make scripts/ importable so we can reach eval.py from scripts/repo/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval import code_field_accuracy, code_format_score, code_signature_valid  # noqa: E402

_VALID_UNIT_TYPES = ("function", "method", "class", "api_call")


# ── Verifier (replaces _api_ok) ────────────────────────────────────────────

def _output_is_valid(output: object) -> bool:
    """Return True iff output passes both the format check and signature check.

    Verifier predicate:
        code_format_score(output) == 1.0  AND  code_signature_valid(output) is True
    """
    return code_format_score(output) == 1.0 and code_signature_valid(output) is True


def _is_valid_negative(rejected: object, chosen: dict) -> bool:
    """Return True iff ``rejected`` is a strictly-worse answer than the gold ``chosen``.

    Keep-rule predicate (the contract a written DPO negative must satisfy):

        rejected != chosen  AND  NOT( _output_is_valid(rejected)
                                      AND field_accuracy(rejected, chosen) == 1.0 )

    Equivalently, a candidate survives when it differs from gold AND is EITHER
    malformed (fails ``_output_is_valid`` — a structural/syntax negative) OR
    semantically wrong (``code_field_accuracy < 1.0`` — a content negative).
    A candidate that is both structurally valid AND field-identical to gold is
    NOT a worse answer and is rejected as a negative.
    """
    if rejected == chosen:
        return False
    structurally_perfect = _output_is_valid(rejected)
    field_perfect = (
        code_field_accuracy(rejected, chosen).get("field_accuracy") == 1.0
    )
    # Strictly worse iff NOT (structurally perfect AND field-perfect).
    return not (structurally_perfect and field_perfect)


# ── Rejected candidate generation ─────────────────────────────────────────

def _generate_rejected_code_outputs(output: dict) -> list[dict]:
    """Generate deliberately corrupted output dicts for use as rejected candidates.

    Each strategy mutates exactly ONE aspect of the chosen output so the
    corruption is targeted and deterministic.  The same input always yields
    the same list (no randomness), and candidate order is stable so the caller's
    first-surviving-candidate choice is deterministic.

    Every strategy changes a field relative to the gold output, so each one can
    yield a surviving DPO negative under the keep-rule in ``_is_valid_negative``:
    a candidate survives if it differs from gold AND is either malformed
    (structural negative) or semantically wrong (``field_accuracy`` < 1.0).
    Strategies 1–3 and 6 are SEMANTIC corruptions (structurally valid but wrong
    content); strategies 4–5 are STRUCTURAL corruptions (malformed output).

    Strategies
    ----------
    1. Wrong unit type   — swap to the next valid unit type in the cycle
                           (semantic: structurally valid, wrong ``unit``).
    2. Perturbed name    — append ``_TYPO`` to the name
                           (semantic: structurally valid, wrong ``name``).
    3. Wrong file path   — replace the file value with a clearly wrong path
                           (semantic: structurally valid, wrong ``file``).
    4. Garbled signature — produce a syntactically invalid signature (unbalanced
                           parens) so ``code_signature_valid`` returns False
                           (structural: malformed).
    5. Dropped key       — remove the ``signature`` key entirely so
                           ``code_format_score`` returns 0.0 (structural: malformed).
    6. Wrong/missing class (only when the chosen output carries a class key) —
                           first candidate changes it to a wrong value, second
                           candidate drops it altogether (semantic: structurally
                           valid, wrong/absent ``class``).
    """
    candidates: list[dict] = []

    # Strategy 1: wrong unit type
    current_unit = output.get("unit", "function")
    idx = _VALID_UNIT_TYPES.index(current_unit) if current_unit in _VALID_UNIT_TYPES else 0
    wrong_unit = _VALID_UNIT_TYPES[(idx + 1) % len(_VALID_UNIT_TYPES)]
    bad = copy.deepcopy(output)
    bad["unit"] = wrong_unit
    candidates.append(bad)

    # Strategy 2: perturbed name
    bad = copy.deepcopy(output)
    bad["name"] = str(output.get("name", "")) + "_TYPO"
    candidates.append(bad)

    # Strategy 3: wrong file path
    bad = copy.deepcopy(output)
    bad["file"] = "/wrong/path/does_not_exist.py"
    candidates.append(bad)

    # Strategy 4: garbled signature — unbalanced parens make AST parse fail
    bad = copy.deepcopy(output)
    bad["signature"] = "((( unbalanced"
    candidates.append(bad)

    # Strategy 5: dropped required key (signature)
    bad = copy.deepcopy(output)
    bad.pop("signature", None)
    candidates.append(bad)

    # Strategy 6: wrong/missing class (only when the output has a class key)
    if "class" in output:
        # 6a: change class to a wrong value
        bad = copy.deepcopy(output)
        bad["class"] = "WrongClass_TYPO"
        candidates.append(bad)

        # 6b: drop the class key entirely
        bad = copy.deepcopy(output)
        del bad["class"]
        candidates.append(bad)

    return candidates


# ── Record builder ─────────────────────────────────────────────────────────

def _make_code_dpo_record(question: str, chosen: dict, rejected: dict) -> dict:
    """Return a DPO record dict with type, question, chosen, and rejected."""
    return {
        "type": "code",
        "question": question,
        "chosen": chosen,
        "rejected": rejected,
    }


# ── Main generator ─────────────────────────────────────────────────────────

def gen_code_dpo(input_path: Path, output_path: Path, dry_run: bool = False) -> int:
    """Generate DPO pairs from a code training.jsonl.  Returns pairs written."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    written = 0
    skipped = 0

    for record in records:
        # Only process code-type records.
        if record.get("type") != "code":
            skipped += 1
            continue

        chosen = record.get("output")
        question = record.get("question", "")

        if not isinstance(chosen, dict):
            skipped += 1
            continue

        # Validate chosen — if it fails the verifier, skip this record.
        if not _output_is_valid(chosen):
            print(f"  SKIP  chosen invalid for record: {record.get('output')}")
            skipped += 1
            continue

        rejected_list = _generate_rejected_code_outputs(chosen)

        for rejected in rejected_list:
            # Keep a candidate only if it is a strictly-worse answer than gold:
            # it must differ from chosen AND be either malformed (fails the
            # verifier) OR semantically wrong (field_accuracy < 1.0). This keeps
            # semantic corruptions as negatives, not just structural ones.
            if not _is_valid_negative(rejected, chosen):
                continue

            pair = _make_code_dpo_record(question, chosen, rejected)
            if dry_run:
                print(f"  DRY  {question[:60]}")
                print(f"       chosen:   {chosen}")
                print(f"       rejected: {rejected}")
            else:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pair) + "\n")
            written += 1
            break  # One good pair per record is enough

    print(f"\nRecords: {len(records)}, pairs written: {written}, skipped: {skipped}")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a code training.jsonl")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output dpo.jsonl (appended)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing")
    args = parser.parse_args(argv)

    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    gen_code_dpo(args.input, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
