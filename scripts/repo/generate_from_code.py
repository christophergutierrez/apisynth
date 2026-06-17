#!/usr/bin/env python3
"""Generate synthetic training records from scanned code units.

Reads a repo.yaml, scans the target repository for code units (functions,
methods, classes, and API call sites), then writes training.jsonl and
holdout.jsonl files under data/repos/<repo-name>/.

Each record carries:
  - type: "code"
  - question: templated natural-language prompt keyed off unit type
  - thinking: deterministic Entity/Scope/Use/… reasoning trace
  - output: structured code-unit descriptor (unit, name, file, [class], [signature])

The train/holdout split is deterministic: each unit's SHA-256 hash of
"<file>:<name>" is mapped into [0, 1) (by dividing the integer digest by
2**256), and the unit is assigned to holdout when that value is below
holdout_ratio. Because SHA-256 is process-independent (unlike the builtin
salted hash()), the same input always produces the same split across runs and
processes — no random seed required. See _split_records() / _in_holdout().

Usage:
    python scripts/repo/generate_from_code.py <path/to/repo.yaml>
    python scripts/repo/generate_from_code.py <path/to/repo.yaml> --dry-run
    python scripts/repo/generate_from_code.py <path/to/repo.yaml> --output-dir data/repos
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure the repo root is on sys.path so that `scripts.*` imports work
# regardless of the working directory when this file is executed directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Question templates keyed by unit type
# ---------------------------------------------------------------------------

# Each entry is a list of phrasings; we pick by stable index (hash % len) so
# the same unit always gets the same question phrasing across runs.
_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "function": [
        "How do I use `{name}`?",
        "Call the `{name}` function",
        "How do I call `{name}`?",
        "Show me an example of using `{name}`",
    ],
    "method": [
        "How do I call `{name}` on a `{class}`?",
        "Use the `{name}` method of `{class}`",
        "How do I invoke `{class}.{name}`?",
        "Show me how to call `{name}` on `{class}`",
    ],
    "class": [
        "How do I instantiate `{name}`?",
        "Create an instance of `{name}`",
        "How do I use the `{name}` class?",
        "Show me how to construct a `{name}` object",
    ],
    "api_call": [
        "How is the `{name}` API call made?",
        "Show me how to use `{name}`",
        "Demonstrate the `{name}` API call",
        "How do I make a `{name}` request?",
    ],
}


def _pick_question(unit: Dict[str, Any]) -> str:
    """Return a deterministic question string for the given unit."""
    unit_type = unit["type"]
    templates = _QUESTION_TEMPLATES.get(unit_type, _QUESTION_TEMPLATES["function"])

    # Use a stable hash to pick the phrasing so same unit → same question.
    key = f"{unit['file']}:{unit['name']}"
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(templates)
    template = templates[idx]

    # Always supply `class` (defaulting to "") so the format call is total even
    # for hand-built method units that omit the key. Non-method templates simply
    # ignore the unused field.
    return template.format(
        name=unit["name"],
        **{"class": unit.get("class", "")},
    )


# ---------------------------------------------------------------------------
# Thinking trace generators (deterministic, Entity/Scope/Use/NOT style)
# ---------------------------------------------------------------------------

def _signature_for(unit: Dict[str, Any]) -> str:
    """Return a best-effort signature string.

    scan_repo does not store full argument lists — only name, type, file, lineno,
    and (for methods) class. We synthesise a plausible signature from context:
      - function / method: "<name>(…)" with a stub params note
      - api_call: "<receiver>.<method>(url, **kwargs)"
      - class: "<name>()"
    """
    name = unit["name"]
    unit_type = unit["type"]
    if unit_type == "function":
        return f"{name}(...)"
    if unit_type == "method":
        return f"{name}(self, ...)"
    if unit_type == "class":
        return f"{name}()"
    if unit_type == "api_call":
        return f"{name}(url, **kwargs)"
    return f"{name}(...)"


def _make_thinking(unit: Dict[str, Any]) -> str:
    """Return a deterministic thinking trace for the unit (Entity/Scope/Use/NOT style)."""
    name = unit["name"]
    unit_type = unit["type"]
    file_path = unit["file"]
    cls = unit.get("class", "")
    sig = _signature_for(unit)

    if unit_type == "function":
        return (
            f"Entity: function {name}\n"
            f"File: {file_path}\n"
            f"Scope: single unit — top-level function\n"
            f"Use: call {sig}\n"
            f"Params: see function signature in {file_path}"
        )

    if unit_type == "method":
        return (
            f"Entity: method {name} on class {cls}\n"
            f"File: {file_path}\n"
            f"Scope: single unit — instance method\n"
            f"Use: instance.{sig}\n"
            f"NOT: calling {name}() as a standalone function"
        )

    if unit_type == "class":
        return (
            f"Entity: class {name}\n"
            f"File: {file_path}\n"
            f"Scope: single unit — class definition\n"
            f"Use: instantiate with {sig}\n"
            f"Params: see __init__ in {file_path}"
        )

    if unit_type == "api_call":
        return (
            f"Entity: api_call {name}\n"
            f"File: {file_path}\n"
            f"Scope: single call site — HTTP/API invocation\n"
            f"Use: {sig}\n"
            f"NOT: a plain dict.get() or queue.get() — this is an HTTP/API call"
        )

    # Fallback (should not occur with known scanner types)
    return (
        f"Entity: {unit_type} {name}\n"
        f"File: {file_path}\n"
        f"Scope: single unit"
    )


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _make_record(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a scanner unit into a training record."""
    output: Dict[str, Any] = {
        "unit": unit["type"],
        "name": unit["name"],
        "file": unit["file"],
        "signature": _signature_for(unit),
    }
    # Methods carry the parent class name.
    if unit.get("class"):
        output["class"] = unit["class"]

    return {
        "type": "code",
        "question": _pick_question(unit),
        "thinking": _make_thinking(unit),
        "output": output,
    }


# ---------------------------------------------------------------------------
# Holdout split — deterministic, no random
# ---------------------------------------------------------------------------

def _in_holdout(unit: Dict[str, Any], holdout_ratio: float) -> bool:
    """Return True if the unit belongs to the holdout split.

    Uses a stable SHA-256 hash of (file, name) so the assignment is identical
    across runs regardless of insertion order. The hash integer is mapped to
    [0, 1) by dividing by 2**256; the unit goes to holdout when the value
    falls below holdout_ratio.
    """
    key = f"{unit['file']}:{unit['name']}"
    digest = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    # Map to [0, 1)
    value = digest / (2 ** 256)
    return value < holdout_ratio


def _split_records(
    units: List[Dict[str, Any]], holdout_ratio: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition units into (train, holdout) records deterministically."""
    train_records: List[Dict[str, Any]] = []
    holdout_records: List[Dict[str, Any]] = []

    for unit in units:
        record = _make_record(unit)
        if _in_holdout(unit, holdout_ratio):
            holdout_records.append(record)
        else:
            train_records.append(record)

    return train_records, holdout_records


# ---------------------------------------------------------------------------
# Unit filtering and capping helpers
# ---------------------------------------------------------------------------

# Maps config extraction_units plural names to scanner type singular names.
_UNIT_TYPE_MAP = {
    "functions": "function",
    "classes": "class",
    "api_calls": "api_call",
    "methods": "method",
}


def _filter_units_by_config(units: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
    """Keep only units whose scanner type is enabled by config.extraction_units.

    If config.extraction_units is empty/falsy, all units are kept (defensive).
    """
    enabled = getattr(config, "extraction_units", None)
    if not enabled:
        return units
    allowed_types = {_UNIT_TYPE_MAP[u] for u in enabled if u in _UNIT_TYPE_MAP}
    return [u for u in units if u.get("type") in allowed_types]


def _cap_units(units: List[Dict[str, Any]], target_records: int) -> List[Dict[str, Any]]:
    """Deterministically cap units to at most target_records entries.

    Uses SHA-256 of "<file>:<name>" as sort key — stable across processes.
    """
    if len(units) <= target_records:
        return units
    return sorted(
        units,
        key=lambda u: hashlib.sha256(f"{u['file']}:{u['name']}".encode()).hexdigest(),
    )[:target_records]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_from_code(config, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate training records from pre-scanned code units.

    Args:
        config: A RepoConfig instance (used for holdout_ratio).
        units:  List of unit dicts from scan_repo().

    Returns:
        All records (train + holdout combined) as a list of dicts.
        Call _split_records() separately if you need the partition.
    """
    records = [_make_record(unit) for unit in units]
    return records


def generate_from_repo(config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scan the repo described by config and return (train_records, holdout_records).

    This is the thin all-in-one wrapper: it calls scan_repo internally,
    filters by extraction_units (#2), caps to target_records (#3), then
    partitions the resulting records.
    """
    from scripts.repo.scan_repo import scan_repo  # local import to keep module testable

    units = scan_repo(config)
    units = _filter_units_by_config(units, config)
    units = _cap_units(units, config.target_records)
    return _split_records(units, config.holdout_ratio)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to repo.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root directory for output files. "
            "Defaults to data/repos/ relative to repo root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files — print counts only.",
    )
    args = parser.parse_args()

    # Load config
    from scripts.repo.loader import load_repo_config

    try:
        config = load_repo_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error loading config: {exc}")

    # Guard against path traversal: config.name is used as an output subdir, so
    # reject any value that could escape the output root (path separators,
    # parent refs, or absolute paths).
    if (
        "/" in config.name
        or "\\" in config.name
        or ".." in config.name
        or Path(config.name).is_absolute()
        or config.name in (".", "")
    ):
        sys.exit(f"Error: unsafe repo name for output path: {config.name!r}")

    # Determine output directory
    if args.output_dir:
        out_root = Path(args.output_dir).expanduser()
    else:
        # Default: data/repos/ relative to the apisynth repo root, not the
        # repo.yaml location. This is consistent with repo_pipeline.py and
        # pipeline.py conventions (_REPO_ROOT / "data" / ...).  Every CLI test
        # passes --output-dir explicitly, so this default is safe to change.
        out_root = _REPO_ROOT / "data" / "repos"

    repo_out_dir = out_root / config.name
    train_path = repo_out_dir / "training.jsonl"
    holdout_path = repo_out_dir / "holdout.jsonl"

    # Scan and split
    from scripts.repo.scan_repo import scan_repo

    print(f"Scanning repo: {config.name}")
    units = scan_repo(config)
    print(f"  Found {len(units)} units")
    units = _filter_units_by_config(units, config)
    units = _cap_units(units, config.target_records)
    print(f"  After filter/cap: {len(units)} units")

    train_records, holdout_records = _split_records(units, config.holdout_ratio)
    print(
        f"  Split: {len(train_records)} train / {len(holdout_records)} holdout "
        f"(ratio={config.holdout_ratio})"
    )

    if args.dry_run:
        print("  (dry run — no files written)")
        return

    _write_jsonl(train_records, train_path)
    _write_jsonl(holdout_records, holdout_path)
    print(f"  Wrote {train_path}")
    print(f"  Wrote {holdout_path}")


if __name__ == "__main__":
    main()
