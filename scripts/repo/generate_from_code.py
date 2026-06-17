#!/usr/bin/env python3
"""Generate synthetic training records from scanned code units.

Reads a repo.yaml, scans the target repository for code units (functions,
methods, classes, and API call sites), then writes training.jsonl and
holdout.jsonl files under data/repos/<repo-name>/.

Each record carries:
  - type: "code"
  - question: templated natural-language prompt keyed off unit type
  - thinking: deterministic Entity/Scope/Use/… reasoning trace (or QOC)
  - output: structured code-unit descriptor (unit, name, file, [class], [signature])

The train/holdout split is deterministic: each unit's SHA-256 hash of
"<file>:<name>" is mapped into [0, 1) (by dividing the integer digest by
2**256), and the unit is assigned to holdout when that value is below
holdout_ratio. Because SHA-256 is process-independent (unlike the builtin
salted hash()), the same input always produces the same split across runs and
processes — no random seed required. See _split_records() / _in_holdout().

Thinking style mapping (used when generate_from_code style=None):
  config.thinking_style == "deterministic"  →  "linear"  (default)
  config.thinking_style == "hybrid"         →  "qoc"

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
from typing import Any, Dict, List, Optional, Tuple

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
# Signature helper
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


# ---------------------------------------------------------------------------
# Thinking trace generators — LINEAR style (Entity/Scope/Use/NOT)
# ---------------------------------------------------------------------------

def _linear_function(unit: Dict[str, Any]) -> str:
    """Linear trace for a top-level function unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Entity: function {name}\n"
        f"File: {location}\n"
        f"Scope: single unit — top-level function\n"
        f"Use: call {sig}\n"
        f"Params: see function signature in {file_path}"
    )


def _linear_method(unit: Dict[str, Any]) -> str:
    """Linear trace for a method unit.

    Classless method units (no 'class' key) must NOT emit the literal 'None'.
    """
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    cls = unit.get("class") or ""  # guard against None and missing key
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path

    if cls:
        entity_line = f"Entity: method {name} on class {cls}"
        use_line = f"Use: instance.{sig}  # where instance is a {cls}"
    else:
        entity_line = f"Entity: method {name}"
        use_line = f"Use: instance.{sig}"

    return (
        f"{entity_line}\n"
        f"File: {location}\n"
        f"Scope: single unit — instance method\n"
        f"{use_line}\n"
        f"NOT: calling {name}() as a standalone function"
    )


def _linear_class(unit: Dict[str, Any]) -> str:
    """Linear trace for a class unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Entity: class {name}\n"
        f"File: {location}\n"
        f"Scope: single unit — class definition\n"
        f"Use: instantiate with {sig}\n"
        f"Params: see __init__ in {file_path}"
    )


def _linear_api_call(unit: Dict[str, Any]) -> str:
    """Linear trace for an api_call unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Entity: api_call {name}\n"
        f"File: {location}\n"
        f"Scope: single call site — HTTP/API invocation\n"
        f"Use: {sig}\n"
        f"NOT: a plain dict.get() or queue.get() — this is an HTTP/API call"
    )


def _make_thinking_linear(unit: Dict[str, Any]) -> str:
    """Dispatch to the appropriate linear-style per-type helper."""
    unit_type = unit["type"]
    if unit_type == "function":
        return _linear_function(unit)
    if unit_type == "method":
        return _linear_method(unit)
    if unit_type == "class":
        return _linear_class(unit)
    if unit_type == "api_call":
        return _linear_api_call(unit)
    # Fallback (should not occur with known scanner types)
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Entity: {unit_type} {name}\n"
        f"File: {location}\n"
        f"Scope: single unit"
    )


# ---------------------------------------------------------------------------
# Thinking trace generators — QOC style (Question/Option/Criteria)
# Adapted for code units, mirroring add_thinking.py's _qoc_* generators.
# ---------------------------------------------------------------------------

def _qoc_function(unit: Dict[str, Any]) -> str:
    """QOC trace for a top-level function unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Question: How should `{name}` be invoked — as a direct call or via an object?\n"
        f"Option A: call directly — {sig}  (top-level function, no instance needed)\n"
        f"Option B: call on an instance — would only apply if it were a method\n"
        f"Criteria: `{name}` is a top-level function at {location}. "
        f"No instance required. Option A wins.\n"
        f"Use: {sig}"
    )


def _qoc_method(unit: Dict[str, Any]) -> str:
    """QOC trace for a method unit.

    Classless method units (no 'class' key) must NOT emit the literal 'None'.
    """
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    cls = unit.get("class") or ""  # guard against None and missing key
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path

    if cls:
        question_line = f"Question: Should `{name}` be called on a `{cls}` instance or as a standalone function?"
        option_a = f"Option A: call on an instance — instance.{sig}  where instance is a {cls}"
        criteria = (
            f"Criteria: `{name}` is an instance method of `{cls}` at {location}. "
            f"An instance of `{cls}` must exist before calling. Option A wins."
        )
    else:
        question_line = f"Question: Should `{name}` be called on an instance or as a standalone function?"
        option_a = f"Option A: call on an instance — instance.{sig}"
        criteria = (
            f"Criteria: `{name}` is an instance method at {location}. "
            f"An instance of the owning class must exist before calling. Option A wins."
        )

    return (
        f"{question_line}\n"
        f"{option_a}\n"
        f"Option B: call as a standalone function — {name}(...)  (incorrect — not a top-level function)\n"
        f"{criteria}\n"
        f"NOT: {name}() as a standalone function"
    )


def _qoc_class(unit: Dict[str, Any]) -> str:
    """QOC trace for a class unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Question: Should `{name}` be instantiated or referenced as a type?\n"
        f"Option A: instantiate — {sig}  (creates a new {name} object)\n"
        f"Option B: reference the type — {name}  (use as a class object, e.g. for isinstance checks)\n"
        f"Criteria: `{name}` is a class defined at {location}. "
        f"For object creation, Option A wins. For type introspection, Option B applies.\n"
        f"Use: {sig}  (default — create an instance)"
    )


def _qoc_api_call(unit: Dict[str, Any]) -> str:
    """QOC trace for an api_call unit."""
    name = unit["name"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path
    return (
        f"Question: Is `{name}` a real HTTP/API call or a plain data-structure operation?\n"
        f"Option A: real HTTP/API call — {sig}  (network request, may raise on HTTP errors)\n"
        f"Option B: plain dict.get() or queue.get()  (local, no network)\n"
        f"Criteria: `{name}` is an API call site at {location}. "
        f"It performs an actual HTTP/API request. Option A wins.\n"
        f"NOT: a plain dict.get() or queue.get() — this is an HTTP/API call"
    )


def _make_thinking_qoc(unit: Dict[str, Any]) -> str:
    """Dispatch to the appropriate QOC-style per-type helper."""
    unit_type = unit["type"]
    if unit_type == "function":
        return _qoc_function(unit)
    if unit_type == "method":
        return _qoc_method(unit)
    if unit_type == "class":
        return _qoc_class(unit)
    if unit_type == "api_call":
        return _qoc_api_call(unit)
    # Fallback
    return _make_thinking_linear(unit)


# ---------------------------------------------------------------------------
# Public thinking dispatcher
# ---------------------------------------------------------------------------

def generate_code_thinking(unit: Dict[str, Any], style: str = "linear") -> str:
    """Return a deterministic thinking trace for the given code unit.

    Args:
        unit:  A scanner unit dict with at minimum 'type', 'name', 'file'.
        style: Trace style — 'linear' (Entity/Scope/Use/NOT) or 'qoc'
               (Question/Option/Criteria). Unknown values fall back to 'linear'.

    Returns:
        A multi-line string suitable for the 'thinking' field of a training record.
    """
    if style == "qoc":
        return _make_thinking_qoc(unit)
    # 'linear' is the default; any unknown style also falls back here.
    return _make_thinking_linear(unit)


# Keep the original name as an internal alias for backward compatibility with
# existing direct test imports that call _make_thinking(unit) with one argument.
def _make_thinking(unit: Dict[str, Any], style: str = "linear") -> str:
    """Return a deterministic thinking trace (backward-compatible wrapper)."""
    return generate_code_thinking(unit, style=style)


# ---------------------------------------------------------------------------
# Pattern template generators (Milestone 2.2)
#
# Each function takes a unit dict and returns a deterministic multi-line trace
# keyed to a specific code-task pattern ("implement", "call_helper", "refactor").
# These are standalone opt-in generators. They are NOT wired into the default
# pipeline (generate_from_repo / main) — that wiring is deferred to a later phase.
# ---------------------------------------------------------------------------

def _pattern_implement(unit: Dict[str, Any]) -> str:
    """Structured reasoning trace for the 'implement' pattern.

    Covers: scope of the work, target file:lineno, signature, what to write.
    """
    name = unit["name"]
    unit_type = unit["type"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    cls = unit.get("class") or ""  # guard against None and missing key
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path

    if cls:
        scope_line = f"Scope: implement {unit_type} {name} on class {cls}"
    else:
        scope_line = f"Scope: implement {unit_type} {name}"

    return (
        f"Implement: {name}\n"
        f"Target: {location}\n"
        f"{scope_line}\n"
        f"Signature: {sig}\n"
        f"Write: body of {name} at {location} — do not alter surrounding code\n"
        f"NOT: modifying callers or existing tests — scope is this unit only"
    )


def _pattern_call_helper(unit: Dict[str, Any]) -> str:
    """Structured reasoning trace for the 'call_helper' pattern.

    Covers: the call form, where the helper lives, disambiguation that it is
    an internal helper (not a public API).
    """
    name = unit["name"]
    unit_type = unit["type"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    cls = unit.get("class") or ""  # guard against None and missing key
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path

    if cls:
        helper_line = f"Helper: {name} — internal helper method on {cls} at {location}"
        call_line = f"Call: instance.{sig}  # where instance is a {cls}"
    else:
        helper_line = f"Helper: {name} — internal {unit_type} at {location}"
        call_line = f"Call: {sig}"

    return (
        f"{helper_line}\n"
        f"{call_line}\n"
        f"Scope: single call site — invoke {name} from within the same module\n"
        f"NOT: a public API endpoint — this is an internal helper, not exposed externally\n"
        f"NOT: reimplementing {name} — just call it"
    )


def _pattern_refactor(unit: Dict[str, Any]) -> str:
    """Structured reasoning trace for the 'refactor' pattern.

    Covers: identify the unit at file:lineno, preserve behavior/signature,
    scope of the change.
    """
    name = unit["name"]
    unit_type = unit["type"]
    file_path = unit["file"]
    lineno = unit.get("lineno")
    cls = unit.get("class") or ""  # guard against None and missing key
    sig = _signature_for(unit)
    location = f"{file_path}:{lineno}" if lineno is not None else file_path

    if cls:
        target_line = f"Target: {unit_type} {name} on class {cls} at {location}"
    else:
        target_line = f"Target: {unit_type} {name} at {location}"

    return (
        f"Refactor: {name}\n"
        f"{target_line}\n"
        f"Preserve: external behavior — callers must not require changes\n"
        f"Preserve: signature {sig} — do not alter the public interface\n"
        f"Scope: internals of {name} only — do not change callers or tests\n"
        f"NOT: adding new features or changing observable behavior"
    )


# Registry of all pattern template generators.
_PATTERN_TEMPLATES: Dict[str, Any] = {
    "implement": _pattern_implement,
    "call_helper": _pattern_call_helper,
    "refactor": _pattern_refactor,
}


def generate_pattern_thinking(unit: Dict[str, Any], pattern: str) -> str:
    """Return a deterministic thinking trace for the given code unit and pattern.

    Pattern templates are explicit opt-in generators keyed by pattern name.
    Unlike ``generate_code_thinking``, an unknown pattern raises ``ValueError``
    (rather than silently falling back) because each pattern carries a distinct
    reasoning structure and guessing the wrong one would produce misleading traces.

    Args:
        unit:    A scanner unit dict with at minimum 'type', 'name', 'file'.
        pattern: One of the known pattern names: 'implement', 'call_helper',
                 'refactor'. Any other value raises ValueError.

    Returns:
        A multi-line string suitable for the 'thinking' field of a training record.

    Raises:
        ValueError: If ``pattern`` is not a recognised pattern name.
    """
    if pattern not in _PATTERN_TEMPLATES:
        valid = ", ".join(sorted(_PATTERN_TEMPLATES))
        raise ValueError(
            f"Unknown pattern {pattern!r}. Valid patterns are: {valid}"
        )
    return _PATTERN_TEMPLATES[pattern](unit)


def _style_from_config(config) -> str:
    """Map config.thinking_style to a trace style. Single source of truth.

    "hybrid"        → "qoc"
    "deterministic" → "linear"  (and anything else / missing → "linear")
    """
    if getattr(config, "thinking_style", "deterministic") == "hybrid":
        return "qoc"
    return "linear"


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _make_record(unit: Dict[str, Any], style: str = "linear") -> Dict[str, Any]:
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
        "thinking": _make_thinking(unit, style=style),
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
    units: List[Dict[str, Any]], holdout_ratio: float, style: str = "linear"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition units into (train, holdout) records deterministically.

    The `style` arg ("linear" default, or "qoc") is threaded into _make_record
    so the production path (main()/generate_from_repo) honors thinking_style.
    """
    train_records: List[Dict[str, Any]] = []
    holdout_records: List[Dict[str, Any]] = []

    for unit in units:
        record = _make_record(unit, style=style)
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

def generate_from_code(
    config, units: List[Dict[str, Any]], style: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate training records from pre-scanned code units.

    Args:
        config: A RepoConfig instance (used for holdout_ratio and thinking_style).
        units:  List of unit dicts from scan_repo().
        style:  Thinking trace style — 'linear' or 'qoc'. When None (default),
                the style is derived from config.thinking_style via
                _style_from_config (the single source of truth):
                  "deterministic" → "linear"  (the default)
                  "hybrid"        → "qoc"

    Returns:
        All records (train + holdout combined) as a list of dicts.
        Call _split_records() separately if you need the partition.
    """
    # Resolve style from config when not explicitly supplied (single source of truth).
    if style is None:
        style = _style_from_config(config)

    records = [_make_record(unit, style=style) for unit in units]
    return records


def generate_from_repo(config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scan the repo described by config and return (train_records, holdout_records).

    This is the thin all-in-one wrapper: it calls scan_repo internally,
    filters by extraction_units (#2), caps to target_records (#3), then
    partitions the resulting records. The thinking style is derived from
    config.thinking_style so the production path honors "hybrid" → QOC.
    """
    from scripts.repo.scan_repo import scan_repo  # local import to keep module testable

    units = scan_repo(config)
    units = _filter_units_by_config(units, config)
    units = _cap_units(units, config.target_records)
    style = _style_from_config(config)
    return _split_records(units, config.holdout_ratio, style=style)


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

    # Derive trace style from config so written records honor thinking_style.
    style = _style_from_config(config)
    train_records, holdout_records = _split_records(
        units, config.holdout_ratio, style=style
    )
    print(
        f"  Split: {len(train_records)} train / {len(holdout_records)} holdout "
        f"(ratio={config.holdout_ratio}, style={style})"
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
