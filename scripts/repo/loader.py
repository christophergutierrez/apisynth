"""Repository configuration schema and loader for repo ingestion.

Defines repo.yaml schema and provides load_repo_config() function.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

# Constants for schema validation
VALID_LANGUAGES = frozenset({"python"})
VALID_EXTRACTION_UNITS = frozenset({"functions", "classes", "api_calls", "methods"})
VALID_HOLDOUT_STRATEGIES = frozenset({"hash", "stratified"})

DEFAULT_INCLUDE = ["**/*.py"]
DEFAULT_EXTRACTION_UNITS = ["functions", "classes"]
DEFAULT_MIN_COMPLEXITY = "medium"
DEFAULT_TARGET_RECORDS = 500
DEFAULT_HOLDOUT_RATIO = 0.15
DEFAULT_THINKING_STYLE = "deterministic"
DEFAULT_HOLDOUT_STRATEGY = "hash"
DEFAULT_VALIDATE_SYNTAX = False
DEFAULT_REJECT_TRIVIAL = False


@dataclass
class RepoConfig:
    """Schema for repo.yaml configuration."""
    name: str
    path: str | None = None
    language: str = "python"
    include: List[str] = field(default_factory=lambda: DEFAULT_INCLUDE.copy())
    exclude: List[str] = field(default_factory=list)
    extraction_units: List[str] = field(default_factory=lambda: DEFAULT_EXTRACTION_UNITS.copy())
    min_complexity: str = DEFAULT_MIN_COMPLEXITY
    target_records: int = DEFAULT_TARGET_RECORDS
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO
    thinking_style: str = DEFAULT_THINKING_STYLE
    holdout_strategy: str = DEFAULT_HOLDOUT_STRATEGY
    url: str | None = None
    branch: str | None = None
    commit: str | None = None
    manual_overrides: str | None = None
    validate_syntax: bool = DEFAULT_VALIDATE_SYNTAX
    reject_trivial: bool = DEFAULT_REJECT_TRIVIAL


def _ensure_list(value, default):
    if value is None:
        return default.copy() if hasattr(default, "copy") else list(default)
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def load_repo_config(config_path: str | Path) -> RepoConfig:
    """Load and validate repo.yaml configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {e}") from e

    # Keep a reference to the top-level dict (before unwrapping 'repo:').
    top = data

    # Support both top-level and nested under 'repo:'
    if "repo" in data:
        repo_block = data["repo"]
    else:
        repo_block = data

    # Identity fields come from the repo block (or top-level when no 'repo:' wrapper).
    identity = repo_block

    # Required fields
    if "name" not in identity or not isinstance(identity.get("name"), str):
        raise ValueError("Missing or invalid required field: name")

    # At least one of path or url must be present (non-empty strings;
    # treat ""/whitespace-only as absent)
    raw_path = identity.get("path")
    raw_url = identity.get("url")
    has_path = isinstance(raw_path, str) and raw_path.strip() != ""
    has_url = isinstance(raw_url, str) and raw_url.strip() != ""
    if not has_path and not has_url:
        raise ValueError("Missing required field: one of path or url must be provided")

    # Resolve relative path against the config file's directory (Fix #4b).
    # Only applies to path-based (non-URL) configs.
    if has_path and not has_url:
        p = Path(raw_path)
        if not p.is_absolute():
            raw_path = str((config_path.parent / p).resolve())
            has_path = True

    # Sections: prefer TOP-LEVEL siblings (documented form), fall back to
    # inside the 'repo:' block (legacy/tested form). Fix #1.
    extraction_section = top.get("extraction") or repo_block.get("extraction") or {}
    generation_section = top.get("generation") or repo_block.get("generation") or {}

    # Handle extraction units (support both list and old dict format)
    raw_units = extraction_section.get("units", DEFAULT_EXTRACTION_UNITS)
    if isinstance(raw_units, dict):
        extraction_units = list(raw_units.keys())
    else:
        extraction_units = _ensure_list(raw_units, DEFAULT_EXTRACTION_UNITS)

    # Resolve manual_overrides path from generation section.
    # If it's a relative path, resolve it against the config file's directory.
    raw_manual_overrides = generation_section.get("manual_overrides")
    if isinstance(raw_manual_overrides, str) and raw_manual_overrides.strip():
        mo_path = Path(raw_manual_overrides)
        if not mo_path.is_absolute():
            raw_manual_overrides = str((config_path.parent / mo_path).resolve())
    else:
        raw_manual_overrides = None

    config = RepoConfig(
        name=identity["name"],
        path=raw_path if has_path else None,
        language=identity.get("language", "python"),
        include=_ensure_list(identity.get("include"), DEFAULT_INCLUDE),
        exclude=_ensure_list(identity.get("exclude"), []),
        extraction_units=extraction_units,
        min_complexity=extraction_section.get("min_complexity", DEFAULT_MIN_COMPLEXITY),
        target_records=generation_section.get("target_records", DEFAULT_TARGET_RECORDS),
        holdout_ratio=generation_section.get("holdout_ratio", DEFAULT_HOLDOUT_RATIO),
        thinking_style=generation_section.get("thinking_style", DEFAULT_THINKING_STYLE),
        holdout_strategy=generation_section.get("holdout_strategy", DEFAULT_HOLDOUT_STRATEGY),
        url=raw_url if has_url else None,
        branch=identity.get("branch") if isinstance(identity.get("branch"), str) else None,
        commit=identity.get("commit") if isinstance(identity.get("commit"), str) else None,
        manual_overrides=raw_manual_overrides,
        validate_syntax=bool(generation_section.get("validate_syntax", DEFAULT_VALIDATE_SYNTAX)),
        reject_trivial=bool(generation_section.get("reject_trivial", DEFAULT_REJECT_TRIVIAL)),
    )

    # Validation
    if config.language not in VALID_LANGUAGES:
        raise ValueError(f"Unsupported language: {config.language}")

    for unit in config.extraction_units:
        if unit not in VALID_EXTRACTION_UNITS:
            raise ValueError(f"Invalid extraction unit: {unit}")

    if not (0 < config.holdout_ratio < 1):
        raise ValueError("holdout_ratio must be between 0 and 1")
    if config.target_records <= 0:
        raise ValueError("target_records must be positive")
    if config.holdout_strategy not in VALID_HOLDOUT_STRATEGIES:
        raise ValueError(
            f"Invalid holdout_strategy: {config.holdout_strategy!r}. "
            f"Must be one of: {sorted(VALID_HOLDOUT_STRATEGIES)}"
        )

    # Path validation — only when url is not set (cannot stat a remote)
    if not has_url:
        repo_path = Path(config.path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {config.path}")
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {config.path}")

    return config


def main():
    """CLI entrypoint for smoke testing config loading."""
    import argparse
    parser = argparse.ArgumentParser(description="Load and validate repo.yaml")
    parser.add_argument("config", help="Path to repo.yaml")
    args = parser.parse_args()
    try:
        cfg = load_repo_config(args.config)
        print(f"Loaded repo config: {cfg.name} (lang={cfg.language}, units={cfg.extraction_units})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
