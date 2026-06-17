#!/usr/bin/env python3
"""
Repo pipeline: scan → generate

Thin orchestrator that takes a local repo.yaml source and produces
data/repos/<repo-name>/training.jsonl + holdout.jsonl end-to-end.

Usage:
    python scripts/repo_pipeline.py --repo-dir repos/example

    # Preview without writing files
    python scripts/repo_pipeline.py --repo-dir repos/example --dry-run

    # Override the data root (useful for testing)
    python scripts/repo_pipeline.py --repo-dir repos/example --data-dir /tmp/out
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so that `scripts.*` imports work regardless
# of the working directory when this file is executed directly.
_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _validate_repo_name(name: str) -> None:
    """Reject config.name values that could escape the output root.

    Mirrors the guard in scripts/repo/generate_from_code.main().
    """
    if (
        "/" in name
        or "\\" in name
        or ".." in name
        or Path(name).is_absolute()
        or name in (".", "")
    ):
        sys.exit(f"Error: unsafe repo name for output path: {name!r}")


def run_pipeline(
    repo_dir: Path,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the repo pipeline (scan → generate) and return a result summary dict.

    Args:
        repo_dir:  Directory containing repo.yaml.
        data_dir:  Root for output files. Defaults to _REPO / "data" / "repos"
                   (consistent with pipeline.py's _REPO / "data" / ... convention).
        dry_run:   If True, compute counts but write no files.

    Returns:
        A dict with keys: name, train_count, holdout_count, skipped, dry_run.

    Raises:
        SystemExit on configuration errors (missing repo.yaml, unsafe name, …).
    """
    from scripts.repo.loader import load_repo_config
    from scripts.repo.generate_from_code import (
        generate_from_repo,
        _write_jsonl,
    )

    repo_dir = repo_dir.resolve()
    yaml_path = repo_dir / "repo.yaml"

    if not yaml_path.exists():
        sys.exit(f"Error: repo.yaml not found in {repo_dir}")

    # Load and validate the config.
    try:
        config = load_repo_config(yaml_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error loading config: {exc}")

    # Name-safety guard — config.name is used as an output sub-directory.
    _validate_repo_name(config.name)

    # Determine output paths.
    out_root = (data_dir or (_REPO / "data" / "repos")).resolve()
    repo_out_dir = out_root / config.name
    train_path = repo_out_dir / "training.jsonl"
    holdout_path = repo_out_dir / "holdout.jsonl"

    print(f"\napisynth repo pipeline — {config.name}")
    print(f"  repo-dir:  {repo_dir}")
    print(f"  data-dir:  {out_root}")
    print(f"  dry-run:   {dry_run}")

    # Skip-if-done: both output files exist and are non-empty.
    if (
        not dry_run
        and train_path.exists() and train_path.stat().st_size > 0
        and holdout_path.exists() and holdout_path.stat().st_size > 0
    ):
        print("  already generated — skipping (delete outputs to regenerate)")
        train_count = sum(1 for l in train_path.read_text().splitlines() if l.strip())
        hold_count = sum(1 for l in holdout_path.read_text().splitlines() if l.strip())
        return {
            "name": config.name,
            "train_count": train_count,
            "holdout_count": hold_count,
            "skipped": True,
            "dry_run": False,
        }

    # Step 1 + 2: scan → generate (in-process, no subprocess).
    print(f"\n  → scan + generate")
    train_records, holdout_records = generate_from_repo(config)

    train_count = len(train_records)
    hold_count = len(holdout_records)
    total = train_count + hold_count
    print(
        f"    Found {total} units → "
        f"{train_count} train / {hold_count} holdout "
        f"(ratio={config.holdout_ratio})"
    )

    if dry_run:
        print("    (dry-run — no files written)")
        return {
            "name": config.name,
            "train_count": train_count,
            "holdout_count": hold_count,
            "skipped": False,
            "dry_run": True,
        }

    _write_jsonl(train_records, train_path)
    _write_jsonl(holdout_records, holdout_path)
    print(f"    Wrote {train_path}")
    print(f"    Wrote {holdout_path}")

    return {
        "name": config.name,
        "train_count": train_count,
        "holdout_count": hold_count,
        "skipped": False,
        "dry_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        type=Path,
        help="Directory containing repo.yaml",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        type=Path,
        help=(
            "Override data output root "
            "(default: data/repos/ relative to apisynth repo root)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions and counts without writing any files.",
    )
    args = parser.parse_args()

    result = run_pipeline(
        repo_dir=args.repo_dir,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    if not result["skipped"] and not result["dry_run"]:
        print(
            f"\nDone — {result['train_count']} train, "
            f"{result['holdout_count']} holdout records written."
        )


if __name__ == "__main__":
    main()
