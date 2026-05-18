#!/usr/bin/env python3
"""
Full apisynth pipeline: sweep → run → thinking → enrich → evolve → router

Orchestrates all data generation steps for a vendor in sequence.
Each step is resumable — already-completed work is detected and skipped.

Usage:
    python scripts/pipeline.py --vendor-dir apis/<vendor>

    # Skip specific steps
    python scripts/pipeline.py --vendor-dir apis/<vendor> --skip-sweep --skip-evolve

    # Resume from a specific step (skips everything before it)
    python scripts/pipeline.py --vendor-dir apis/<vendor> --from-step thinking

    # Preview without running
    python scripts/pipeline.py --vendor-dir apis/<vendor> --dry-run

Steps (in order):
    sweep     Discover valid param values and confirm variants (per endpoint, live API)
    run       Generate training records (per endpoint, live API)
    thinking  Add deterministic reasoning traces (no API calls)
    enrich    Add schema + intent_category fields (no API calls)
    evolve    Diversify questions via LLM mutation (Claude API, optional)
    router    Generate router training data + train classifier (no API calls)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

_REPO  = Path(__file__).parents[1]
_PY    = sys.executable
_SCRIPTS = Path(__file__).parent

STEPS = ["sweep", "run", "thinking", "enrich", "evolve", "router"]

# ── Status detection ───────────────────────────────────────────────────────

def _count_records(data_dir: Path, field: str | None = None) -> int:
    total = 0
    for f in data_dir.glob("*/training.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            if field is None:
                total += 1
            else:
                try:
                    if field in json.loads(line):
                        total += 1
                except json.JSONDecodeError:
                    pass
    return total


def _has_confirmed_variants(config_path: Path) -> bool:
    cfg = yaml.safe_load(config_path.read_text())
    variants = cfg.get("status", {}).get("variants", [])
    return any(v.get("confirmed") for v in variants)


def _status_table(vendor_dir: Path, data_dir: Path, configs: list[Path]) -> None:
    total     = _count_records(data_dir)
    thinking  = _count_records(data_dir, "thinking")
    schema    = _count_records(data_dir, "schema")
    intent    = _count_records(data_dir, "intent_category")
    evol      = sum(
        1 for f in data_dir.glob("*/training.jsonl")
        for line in f.read_text().splitlines()
        if line.strip() and json.loads(line).get("source") == "evol"
    )
    swept     = sum(1 for c in configs if _has_confirmed_variants(c))
    router_ok = (data_dir / "router" / "router_classifier.joblib").exists()

    print("  ┌─────────────────────────────────────────┐")
    print(f"  │ Vendor:   {vendor_dir.name:<30} │")
    print(f"  │ Endpoints: {len(configs):<5}  swept: {swept:<5}              │")
    print(f"  │ Records:   {total:<5}  thinking: {thinking:<5}           │")
    print(f"  │ Schema:    {schema:<5}  intent:   {intent:<5}           │")
    print(f"  │ Evolved:   {evol:<5}  router:   {'yes' if router_ok else 'no':<5}           │")
    print("  └─────────────────────────────────────────┘")


# ── Step runners ───────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str, dry_run: bool) -> bool:
    print(f"\n  → {label}")
    if dry_run:
        print(f"    [dry-run] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, cwd=_REPO)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        return False
    return True


def step_sweep(configs: list[Path], dry_run: bool) -> bool:
    ok = True
    for cfg_path in configs:
        if _has_confirmed_variants(cfg_path):
            ep = cfg_path.parent.name
            print(f"  ✓ sweep  {ep} (already confirmed)")
            continue
        if not _run(
            [_PY, str(_SCRIPTS / "sweep.py"), "--config", str(cfg_path)],
            f"sweep {cfg_path.parent.name}",
            dry_run,
        ):
            ok = False
    return ok


def step_run(configs: list[Path], dry_run: bool) -> bool:
    ok = True
    for cfg_path in configs:
        if not _has_confirmed_variants(cfg_path):
            print(f"  ✗ run    {cfg_path.parent.name} — no confirmed variants, run sweep first")
            continue
        if not _run(
            [_PY, str(_SCRIPTS / "run.py"), "--config", str(cfg_path)],
            f"run {cfg_path.parent.name}",
            dry_run,
        ):
            ok = False
    return ok


def step_thinking(data_dir: Path, dry_run: bool) -> bool:
    total  = _count_records(data_dir)
    traced = _count_records(data_dir, "thinking")
    if total == 0:
        print("  ✗ thinking — no records found, run 'run' step first")
        return True
    if traced == total:
        print(f"  ✓ thinking ({traced}/{total} records already have traces)")
        return True
    return _run(
        [_PY, str(_SCRIPTS / "add_thinking.py"), "--input-dir", str(data_dir)],
        f"add_thinking ({total - traced} records need traces)",
        dry_run,
    )


def step_enrich(vendor_dir: Path, data_dir: Path, dry_run: bool) -> bool:
    total    = _count_records(data_dir)
    enriched = _count_records(data_dir, "schema")
    if total == 0:
        print("  ✗ enrich — no records found")
        return True
    if enriched == total:
        print(f"  ✓ enrich ({enriched}/{total} records already have schema)")
        return True
    return _run(
        [_PY, str(_SCRIPTS / "enrich_schema.py"), "--vendor-dir", str(vendor_dir)],
        f"enrich_schema ({total - enriched} records need schema)",
        dry_run,
    )


def step_evolve(data_dir: Path, per_record: int, sample: int | None, dry_run: bool) -> bool:
    args = [_PY, str(_SCRIPTS / "evolve_questions.py"), "--input-dir", str(data_dir),
            "--per-record", str(per_record)]
    if sample:
        args += ["--sample", str(sample)]
    return _run(args, f"evolve_questions (per-record={per_record})", dry_run)


def step_router(vendor_dir: Path, data_dir: Path, dry_run: bool) -> bool:
    router_dir = data_dir / "router"
    gen_ok = _run(
        [_PY, str(_SCRIPTS / "gen_router_data.py"), "--apis-dir", str(vendor_dir),
         "--out-dir", str(router_dir)],
        "gen_router_data",
        dry_run,
    )
    if not gen_ok:
        return False
    return _run(
        [_PY, str(_SCRIPTS / "train_router.py"), "--data-dir", str(router_dir)],
        "train_router",
        dry_run,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor-dir", required=True, type=Path,
                        help="Path to vendor API directory (e.g. apis/videoamp)")
    parser.add_argument("--data-dir", default=None, type=Path,
                        help="Override data output directory (default: data/<vendor-name>)")
    parser.add_argument("--from-step", choices=STEPS, default=None,
                        help="Start from this step, skipping all earlier steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--skip-sweep",   action="store_true")
    parser.add_argument("--skip-run",     action="store_true")
    parser.add_argument("--skip-thinking",action="store_true")
    parser.add_argument("--skip-enrich",  action="store_true")
    parser.add_argument("--skip-evolve",  action="store_true",
                        help="Skip question evolution (requires Claude API key)")
    parser.add_argument("--skip-router",  action="store_true")
    parser.add_argument("--evolve-per-record", type=int, default=1,
                        help="Mutations per seed record for evolve step (default: 1)")
    parser.add_argument("--evolve-sample", type=int, default=None,
                        help="Only evolve a random sample of N records per endpoint")
    args = parser.parse_args()

    vendor_dir = args.vendor_dir.resolve()
    if not vendor_dir.is_dir():
        sys.exit(f"vendor-dir not found: {vendor_dir}")

    vendor_name = vendor_dir.name
    data_dir = (args.data_dir or (_REPO / "data" / vendor_name)).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(vendor_dir.glob("*/config.yaml"))
    if not configs:
        sys.exit(f"No endpoint config.yaml files found under {vendor_dir}")

    skip = set()
    if args.from_step:
        skip = set(STEPS[:STEPS.index(args.from_step)])
    if args.skip_sweep:    skip.add("sweep")
    if args.skip_run:      skip.add("run")
    if args.skip_thinking: skip.add("thinking")
    if args.skip_enrich:   skip.add("enrich")
    if args.skip_evolve:   skip.add("evolve")
    if args.skip_router:   skip.add("router")

    print(f"\napisynth pipeline — {vendor_name}")
    print(f"  vendor-dir: {vendor_dir}")
    print(f"  data-dir:   {data_dir}")
    print(f"  endpoints:  {len(configs)}")
    print(f"  skipping:   {sorted(skip) or 'none'}")
    print(f"  dry-run:    {args.dry_run}")
    print()
    _status_table(vendor_dir, data_dir, configs)

    t0 = time.perf_counter()
    results: dict[str, bool] = {}

    if "sweep" not in skip:
        results["sweep"] = step_sweep(configs, args.dry_run)

    if "run" not in skip:
        results["run"] = step_run(configs, args.dry_run)

    if "thinking" not in skip:
        results["thinking"] = step_thinking(data_dir, args.dry_run)

    if "enrich" not in skip:
        results["enrich"] = step_enrich(vendor_dir, data_dir, args.dry_run)

    if "evolve" not in skip:
        results["evolve"] = step_evolve(
            data_dir, args.evolve_per_record, args.evolve_sample, args.dry_run
        )

    if "router" not in skip:
        results["router"] = step_router(vendor_dir, data_dir, args.dry_run)

    elapsed = time.perf_counter() - t0

    print(f"\n{'─'*50}")
    print(f"Pipeline complete in {elapsed:.0f}s")
    print()
    _status_table(vendor_dir, data_dir, configs)

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed steps: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
