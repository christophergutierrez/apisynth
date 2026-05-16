#!/usr/bin/env python3
"""
Generate router classifier training data from all config.yaml files in an APIs directory.

For each confirmed variant, generates question phrasings using the same logic as
run.py but without any API calls. Writes flat (question, route_key) records.

Route key format: "{vendor}/api/{name}"  (e.g. "acme/api/programs")
Vendor is read from each config's endpoint.vendor field.

Usage:
    python scripts/gen_router_data.py --apis-dir apis/<vendor>
    python scripts/gen_router_data.py --apis-dir apis/<vendor> --out-dir data/<vendor>/router --target 50
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import CFG_VARIANTS, CFG_CONFIRMED, CFG_TARGET_PER_VARIANT, FIELD_QUESTION, \
    PYYAML_REQUIRED

try:
    import yaml
except ImportError:
    sys.exit(PYYAML_REQUIRED)
from run import gen_questions, gen_chained_questions

_REPO = Path(__file__).parents[1]
TRAIN_RATIO = 0.8
SEED = 42


def load_all_configs(apis_dir: Path) -> list[tuple[str, dict]]:
    result = []
    for config_path in sorted(apis_dir.glob("*/config.yaml")):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        name = cfg["endpoint"]["name"]
        vendor = cfg["endpoint"]["vendor"]
        route_key = f"{vendor}/api/{name}"
        result.append((route_key, cfg))
    return result


def questions_for_config(cfg: dict, target_per_variant: int) -> list[str]:
    status = cfg.get("status") or {}
    confirmed = [v for v in (status.get(CFG_VARIANTS) or []) if v.get(CFG_CONFIRMED)]
    if not confirmed:
        return []

    seen: set[str] = set()
    questions: list[str] = []

    for v in confirmed:
        vparams = v["params"]
        t = v.get("target") or cfg.get("training", {}).get(CFG_TARGET_PER_VARIANT) or target_per_variant
        t = max(t, target_per_variant)
        for q, _ in gen_questions(cfg, status, vparams, t):
            if q not in seen:
                seen.add(q)
                questions.append(q)

    if cfg.get("parent") and cfg.get("path_params"):
        t = cfg.get("training", {}).get(CFG_TARGET_PER_VARIANT) or target_per_variant
        for q, _ in gen_chained_questions(cfg, t):
            if q not in seen:
                seen.add(q)
                questions.append(q)

    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apis-dir", required=True, help="Directory containing endpoint subdirs with config.yaml")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: data/<vendor>/router)")
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="minimum questions to generate per variant (default: 50)",
    )
    args = parser.parse_args()

    apis_dir = Path(args.apis_dir)
    configs = load_all_configs(apis_dir)

    if not configs:
        sys.exit(f"No config.yaml files found under {apis_dir}")

    # Derive output dir from vendor if not specified
    vendor = configs[0][1]["endpoint"]["vendor"]
    out_dir = Path(args.out_dir) if args.out_dir else _REPO / "data" / vendor / "router"

    all_records: list[dict] = []
    for route_key, cfg in configs:
        questions = questions_for_config(cfg, args.target)
        for q in questions:
            all_records.append({FIELD_QUESTION: q, "route_key": route_key})
        status = f"{len(questions)} questions" if questions else "no confirmed variants — skipped"
        print(f"  {route_key}: {status}")

    if not all_records:
        sys.exit("No records generated.")

    random.seed(SEED)
    random.shuffle(all_records)

    split = int(len(all_records) * TRAIN_RATIO)
    train = all_records[:split]
    test = all_records[split:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "router_train.jsonl"
    test_path = out_dir / "router_test.jsonl"

    with open(train_path, "w") as f:
        for rec in train:
            f.write(json.dumps(rec) + "\n")

    with open(test_path, "w") as f:
        for rec in test:
            f.write(json.dumps(rec) + "\n")

    route_counts: dict[str, int] = {}
    for rec in all_records:
        route_counts[rec["route_key"]] = route_counts.get(rec["route_key"], 0) + 1

    print(f"\nTotal: {len(all_records)} records — {len(train)} train / {len(test)} test")
    print(f"Routes: {len(route_counts)}")
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")


if __name__ == "__main__":
    main()
