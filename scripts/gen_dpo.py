#!/usr/bin/env python3
"""
On-policy DPO pair generator for any endpoint config.yaml.

Generates (chosen, rejected) preference pairs using the live API as a reward
function. For each confirmed variant, a valid (chosen) candidate is paired
with a deliberately invalid (rejected) candidate. Both are validated against
the live API to confirm correct label assignment.

Usage:
    python scripts/gen_dpo.py --config apis/<vendor>/<endpoint>/config.yaml

Output: data/<vendor>/<endpoint>/dpo.jsonl (appended)
"""

import argparse
import copy
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_token, PAGE_SIZES, PYYAML_REQUIRED,
    FIELD_QUESTION, FIELD_API_CALL, CFG_VARIANTS, CFG_CONFIRMED,
    extract_schema, infer_intent,
)

try:
    import yaml
except ImportError:
    sys.exit(PYYAML_REQUIRED)

_REPO = Path(__file__).parents[1]
_write_lock = threading.Lock()


class RateLimitError(Exception):
    pass


# ── API validation ─────────────────────────────────────────────────────────

def _build_url(cfg: dict, query_params: dict, path_values: dict) -> str:
    if path_values:
        domain = "/".join(cfg["endpoint"]["base_url"].split("/")[:3])
        path = cfg["endpoint"]["path"]
        for k, v in path_values.items():
            path = path.replace(f"{{{k}}}", str(v))
        url = domain + path
        if query_params:
            url = f"{url}?{urllib.parse.urlencode(query_params)}"
        return url
    url = cfg["endpoint"]["base_url"]
    if query_params:
        url = f"{url}?{urllib.parse.urlencode(query_params)}"
    return url


def _api_ok(cfg: dict, token: str, params: dict) -> bool | None:
    """Return True if params produce a 2xx, False if 4xx/5xx, None on network error."""
    path_params_cfg = cfg.get("path_params") or {}
    path_values, query_params = {}, {}
    for k, v in params.items():
        if k in path_params_cfg:
            path_values[k] = v
        else:
            query_params[k] = v

    url = _build_url(cfg, query_params, path_values)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimitError()
        return e.code < 400
    except Exception:
        return None


# ── Rejected candidate generation ─────────────────────────────────────────

def _generate_rejected_candidates(cfg: dict, chosen_params: dict) -> list[dict]:
    """Generate deliberately invalid param dicts to serve as rejected candidates.

    Strategies:
      1. Wrong endpoint path token (swap singular ↔ plural in endpoint name)
      2. Extra spurious param not in config schema
      3. Missing a non-optional path param (for by-id endpoints)
      4. Out-of-range pageSize (0 or negative)
    """
    candidates = []
    all_params = {**(cfg.get("params") or {}), **(cfg.get("path_params") or {})}

    # Strategy 1: Spurious unknown param
    bad = copy.deepcopy(chosen_params)
    bad["_invalid_param_xyz"] = 1
    candidates.append(bad)

    # Strategy 2: Replace a valid ID with 0 (typically invalid for real IDs)
    for k, v in chosen_params.items():
        if isinstance(v, int) and v > 0:
            bad = copy.deepcopy(chosen_params)
            bad[k] = 0
            candidates.append(bad)
            break

    # Strategy 3: Out-of-range pageSize (0)
    if "pageSize" in chosen_params:
        bad = copy.deepcopy(chosen_params)
        bad["pageSize"] = 0
        candidates.append(bad)
    elif chosen_params == {} and cfg.get("params"):
        # Bare-list: inject a clearly wrong pageSize
        bad = {"pageSize": -1}
        candidates.append(bad)

    # Strategy 4: Drop a required path param entirely
    path_params_cfg = cfg.get("path_params") or {}
    for k in list(path_params_cfg.keys()):
        if k in chosen_params:
            bad = {kk: vv for kk, vv in chosen_params.items() if kk != k}
            candidates.append(bad)
            break

    return candidates


def _make_dpo_record(
    cfg: dict, question: str, chosen_params: dict, rejected_params: dict
) -> dict:
    endpoint = f"{cfg['endpoint']['method']} {cfg['endpoint']['path']}"
    return {
        FIELD_QUESTION: question,
        "chosen": {"endpoint": endpoint, "params": chosen_params},
        "rejected": {"endpoint": endpoint, "params": rejected_params},
        "schema": extract_schema(cfg),
        "intent_category": infer_intent(
            {"endpoint": endpoint, "params": chosen_params},
            cfg.get("path_params") or {},
        ),
    }


# ── Simple question sampler ────────────────────────────────────────────────

def _sample_question(cfg: dict, params: dict) -> str:
    """Generate a simple question for a given param set (no API call needed)."""
    ep_name = cfg["endpoint"].get("name", "resource")
    from utils import humanize, singular

    noun = humanize(ep_name)
    sing = singular(noun)

    if not params:
        return f"List all {noun}"
    if cfg.get("path_params") and any(k in cfg["path_params"] for k in params):
        id_val = next(v for k, v in params.items() if k in (cfg.get("path_params") or {}))
        return f"Get {sing} {id_val}"
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return f"List {noun} with {', '.join(parts)}"


# ── Main ───────────────────────────────────────────────────────────────────

def gen_dpo(cfg: dict, token: str, output: Path, dry_run: bool = False) -> int:
    """Generate DPO pairs for all confirmed variants in cfg. Returns count written."""
    variants = cfg.get("status", {}).get(CFG_VARIANTS, [])
    confirmed = [v for v in variants if v.get(CFG_CONFIRMED)]

    if not confirmed:
        print("No confirmed variants — run sweep.py first.")
        return 0

    written = 0
    skipped = 0

    for variant in confirmed:
        chosen_params = variant.get("params", {})
        question = _sample_question(cfg, chosen_params)
        rejected_list = _generate_rejected_candidates(cfg, chosen_params)

        # Validate chosen
        try:
            chosen_ok = _api_ok(cfg, token, chosen_params)
        except RateLimitError:
            print(f"  RATE_LIMITED on chosen — sleeping 60s")
            time.sleep(60)
            chosen_ok = _api_ok(cfg, token, chosen_params)

        if not chosen_ok:
            print(f"  SKIP  chosen invalid for variant {chosen_params} — skipping pair")
            skipped += 1
            continue

        for rejected_params in rejected_list:
            try:
                rejected_ok = _api_ok(cfg, token, rejected_params)
            except RateLimitError:
                time.sleep(60)
                rejected_ok = _api_ok(cfg, token, rejected_params)

            if rejected_ok:
                # Rejected candidate accidentally valid — discard
                continue

            record = _make_dpo_record(cfg, question, chosen_params, rejected_params)
            if dry_run:
                print(f"  DRY  {question[:60]}")
                print(f"       chosen:   {chosen_params}")
                print(f"       rejected: {rejected_params}")
            else:
                with _write_lock:
                    with open(output, "a") as f:
                        f.write(json.dumps(record) + "\n")
            written += 1
            break  # One good pair per variant is enough

    print(f"\nVariants: {len(confirmed)}, pairs written: {written}, skipped: {skipped}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    token = get_token(cfg)

    training_cfg = cfg.get("training", {})
    vendor = cfg["endpoint"]["vendor"]
    name = cfg["endpoint"]["name"]
    default_out = _REPO / "data" / vendor / name / "dpo.jsonl"
    output_tpl = training_cfg.get("output", str(default_out))
    output = Path(output_tpl.format(vendor=vendor, name=name))

    print(f"Config:  {args.config}")
    print(f"Output:  {output}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)

    gen_dpo(cfg, token, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
