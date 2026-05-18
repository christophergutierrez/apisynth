#!/usr/bin/env python3
"""
Enrich existing training.jsonl records with schema and intent_category fields.

Reads each training.jsonl under a vendor's data directory, matches each record's
endpoint to its config.yaml, then adds:
  - schema:          concise text description of the endpoint's parameters
  - intent_category: one of bare-list, paginated, filtered, by-id, chained, no-param

No API calls required. Safe to re-run — records already having both fields are skipped.

Usage:
    python scripts/enrich_schema.py --vendor-dir apis/<vendor>
    python scripts/enrich_schema.py --vendor-dir apis/<vendor> --data-dir data/<vendor>
    python scripts/enrich_schema.py --vendor-dir apis/<vendor> --force  # re-enrich all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

from utils import extract_schema, infer_intent

_REPO = Path(__file__).parents[1]


def _build_path_map(vendor_dir: Path) -> dict[str, dict]:
    """Return {endpoint_path: cfg} for all config.yaml files under vendor_dir."""
    path_to_cfg: dict[str, dict] = {}
    for cfg_file in vendor_dir.glob("*/config.yaml"):
        try:
            cfg = yaml.safe_load(cfg_file.read_text())
            path = cfg.get("endpoint", {}).get("path", "")
            if path:
                path_to_cfg[path] = cfg
        except Exception:
            pass
    return path_to_cfg


def _match_config(endpoint_str: str, path_map: dict[str, dict]) -> dict | None:
    if not endpoint_str:
        return None
    path = endpoint_str.split(" ", 1)[-1]
    if path in path_map:
        return path_map[path]
    for cfg_path, cfg in path_map.items():
        base = cfg_path.split("{")[0].rstrip("/")
        if path.startswith(base):
            return cfg
    return None


def enrich_dir(data_dir: Path, path_map: dict[str, dict], force: bool = False) -> dict:
    stats = {"enriched": 0, "skipped": 0, "no_cfg": 0}

    for jsonl in sorted(data_dir.glob("*/training.jsonl")):
        lines = jsonl.read_text().splitlines()
        out_lines = []
        ep_stats = {"enriched": 0, "skipped": 0, "no_cfg": 0}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            already_done = rec.get("schema") and rec.get("intent_category")
            if already_done and not force:
                out_lines.append(line)
                ep_stats["skipped"] += 1
                continue

            api_call = rec.get("api_call", {})
            if "steps" in api_call:
                endpoint_str = (api_call["steps"][0].get("endpoint", "")
                                if api_call["steps"] else "")
            else:
                endpoint_str = api_call.get("endpoint", "")

            cfg = _match_config(endpoint_str, path_map)
            if cfg is None:
                out_lines.append(line)
                ep_stats["no_cfg"] += 1
                continue

            rec["schema"] = extract_schema(cfg)
            rec["intent_category"] = infer_intent(api_call, cfg.get("path_params") or {})
            out_lines.append(json.dumps(rec))
            ep_stats["enriched"] += 1

        jsonl.write_text("\n".join(out_lines) + "\n")
        ep = jsonl.parent.name
        print(f"  {ep:<28} enriched={ep_stats['enriched']:4d}  "
              f"skipped={ep_stats['skipped']:4d}  no_cfg={ep_stats['no_cfg']:2d}")
        for k in stats:
            stats[k] += ep_stats[k]

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor-dir", required=True, type=Path,
                        help="Vendor API directory (e.g. apis/videoamp)")
    parser.add_argument("--data-dir", default=None, type=Path,
                        help="Override data directory (default: data/<vendor-name>)")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich records that already have schema/intent fields")
    args = parser.parse_args()

    vendor_dir = args.vendor_dir.resolve()
    vendor_name = vendor_dir.name
    data_dir = (args.data_dir or (_REPO / "data" / vendor_name)).resolve()

    if not vendor_dir.is_dir():
        sys.exit(f"vendor-dir not found: {vendor_dir}")
    if not data_dir.is_dir():
        sys.exit(f"data-dir not found: {data_dir}")

    path_map = _build_path_map(vendor_dir)
    print(f"Enriching {data_dir} using {len(path_map)} endpoint configs...\n")

    stats = enrich_dir(data_dir, path_map, force=args.force)

    print(f"\nTotal enriched:  {stats['enriched']}")
    print(f"Total skipped:   {stats['skipped']} (already had fields)")
    print(f"Total no config: {stats['no_cfg']} (could not match endpoint)")


if __name__ == "__main__":
    main()
