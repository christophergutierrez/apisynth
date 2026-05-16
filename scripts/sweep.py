#!/usr/bin/env python3
"""
Generic sweep for any endpoint config.yaml.

- Sweeps integer params (query or path) to find valid values
- Confirms variant combinations via test API calls
- Writes results back to config.yaml status section

Usage:
    python scripts/sweep.py --config apis/<vendor>/<endpoint>/config.yaml
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

from utils import get_skip_variant, get_token


class RateLimitError(Exception):
    pass


def api_get(url: str, token: str, params: dict | None = None) -> dict | None:
    """Make a GET request to the API. Return parsed JSON body or None on 4xx/5xx.

    Raises RateLimitError on HTTP 429. Re-raises other HTTP errors.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code in (400, 403, 404, 422, 500):
            return None
        if e.code == 429:
            raise RateLimitError("429 Too Many Requests")
        raise


def path_url(cfg: dict, path_values: dict) -> str:
    """Build a URL by substituting path param values into the endpoint path template."""
    domain = "/".join(cfg["endpoint"]["base_url"].split("/")[:3])
    path = cfg["endpoint"]["path"]
    for k, v in path_values.items():
        path = path.replace(f"{{{k}}}", str(v))
    return domain + path


def do_sweep(
    param_name: str,
    sweep_cfg: dict,
    is_valid_fn,
    existing: dict,
) -> tuple[list[int], int]:
    """Sweep an integer parameter to find valid values.

    Increments from sweep_cfg['start'], calling is_valid_fn(value) for each.
    Stops after sweep_cfg['stop_after_misses'] consecutive invalid values.
    Resumes from existing['swept_through'] if a previous partial sweep exists.
    Returns (sorted list of valid values, highest value swept).
    """
    start = sweep_cfg["start"]
    stop_after_misses = sweep_cfg["stop_after_misses"]
    hint_max = existing.get("swept_through") or sweep_cfg.get("hint_max", 1000)
    valid = list(existing.get("valid_values") or [])
    resume_from = (existing.get("swept_through") or 0) + 1

    if resume_from > start:
        print(f"  Resuming {param_name} from {resume_from} (swept through {resume_from - 1})")
    else:
        print(f"  Sweeping {param_name} (stop_after_misses={stop_after_misses}, hint_max={hint_max})")

    misses, i, swept = 0, max(start, resume_from), resume_from - 1
    try:
        while misses < stop_after_misses:
            if is_valid_fn(i):
                misses = 0
                if i not in valid:
                    valid.append(i)
                    print(f"    {param_name}={i} ✓  ({len(valid)} found)")
            else:
                misses += 1
            swept = i
            i += 1
    except RateLimitError:
        print(f"  Rate limited at {param_name}={i} — saving partial progress ({len(valid)} found through {swept})")

    valid.sort()
    print(f"  {param_name}: {len(valid)} valid values through {swept}")
    return valid, swept


def variant_dims(cfg: dict, status: dict) -> list[tuple[str, object]]:
    """Return (param_name, sample_value) pairs for each non-skipped query param.

    Used to build the set of variant combinations to confirm.
    """
    skip_variant = get_skip_variant(cfg)
    dims = []
    for pname, pcfg in (cfg.get("params") or {}).items():
        if pname in skip_variant:
            continue
        if pname == "pageToken":
            dims.append(("pageToken", pcfg.get("example", "CAU=")))
        elif "values" in pcfg and pcfg["values"]:
            dims.append((pname, pcfg["values"][0]))
        else:
            valid = (status.get(pname) or {}).get("valid_values") or []
            if valid:
                dims.append((pname, valid[0]))
    return dims


def build_variant_sets(cfg: dict, status: dict) -> list[list[str]]:
    """Return all param combinations to test as variants.

    For path-param endpoints, returns a single variant with the path param.
    For list endpoints, returns the power set of (pageSize + confirmed filter params).
    """
    params_cfg = cfg.get("params") or {}
    path_params_cfg = cfg.get("path_params") or {}

    if path_params_cfg:
        return [sorted(path_params_cfg.keys())]

    if not params_cfg or "pageSize" not in params_cfg:
        return [[]]

    dims = variant_dims(cfg, status)
    always = ["pageSize"]
    result = []
    for r in range(len(dims) + 1):
        for combo in combinations(dims, r):
            result.append(sorted(always + [p for p, _ in combo]))
    return result


def confirm_variants(cfg: dict, token: str, status: dict, variant_sets: list) -> list[dict]:
    """Test each variant combination against the live API.

    For each variant, makes one API call with pageSize=1 (or a valid path param value).
    Returns a list of dicts with keys: params, confirmed (bool), target.
    Variants that return data are marked confirmed=True; others are skipped.
    """
    base_url = cfg["endpoint"]["base_url"]
    params_cfg = cfg.get("params") or {}
    path_params_cfg = cfg.get("path_params") or {}
    target = cfg["training"]["target_per_variant"]

    sample: dict[str, object] = {}
    for pname, pcfg in params_cfg.items():
        valid = (status.get(pname) or {}).get("valid_values") or []
        if valid:
            sample[pname] = valid[0]
        elif "values" in pcfg and pcfg["values"]:
            sample[pname] = pcfg["values"][0]
    for pname in path_params_cfg:
        valid = (status.get(pname) or {}).get("valid_values") or []
        if valid:
            sample[pname] = valid[0]

    confirmed = []
    print(f"\nConfirming {len(variant_sets)} variant combinations:")

    for variant in variant_sets:
        query_params: dict = {}
        path_values: dict = {}
        skip_reason = None

        for p in variant:
            if p == "pageSize":
                query_params["pageSize"] = 1
            elif p == "pageToken":
                query_params["pageToken"] = "CAU="
            elif p in path_params_cfg:
                if p not in sample:
                    skip_reason = f"no valid {p}"
                    break
                path_values[p] = sample[p]
            else:
                if p not in sample:
                    skip_reason = f"no valid {p}"
                    break
                query_params[p] = sample[p]

        if skip_reason:
            print(f"  {variant} — SKIP ({skip_reason})")
            confirmed.append({"params": variant, "confirmed": False, "target": target})
            continue

        if path_values:
            url = path_url(cfg, path_values)
            try:
                data = api_get(url, token)
            except RateLimitError:
                print(f"  {variant} — SKIP (rate limited)")
                confirmed.append({"params": variant, "confirmed": False, "target": target})
                continue
        else:
            try:
                data = api_get(base_url, token, query_params or None)
            except RateLimitError:
                print(f"  {variant} — SKIP (rate limited)")
                confirmed.append({"params": variant, "confirmed": False, "target": target})
                continue

        ok = data is not None
        print(f"  {variant} {'✓' if ok else '✗'}")
        confirmed.append({"params": variant, "confirmed": ok, "target": target})

    return confirmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["endpoint"]["name"]
    base_url = cfg["endpoint"]["base_url"]
    token = get_token(cfg)
    status = cfg.get("status") or {}
    now = datetime.now(timezone.utc).isoformat()

    print(f"\n=== {name}  ({cfg['endpoint']['path']}) ===")

    for pname, pcfg in (cfg.get("params") or {}).items():
        if "sweep" not in pcfg:
            continue
        existing = status.get(pname) or {}
        print(f"\nSweeping query param: {pname}")
        def _check_query(i, n=pname, url=base_url):
            return bool((api_get(url, token, {"pageSize": 1, n: i}) or {}).get("data"))
        valid, swept = do_sweep(pname, pcfg["sweep"], _check_query, existing)
        status[pname] = {"valid_values": valid, "swept_through": swept, "swept_at": now}

    for pname, pcfg in (cfg.get("path_params") or {}).items():
        if "sweep" not in pcfg:
            continue
        existing = status.get(pname) or {}
        if existing.get("valid_values") and existing.get("swept_through"):
            print(f"\nPath param {pname}: {len(existing['valid_values'])} values already known — skipping sweep.")
            continue
        print(f"\nSweeping path param: {pname}")
        def _check_path(i, n=pname):
            return api_get(path_url(cfg, {n: i}), token) is not None
        valid, swept = do_sweep(pname, pcfg["sweep"], _check_path, existing)
        status[pname] = {"valid_values": valid, "swept_through": swept, "swept_at": now}

    v_sets = build_variant_sets(cfg, status)
    confirmed = confirm_variants(cfg, token, status, v_sets)
    status["variants"] = confirmed

    cfg["status"] = status
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    ok = sum(1 for v in confirmed if v["confirmed"])
    print(f"\nDone — {ok}/{len(confirmed)} variants confirmed → {config_path}")


if __name__ == "__main__":
    main()
