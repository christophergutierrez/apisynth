#!/usr/bin/env python3
"""
Sweep sweepable params (networkId, programId) to find valid values,
then confirm all 8 variant combinations, and write results to config.yaml status section.

Usage:
    python sweep.py [--config config.yaml]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

_HERE = Path(__file__).parent


def get_token(cfg: dict) -> str:
    auth = cfg["auth"]
    token = os.environ.get(auth["env_var"])
    if token:
        return token
    try:
        result = subprocess.run(
            auth["cli_fallback"].split(),
            capture_output=True, text=True, check=True,
        )
        token = result.stdout.strip()
        if token:
            return token
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    sys.exit(f"Error: no access token found. Set {auth['env_var']} or run `videoamp login`.")


def api_call(base_url: str, token: str, params: dict) -> dict | None:
    url = base_url
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return None
        raise


def sweep_integer_param(
    param_name: str,
    sweep_cfg: dict,
    base_url: str,
    token: str,
    existing: dict,
) -> tuple[list[int], int]:
    """
    Sweep an integer param from start upward, stopping after stop_after_misses consecutive
    IDs that return no results. Returns (valid_values, swept_through).
    """
    start = sweep_cfg["start"]
    stop_after_misses = sweep_cfg["stop_after_misses"]
    hint_max = existing.get("swept_through") or sweep_cfg["hint_max"]

    valid = list(existing.get("valid_values") or [])
    resume_from = (existing.get("swept_through") or 0) + 1

    if resume_from > 1:
        print(f"  Resuming {param_name} sweep from {resume_from} (already swept through {resume_from - 1})")
    else:
        print(f"  Sweeping {param_name} (start={start}, stop_after_misses={stop_after_misses}, hint_max={hint_max})")

    misses = 0
    i = max(start, resume_from)
    swept_through = resume_from - 1

    while misses < stop_after_misses:
        data = api_call(base_url, token, {"pageSize": 1, param_name: i})
        if data is None or not data.get("data"):
            misses += 1
        else:
            misses = 0
            if i not in valid:
                valid.append(i)
                print(f"    {param_name}={i} ✓  (valid: {len(valid)} so far)")
        swept_through = i
        i += 1

    valid.sort()
    print(f"  {param_name}: found {len(valid)} valid values through {swept_through}")
    return valid, swept_through


# All 8 variant combinations (pageSize always present)
VARIANT_PARAM_SETS = [
    ["pageSize"],
    ["pageSize", "networkId"],
    ["pageSize", "currencyOfRecord"],
    ["pageSize", "networkId", "currencyOfRecord"],
    ["pageSize", "pageToken"],
    ["pageSize", "pageToken", "networkId"],
    ["pageSize", "pageToken", "currencyOfRecord"],
    ["pageSize", "pageToken", "networkId", "currencyOfRecord"],
]


def confirm_variants(base_url: str, token: str, status: dict) -> list[dict]:
    """Test each variant combination with a minimal API call to confirm it works."""
    network_ids = status.get("networkId", {}).get("valid_values") or []
    cor_values = status.get("currencyOfRecord", {}).get("valid_values") or []

    sample_network = network_ids[0] if network_ids else None
    sample_cor = cor_values[0] if cor_values else None

    confirmed_variants = []
    print("\nConfirming variant combinations:")

    for param_set in VARIANT_PARAM_SETS:
        params: dict = {"pageSize": 1}
        if "networkId" in param_set:
            if sample_network is None:
                print(f"  {param_set} — SKIP (no valid networkId found)")
                confirmed_variants.append({"params": param_set, "confirmed": False, "target": 30})
                continue
            params["networkId"] = sample_network
        if "currencyOfRecord" in param_set:
            if sample_cor is None:
                print(f"  {param_set} — SKIP (no valid currencyOfRecord found)")
                confirmed_variants.append({"params": param_set, "confirmed": False, "target": 30})
                continue
            params["currencyOfRecord"] = sample_cor
        if "pageToken" in param_set:
            params["pageToken"] = "CAU="

        data = api_call(base_url, token, params)
        ok = data is not None
        mark = "✓" if ok else "✗"
        print(f"  {sorted(param_set)} {mark}")
        confirmed_variants.append({"params": sorted(param_set), "confirmed": ok, "target": 30})

    return confirmed_variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_HERE / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_url = cfg["endpoint"]["base_url"]
    token = get_token(cfg)
    status = cfg.get("status") or {}
    now = datetime.now(timezone.utc).isoformat()

    # Sweep integer params that have sweep config
    for param_name, param_cfg in cfg["params"].items():
        if "sweep" not in param_cfg:
            continue
        existing = status.get(param_name) or {}
        print(f"\nSweeping {param_name}...")
        valid, swept_through = sweep_integer_param(
            param_name, param_cfg["sweep"], base_url, token, existing
        )
        status[param_name] = {
            "valid_values": valid,
            "swept_through": swept_through,
            "swept_at": now,
        }

    # Confirm variant combinations
    confirmed_variants = confirm_variants(base_url, token, status)
    status["variants"] = confirmed_variants

    # Write status back to config.yaml
    cfg["status"] = status
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nconfig.yaml updated with sweep results.")
    print(f"Valid networkId values: {status.get('networkId', {}).get('valid_values', [])}")
    print(f"Valid currencyOfRecord values: {status.get('currencyOfRecord', {}).get('valid_values', [])}")
    confirmed_count = sum(1 for v in confirmed_variants if v["confirmed"])
    print(f"Confirmed variants: {confirmed_count}/{len(confirmed_variants)}")


if __name__ == "__main__":
    main()
