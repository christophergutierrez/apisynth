#!/usr/bin/env python3
"""Probe the episodes API to find valid networkId and programId values."""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.parse

BASE_URL = "https://api.videoamp.dev/external/v1/content/episodes"


def get_token() -> str:
    token = os.environ.get("VIDEOAMP_ACCESS_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["videoamp", "config", "get", "--key", "access_token"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def fetch(token: str, params: dict) -> dict:
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "msg": e.reason}


def probe_ids(token: str, param_name: str, ids: list[int]) -> dict[int, int]:
    """Returns {id: count} for IDs that return > 0 results."""
    valid = {}
    for id_val in ids:
        data = fetch(token, {"pageSize": 1, param_name: id_val})
        count = data.get("paging", {}).get("totalResults", 0) if "error" not in data else -1
        if count > 0:
            valid[id_val] = count
            print(f"  {param_name}={id_val}: {count:,} episodes")
        else:
            print(f"  {param_name}={id_val}: 0 (skip)")
    return valid


token = get_token()
print("Token acquired.\n")

# Probe networkIds
print("=== networkId ===")
network_ids_to_probe = list(range(1, 20)) + [25, 30, 40, 50, 75, 100, 150, 200, 300, 500]
valid_networks = probe_ids(token, "networkId", network_ids_to_probe)

print("\n=== programId ===")
program_ids_to_probe = list(range(1, 20)) + [25, 30, 40, 50, 75, 100, 200, 500, 1000, 5000]
valid_programs = probe_ids(token, "programId", program_ids_to_probe)

print("\n=== Summary ===")
print(f"Valid networkIds: {dict(sorted(valid_networks.items()))}")
print(f"Valid programIds: {dict(sorted(valid_programs.items()))}")
