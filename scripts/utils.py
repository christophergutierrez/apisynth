"""Shared utilities for sweep.py, run.py, and gen_router_data.py."""

import os
import shlex
import subprocess
import sys

# Generic pagination/sort/search params that never form useful variant dimensions.
# Add vendor-specific params to skip via the config's top-level `skip_params` list.
_BASE_SKIP_FILTER = frozenset({
    "pageSize", "pageToken", "orderBy", "sorts", "q", "name", "query",
})

_BASE_SKIP_VARIANT = _BASE_SKIP_FILTER - {"pageToken"}

PAGE_SIZES = [1, 3, 5, 10, 20, 25, 50, 100, 200, 500, 1000]

PYYAML_REQUIRED = "Error: PyYAML required. Run: pip install pyyaml"

# JSONL record field names
FIELD_QUESTION = "question"
FIELD_API_CALL = "api_call"
FIELD_THINKING = "thinking"

# Config schema keys
CFG_VARIANTS = "variants"
CFG_CONFIRMED = "confirmed"
CFG_TARGET_PER_VARIANT = "target_per_variant"


def get_skip_filter(cfg: dict) -> frozenset:
    return _BASE_SKIP_FILTER | frozenset(cfg.get("skip_params") or [])


def get_skip_variant(cfg: dict) -> frozenset:
    return _BASE_SKIP_VARIANT | frozenset(cfg.get("skip_params") or [])


def humanize(s: str) -> str:
    return s.replace("-", " ").replace("_", " ")


def singular(s: str) -> str:
    if s.endswith("ies"):
        return s[:-3] + "y"
    if s.endswith("ses"):
        return s[:-2]
    if s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def get_token(cfg: dict) -> str:
    auth = cfg["auth"]
    token = os.environ.get(auth["env_var"])
    if token:
        return token
    try:
        r = subprocess.run(
            shlex.split(auth["cli_fallback"]),
            capture_output=True, text=True, check=True,
        )
        if r.stdout.strip():
            return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    sys.exit(f"No token. Set {auth['env_var']} or configure the CLI fallback.")
