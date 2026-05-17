"""Shared utilities for sweep.py, run.py, and gen_router_data.py."""

import os
import shlex
import subprocess
import sys
from typing import Optional

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


def extract_schema(cfg: dict) -> str:
    """Return a concise human-readable schema string for a config's endpoint.

    Included in training records so a fine-tuned model can learn to read the
    schema rather than memorise parameter names (retrieval-aware training).

    Example output:
        GET /external/v1/content/episodes
        params: pageSize (integer), networkId (integer)
        path params: episodeId (integer)
    """
    ep = cfg.get("endpoint", {})
    method = ep.get("method", "GET")
    path = ep.get("path") or ep.get("base_url", "")
    # Use only the path component when base_url contains a full URL
    if path.startswith("http"):
        from urllib.parse import urlparse
        path = urlparse(path).path

    lines = [f"{method} {path}"]

    params = cfg.get("params") or {}
    if params:
        parts = []
        for name, meta in params.items():
            ptype = meta.get("type", "string") if isinstance(meta, dict) else "string"
            parts.append(f"{name} ({ptype})")
        lines.append("params: " + ", ".join(parts))
    else:
        lines.append("params: (none)")

    path_params = cfg.get("path_params") or {}
    if path_params:
        parts = []
        for name, meta in path_params.items():
            ptype = meta.get("type", "integer") if isinstance(meta, dict) else "integer"
            parts.append(f"{name} ({ptype})")
        lines.append("path params: " + ", ".join(parts))

    return "\n".join(lines)


# Intent category constants
INTENT_BARE_LIST = "bare-list"
INTENT_PAGINATED = "paginated"
INTENT_FILTERED = "filtered"
INTENT_BY_ID = "by-id"
INTENT_CHAINED = "chained"
INTENT_NO_PARAM = "no-param"


def infer_intent(api_call: dict, path_params_cfg: Optional[dict] = None) -> str:
    """Infer the intent category of a training record from its api_call structure.

    Used to tag records for intent-stratified holdout splits and evaluation grouping.

    Categories:
        chained   — multi-step call (has "steps" key)
        by-id     — single-item lookup via path param
        no-param  — endpoint with no params (e.g. /me)
        filtered  — list endpoint with non-pageSize filter params
        paginated — list endpoint with only pageSize
        bare-list — list endpoint with no params
    """
    if "steps" in api_call:
        return INTENT_CHAINED

    path_params_cfg = path_params_cfg or {}
    params = api_call.get("params", {})

    has_path_param = any(k in path_params_cfg for k in params) if path_params_cfg else False
    if not has_path_param and path_params_cfg:
        # Check if endpoint path contains a path param placeholder
        pass
    # Detect by-id: path param present in params
    if path_params_cfg and any(k in path_params_cfg for k in params):
        return INTENT_BY_ID

    if not params:
        return INTENT_NO_PARAM

    non_pagination = {k for k in params if k not in ("pageSize", "pageToken")}
    if non_pagination:
        return INTENT_FILTERED
    if "pageSize" in params:
        return INTENT_PAGINATED
    return INTENT_BARE_LIST


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
