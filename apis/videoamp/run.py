#!/usr/bin/env python3
"""
Generic training data generator for any VideoAmp endpoint config.yaml.

Reads the config, counts existing records per confirmed variant, computes
deficits, makes real API calls to validate, and writes JSONL training records.

Usage:
    python run.py --config apis/videoamp/programs/config.yaml
    python run.py --config apis/videoamp/episode/config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

from constants import _SKIP_FILTER, humanize, singular, PAGE_SIZES

_REPO = Path(__file__).parents[2]
_write_lock = threading.Lock()


class RateLimitError(Exception):
    pass


# ── Token ──────────────────────────────────────────────────────────────────

def get_token(cfg: dict) -> str:
    auth = cfg["auth"]
    token = os.environ.get(auth["env_var"])
    if token:
        return token
    try:
        r = subprocess.run(
            auth["cli_fallback"].split(), capture_output=True, text=True, check=True,
        )
        if r.stdout.strip():
            return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    sys.exit(f"No token. Set {auth['env_var']} or run `videoamp login`.")


# ── API call ───────────────────────────────────────────────────────────────

def build_url(cfg: dict, query_params: dict, path_values: dict) -> str:
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


def api_validate(cfg: dict, token: str, query_params: dict, path_values: dict) -> bool:
    url = build_url(cfg, query_params, path_values)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimitError()
        return False


def variant_key(params: dict) -> tuple:
    return tuple(sorted(params.keys()))


def count_existing(output: Path) -> dict:
    counts: dict[tuple, int] = defaultdict(int)
    if not output.exists():
        return counts
    with open(output) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = variant_key(rec["api_call"]["params"])
                counts[key] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


# ── Question generation ────────────────────────────────────────────────────

def _filter_value_lists(cfg: dict, status: dict, filter_params: list) -> dict:
    params_cfg = cfg.get("params") or {}
    out = {}
    for fp in filter_params:
        vals = (status.get(fp) or {}).get("valid_values") or []
        if not vals:
            vals = (params_cfg.get(fp) or {}).get("values") or []
        if vals:
            out[fp] = vals
    return out


def _filter_desc(cfg: dict, filter_combo: dict) -> str:
    params_cfg = cfg.get("params") or {}
    parts = []
    for fp, val in filter_combo.items():
        label = (params_cfg.get(fp) or {}).get("label") or humanize(fp)
        parts.append(f"{label} {val}")
    return " and ".join(parts)


def gen_questions(cfg: dict, status: dict, variant_params: list, target: int) -> list:
    """
    Return list of (question_text, params_dict) for a variant.
    params_dict uses canonical API param names (camelCase).
    """
    name = cfg["endpoint"]["name"]
    resource = humanize(name)
    sing = singular(resource)

    path_params_cfg = cfg.get("path_params") or {}
    params_cfg = cfg.get("params") or {}
    variant_set = set(variant_params)

    results = []
    seen: set[str] = set()

    def add(q: str, p: dict) -> None:
        if q not in seen and len(results) < target * 4:
            seen.add(q)
            results.append((q, p))

    # ── Path-param endpoint (single-resource lookup) ──────────────────────
    if path_params_cfg:
        path_pname = list(path_params_cfg.keys())[0]
        valid_ids = (status.get(path_pname) or {}).get("valid_values") or []
        if not valid_ids:
            return []

        phrasing_fns = [
            lambda v, p=path_pname: (f"Get {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Show me {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Fetch {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Retrieve {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Look up {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Show details for {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"{resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"What is {resource} {v}?", {p: v}),
            lambda v, p=path_pname: (f"Details for {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Find {resource} with ID {v}", {p: v}),
            lambda v, p=path_pname: (f"Pull {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Give me {resource} {v}", {p: v}),
            lambda v, p=path_pname: (f"Get {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Show {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Fetch {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Return {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"I need {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Load {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Read {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"Access {sing} {v}", {p: v}),
            # Indirect phrasings — key for disambiguation from list endpoints
            lambda v, p=path_pname: (f"I need to see {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"show me the details for {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"look up {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"{sing} {v} info", {p: v}),
            lambda v, p=path_pname: (f"what's {sing} {v} about?", {p: v}),
            lambda v, p=path_pname: (f"find {sing} with id {v}", {p: v}),
            lambda v, p=path_pname: (f"pull up {sing} {v}", {p: v}),
            lambda v, p=path_pname: (f"{sing} {v} details", {p: v}),
            lambda v, p=path_pname: (f"can I see {sing} {v}?", {p: v}),
            lambda v, p=path_pname: (f"tell me about {sing} {v}", {p: v}),
        ]
        # Iterate all (phrasing × id) combos — avoids lcm-cycle trap
        for fn in phrasing_fns:
            for id_val in valid_ids:
                q, p = fn(id_val)
                add(q, p)
        return results[:target]

    # ── Parameterless endpoint ─────────────────────────────────────────────
    if not params_cfg:
        # Special-case "me" — auto-generated templates produce nonsense for this resource name
        if resource == "me":
            for q in [
                "Who am I?", "Get my user profile", "Show me my profile",
                "What's my account info?", "Get my account details", "Show my user info",
                "Who is the current user?", "What user am I logged in as?",
                "Get my profile", "Show current user", "Get logged-in user",
                "What are my user details?", "Pull my profile", "Get user info",
                "Show user account", "Retrieve my user account", "Get me",
                "Show me my account", "Get user profile", "What account am I using?",
                "Who's logged in?", "Current user info", "Show authenticated user",
                "My user details", "Tell me who I am", "pull up my user profile",
                "My profile", "Show my account", "What's my profile?", "Get current user",
            ]:
                add(q, {})
        else:
            for q in [
                f"Get {resource}", f"Show me {resource}", f"Fetch {resource}",
                f"List {resource}", f"Retrieve {resource}", f"What is {resource}?",
                f"Show {resource} details", f"Get current {resource}", f"{resource}",
                f"Show {resource} information", f"Get my {resource}",
                f"Retrieve {resource} data", f"Pull {resource}", f"Give me {resource}",
                f"Return {resource}", f"Get the {resource}",
                f"What does {resource} return?", f"Show the {resource}",
                f"Fetch current {resource}", f"Tell me about {resource}",
                f"Load {resource}", f"Read {resource}", f"Get {resource} data",
                f"Access {resource}", f"View {resource}", f"Check {resource}",
                f"Look up {resource}", f"Get {resource} info",
                f"Show all {resource}", f"Retrieve current {resource}",
                f"Display {resource}",
            ]:
                add(q, {})
        return results[:target]

    # ── List endpoint ──────────────────────────────────────────────────────
    filter_params = [p for p in variant_params if p not in _SKIP_FILTER]
    fval_lists = _filter_value_lists(cfg, status, filter_params)

    # Build filter combos: cycle through different values for each filter param
    max_fvals = max((len(v) for v in fval_lists.values()), default=1)
    filter_combos = []
    for i in range(max(1, max_fvals)):
        combo = {fp: vals[i % len(vals)] for fp, vals in fval_lists.items()}
        filter_combos.append(combo)

    for n in PAGE_SIZES:
        unit = sing if n == 1 else resource
        for fc in filter_combos:
            p = {"pageSize": n, **fc}
            fdesc = _filter_desc(cfg, fc)
            fclause = f" with {fdesc}" if fdesc else ""
            for verb in ["Get", "List", "Show", "Fetch", "Retrieve", "Pull", "Give me"]:
                add(f"{verb} {n} {unit}{fclause}", p)
            if not fdesc:
                add(f"I need {n} {unit}", p)
                add(f"Show {n} {resource} results", p)

    # Naturalistic (no explicit size — bare params, no default pageSize)
    for fc in filter_combos:
        p_bare = dict(fc)  # filter params only, no pageSize
        fdesc = _filter_desc(cfg, fc)
        fclause = f" with {fdesc}" if fdesc else ""
        for q in [
            f"List {resource}{fclause}",
            f"Show {resource}{fclause}",
            f"Get {resource}{fclause}",
            f"Fetch {resource}{fclause}",
            f"Get all {resource}{fclause}",
            f"Browse {resource}{fclause}",
        ]:
            add(q, p_bare)
        if not fdesc:
            add(f"Show me {resource}", p_bare)
            add(f"What {resource} are available?", p_bare)

    return results


# ── Write record ───────────────────────────────────────────────────────────

def make_record(cfg: dict, question: str, params: dict) -> dict:
    return {
        "question": question,
        "api_call": {
            "endpoint": f"{cfg['endpoint']['method']} {cfg['endpoint']['path']}",
            "params": params,
        },
    }


# ── Run one question ───────────────────────────────────────────────────────

def run_one(cfg: dict, token: str, output: Path, question: str, params: dict) -> tuple[bool, str]:
    path_params_cfg = cfg.get("path_params") or {}

    path_values: dict = {}
    query_params: dict = {}
    for k, v in params.items():
        if k in path_params_cfg:
            path_values[k] = v
        else:
            query_params[k] = v

    t0 = time.perf_counter()
    try:
        ok = api_validate(cfg, token, query_params, path_values)
    except RateLimitError:
        return False, "RATE_LIMITED"
    elapsed = round(time.perf_counter() - t0, 3)

    if ok:
        record = make_record(cfg, question, params)
        with open(output, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True, f"OK   {elapsed:.2f}s  {question[:80]}"
    else:
        return False, f"FAIL        {question[:80]}"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["endpoint"]["name"]
    vendor = cfg["endpoint"]["vendor"]
    target = cfg["training"]["target_per_variant"]
    workers = cfg["limits"]["workers"]

    output = _REPO / "data" / vendor / name / "training.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    token = get_token(cfg)
    status = cfg.get("status") or {}

    confirmed_variants = [
        v for v in (status.get("variants") or [])
        if v.get("confirmed") and "pageToken" not in v["params"]
    ]
    if not confirmed_variants:
        sys.exit("No confirmed variants. Run sweep.py first.")

    existing = count_existing(output)

    print(f"\n=== {name} ===")
    print(f"Variant status (target: {target} each):")

    tasks: list[tuple[str, dict, tuple]] = []  # (question, params, vkey)

    for v in confirmed_variants:
        vparams = v["params"]
        vkey = tuple(sorted(vparams))
        have = existing.get(vkey, 0)
        deficit = max(0, target - have)
        print(f"  {list(vkey)!s:<55} {have}/{target}  {'DONE' if deficit == 0 else f'need {deficit}'}")
        if deficit == 0:
            continue

        questions = gen_questions(cfg, status, vparams, target)
        if not questions:
            print(f"    WARNING: could not generate questions for {vkey}")
            continue

        # Filter to only questions not already counted, deduplicate
        added = 0
        for q, p in questions:
            if added >= deficit:
                break
            tasks.append((q, p, vkey))
            added += 1

        if added < deficit:
            print(f"    WARNING: only generated {added}/{deficit} questions for {vkey}")

    if not tasks:
        print("\nAll variants at target. Nothing to do.")
        return

    print(f"\nGenerating {len(tasks)} records across {workers} workers...")

    passed = failed = rate_limited = 0
    total = len(tasks)

    if workers == 1:
        for i, (q, p, _) in enumerate(tasks, 1):
            ok, msg = run_one(cfg, token, output, q, p)
            print(f"[{i:4d}/{total}] {msg}")
            if ok:
                passed += 1
            elif msg == "RATE_LIMITED":
                rate_limited += 1
                print("  Rate limited — pausing 60s...")
                time.sleep(60)
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fmap = {
                ex.submit(run_one, cfg, token, output, q, p): (i, q)
                for i, (q, p, _) in enumerate(tasks, 1)
            }
            for fut in as_completed(fmap):
                i, q = fmap[fut]
                ok, msg = fut.result()
                print(f"[{i:4d}/{total}] {msg}")
                if ok:
                    passed += 1
                elif msg == "RATE_LIMITED":
                    rate_limited += 1
                else:
                    failed += 1

    if rate_limited:
        print(f"\nRate limited on {rate_limited} records — re-run to fill gaps.")
    print(f"\nDone — {passed} passed, {failed} failed → {output}")


if __name__ == "__main__":
    main()
