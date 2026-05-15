#!/usr/bin/env python3
"""
Generic training data generator for any endpoint config.yaml.

Reads the config, counts existing records per confirmed variant, computes
deficits, makes real API calls to validate, and writes JSONL training records.

Usage:
    python scripts/run.py --config apis/<vendor>/<endpoint>/config.yaml
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

from utils import get_skip_filter, humanize, singular, PAGE_SIZES

_REPO = Path(__file__).parents[1]
_write_lock = threading.Lock()
_CHAINED = "__chained__"


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
    sys.exit(f"No token. Set {auth['env_var']} or configure the CLI fallback.")


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
                if "steps" in rec.get("api_call", {}):
                    counts[(_CHAINED,)] += 1
                else:
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
    skip_filter = get_skip_filter(cfg)

    path_params_cfg = cfg.get("path_params") or {}
    params_cfg = cfg.get("params") or {}

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
        for fn in phrasing_fns:
            for id_val in valid_ids:
                q, p = fn(id_val)
                add(q, p)
        return results[:target]

    # ── Parameterless endpoint ─────────────────────────────────────────────
    if not params_cfg:
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
                "Fetch my profile", "Return my user info", "What is my user account?",
                "Display my account", "Load my profile", "Check who I am",
                "Get the current user", "Show the logged-in user", "Who is logged in?",
                "Get authenticated user", "Retrieve my profile", "My account info",
                "Show me my user", "Get my user", "Fetch current user",
                "What account am I?", "User info", "Profile info",
                "Get user", "Show me who I am", "Retrieve current user info",
                "Access my profile", "View my user profile", "Read my account",
                "My user info", "What is my profile?", "Get account details",
                "Fetch user profile", "Return my profile", "Show my user profile",
                "Current user", "Logged in user", "Get my account info",
                "Retrieve my account", "Pull up my profile", "What user am I?",
                "Show account info", "Get profile", "Check my account",
                "Fetch my account", "View my profile", "My account details",
                "Who is the logged in user?", "What's my user info?", "Get my info",
                "Show me my details", "Retrieve my user", "Current user details",
                "Get current account", "My user account", "Show user details",
                "Fetch my user", "Account info", "User profile",
                "What are my details?", "Show my details", "Get logged in user",
                "Who's the current user?", "My info", "Get my details",
                "Show me my account info", "Retrieve me", "Pull my user profile",
                "Access my account", "View my account", "Load my account",
                "Check my user", "Display my profile", "Display my user info",
                "Get my current profile", "Fetch my current user", "Show current account",
                "Retrieve logged in user", "Get my user account", "Show my user account",
                "What is my account?", "Get my user info", "Fetch me",
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
    filter_params = [p for p in variant_params if p not in skip_filter]
    fval_lists = _filter_value_lists(cfg, status, filter_params)

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

    for fc in filter_combos:
        p_bare = dict(fc)
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


# ── Parent ID collection ───────────────────────────────────────────────────

def collect_parent_ids(cfg: dict, token: str, target: int = 50) -> list:
    parent = cfg.get("parent", {})
    base_url = parent["base_url"]
    id_field = parent["id_field"]

    collected = []
    page_token = None
    while len(collected) < target:
        params = {"pageSize": min(50, target - len(collected))}
        if page_token:
            params["pageToken"] = page_token
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            print(f"  WARNING: error fetching parent IDs from {base_url}: {exc}")
            break
        items = body.get("data") or body.get("results") or []
        for item in items:
            val = item.get(id_field)
            if val is not None and val not in collected:
                collected.append(val)
        page_token = body.get("paging", {}).get("nextPageToken") or body.get("next_page_token")
        if not page_token or not items:
            break
    return collected


def gen_chained_questions(cfg: dict, target: int) -> list:
    name = cfg["endpoint"]["name"]
    resource = humanize(name)
    sing = singular(resource)

    results = []
    seen: set[str] = set()

    def add(q: str) -> None:
        if q not in seen and len(results) < target * 4:
            seen.add(q)
            results.append((q, {_CHAINED: True}))

    for q in [
        f"show me a {sing}", f"get a {sing}", f"fetch a {sing}", f"retrieve a {sing}",
        f"show me {sing} details", f"get {sing} details", f"fetch {sing} details",
        f"look up a {sing}", f"find a {sing}", f"show {sing} info",
        f"get {sing} info", f"get {sing} information", f"show me {sing} information",
        f"pull a {sing}", f"give me a {sing}", f"I need {sing} details",
        f"show {sing}", f"get {sing}", f"fetch {sing}", f"retrieve {sing}",
        f"load a {sing}", f"access {sing} details", f"view {sing} details",
        f"check {sing} details", f"display {sing} details", f"read {sing} details",
        f"show me one {sing}", f"get one {sing}", f"fetch one {sing}",
        f"retrieve one {sing}", f"pull one {sing}", f"show me the first {sing}",
        f"get the first {sing}", f"fetch the first {sing}",
        f"show me my {sing}", f"get my {sing}", f"fetch my {sing}",
        f"retrieve my {sing}", f"show me the latest {sing}", f"get the latest {sing}",
        f"fetch the latest {sing}", f"get the most recent {sing}",
        f"show the most recent {sing}", f"I want to see a {sing}",
        f"can I see a {sing}?", f"can you show me a {sing}?",
        f"tell me about a {sing}", f"give me details on a {sing}",
        f"I need to look at a {sing}", f"inspect a {sing}", f"review a {sing}",
        f"show a {sing}", f"get a {sing} record", f"fetch a {sing} record",
        f"retrieve a {sing} record", f"get {sing} data", f"show {sing} data",
        f"fetch {sing} data", f"retrieve {sing} data", f"pick a {sing}",
        f"get {sing} by ID", f"fetch {sing} by ID", f"show {sing} by ID",
        f"look up {sing} by ID", f"find {sing} by ID", f"retrieve {sing} by ID",
        f"get the {sing}", f"show the {sing}", f"fetch the {sing}",
        f"retrieve the {sing}", f"pull the {sing}", f"open a {sing}",
        f"examine a {sing}", f"show me {sing}", f"get me a {sing}",
        f"fetch me a {sing}", f"pull me a {sing}", f"give me {sing} info",
        f"get {sing} details for me", f"show {sing} details for me",
        f"what does a {sing} look like?", f"what's in a {sing}?",
        f"show me the {sing}", f"get me the {sing}",
    ]:
        add(q)

    return results[:target]


def make_chained_record(cfg: dict, question: str) -> dict:
    parent = cfg["parent"]
    path_pname = list(cfg["path_params"].keys())[0]
    id_field = parent["id_field"]
    return {
        "question": question,
        "api_call": {
            "steps": [
                {
                    "endpoint": parent["endpoint"],
                    "params": {},
                },
                {
                    "endpoint": f"{cfg['endpoint']['method']} {cfg['endpoint']['path']}",
                    "params": {path_pname: f"{{{{steps.0.{id_field}}}}}"},
                },
            ]
        },
    }


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
    if params.get(_CHAINED):
        parent = cfg.get("parent", {})
        req = urllib.request.Request(
            parent["base_url"], headers={"Authorization": f"Bearer {token}"}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req) as resp:
                ok = resp.status < 400
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return False, "RATE_LIMITED"
            ok = False
        elapsed = round(time.perf_counter() - t0, 3)
        if ok:
            record = make_chained_record(cfg, question)
            with _write_lock:
                with open(output, "a") as f:
                    f.write(json.dumps(record) + "\n")
            return True, f"OK   {elapsed:.2f}s  {question[:80]}"
        return False, f"FAIL        {question[:80]}"

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
        with _write_lock:
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

    parent_cfg = cfg.get("parent")
    if parent_cfg and cfg.get("path_params"):
        path_pname = list(cfg["path_params"].keys())[0]
        existing_ids = (status.get(path_pname) or {}).get("valid_values") or []
        if len(existing_ids) < 20:
            print(f"Collecting IDs from parent ({parent_cfg['endpoint']})...")
            fresh_ids = collect_parent_ids(cfg, token, target=50)
            if fresh_ids:
                status.setdefault(path_pname, {})["valid_values"] = fresh_ids
                print(f"  Found {len(fresh_ids)} IDs")
                with open(config_path, "w") as f:
                    cfg["status"] = status
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            else:
                print("  WARNING: parent returned no IDs")

    confirmed_variants = [
        v for v in (status.get("variants") or [])
        if v.get("confirmed") and "pageToken" not in v["params"]
    ]
    if not confirmed_variants:
        sys.exit("No confirmed variants. Run sweep.py first.")

    existing = count_existing(output)

    print(f"\n=== {name} ===")
    print(f"Variant status (target: {target} each):")

    tasks: list[tuple[str, dict, tuple]] = []

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

        added = 0
        for q, p in questions:
            if added >= deficit:
                break
            tasks.append((q, p, vkey))
            added += 1

        if added < deficit:
            print(f"    WARNING: only generated {added}/{deficit} questions for {vkey}")

    parent_cfg = cfg.get("parent")
    if parent_cfg:
        chained_key = (_CHAINED,)
        have_chained = existing.get(chained_key, 0)
        deficit = max(0, target - have_chained)
        print(f"  {'[chained two-step]':<55} {have_chained}/{target}  {'DONE' if deficit == 0 else f'need {deficit}'}")
        if deficit > 0:
            chained_qs = gen_chained_questions(cfg, target)
            added = 0
            for q, p in chained_qs:
                if added >= deficit:
                    break
                tasks.append((q, p, chained_key))
                added += 1
            if added < deficit:
                print(f"    WARNING: only generated {added}/{deficit} chained questions")

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
