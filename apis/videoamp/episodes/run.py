#!/usr/bin/env python3
"""
Config-driven orchestrator: reads config.yaml, counts existing records per variant,
computes deficits, estimates time, and runs qa.py to reach target_per_variant.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Run: pip install pyyaml")

_HERE = Path(__file__).parent
_REPO = _HERE.parents[2]

SCRIPT = str(_HERE / "qa.py")


def load_config() -> dict:
    with open(_HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> tuple[Path, Path]:
    vendor = cfg["endpoint"]["vendor"]
    name = cfg["endpoint"]["name"]
    data_dir = _REPO / "data" / vendor / name
    output = data_dir / "training.jsonl"
    log = data_dir / "run.log.jsonl"
    return output, log


def variant_key(params: dict) -> tuple[str, ...]:
    return tuple(sorted(params.keys()))


def count_existing(output: Path) -> dict[tuple, int]:
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


def build_questions(cfg: dict) -> list[tuple[str, dict]]:
    """
    Build (question, cli_kwargs) pairs organized by variant.
    Values drawn from status.networkId.valid_values and status.currencyOfRecord.valid_values.
    """
    status = cfg.get("status") or {}
    network_ids: list[int] = status.get("networkId", {}).get("valid_values") or []
    cor_values: list[int] = status.get("currencyOfRecord", {}).get("valid_values") or []

    # Sample page tokens (known-valid pagination tokens)
    page_tokens = ["CAU="]

    questions: list[tuple[str, dict]] = []

    # --- Variant: {pageSize} ---
    sizes = [1, 3, 5, 10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
    phrasings = [
        ("Show me just {n} episode", "page_size"),
        ("Give me {n} episodes", "page_size"),
        ("List {n} episodes", "page_size"),
        ("Fetch {n} episodes", "page_size"),
        ("Get {n} episodes", "page_size"),
        ("Retrieve {n} episodes", "page_size"),
        ("Pull {n} episodes", "page_size"),
        ("Return {n} episodes", "page_size"),
        ("I need {n} episodes", "page_size"),
        ("Show {n} episode results", "page_size"),
    ]
    for n in sizes:
        for tmpl, _ in phrasings:
            label = "episode" if n == 1 else "episodes"
            q = tmpl.replace("{n}", str(n)).replace("episode", label if "{n}" not in tmpl else label)
            questions.append((q, {"page_size": n}))
    # Naturalistic
    questions += [
        ("List episodes", {}),
        ("Show me some episodes", {}),
        ("Get episodes from the API", {}),
        ("What episodes are available?", {}),
        ("Retrieve the default episode listing", {}),
        ("Browse episodes", {}),
        ("Pull the episode list", {}),
        ("Show all available episodes", {}),
        ("Fetch episode data", {}),
        ("Give me a small sample, just 3 episodes", {"page_size": 3}),
        ("Quick peek at episodes, just 1 result", {"page_size": 1}),
        ("I need a large batch — 1000 episodes", {"page_size": 1000}),
        ("Get a big batch of 500 episodes", {"page_size": 500}),
        ("List all episode data, 100 per page", {"page_size": 100}),
    ]

    # --- Variant: {pageSize, networkId} ---
    size_network_pairs = [
        (1, "page_size"), (5, "page_size"), (10, "page_size"), (20, "page_size"),
        (25, "page_size"), (50, "page_size"), (100, "page_size"), (200, "page_size"),
        (500, "page_size"), (1000, "page_size"),
    ]
    net_phrasings = [
        "Get {n} episodes from network {nid}",
        "Show {n} episodes on network {nid}",
        "List {n} episodes from network {nid}",
        "Fetch {n} episodes for network {nid}",
        "Retrieve {n} episodes from network {nid}",
        "Pull {n} episodes on network {nid}",
        "Give me {n} episodes from network {nid}",
        "I need {n} episodes from network {nid}",
    ]
    for nid in network_ids:
        for n, _ in size_network_pairs:
            for tmpl in net_phrasings:
                label = "episode" if n == 1 else "episodes"
                q = tmpl.replace("{n}", str(n)).replace("{nid}", str(nid)).replace("episodes", label)
                questions.append((q, {"page_size": n, "network_id": nid}))
        # Also naturalistic (no explicit size)
        questions += [
            (f"Episodes from network {nid}", {"network_id": nid}),
            (f"Get episodes from network {nid}", {"network_id": nid}),
            (f"Show episodes on network {nid}", {"network_id": nid}),
            (f"List episodes from network {nid}", {"network_id": nid}),
            (f"Fetch episodes for network {nid}", {"network_id": nid}),
            (f"What episodes are on network {nid}?", {"network_id": nid}),
            (f"Get all episodes from network {nid}", {"network_id": nid}),
        ]

    # --- Variant: {pageSize, currencyOfRecord} ---
    cor_phrasings = [
        "List {n} episodes with currency of record {cor}",
        "Get {n} episodes using CoR {cor}",
        "Fetch {n} episodes for CoR {cor}",
        "Show {n} episodes with CoR {cor}",
        "Retrieve {n} episodes with currency of record {cor}",
        "Give me {n} episodes using currency {cor}",
        "Pull {n} episodes for currency of record {cor}",
        "I need {n} episodes with CoR {cor}",
    ]
    for cor in cor_values:
        for n in [1, 5, 10, 20, 25, 50, 100, 200, 500, 1000]:
            for tmpl in cor_phrasings:
                label = "episode" if n == 1 else "episodes"
                q = tmpl.replace("{n}", str(n)).replace("{cor}", str(cor)).replace("episodes", label)
                questions.append((q, {"page_size": n, "cor": cor}))
        # Naturalistic (no explicit size)
        questions += [
            (f"List episodes with currency of record {cor}", {"cor": cor}),
            (f"Get episodes using CoR {cor}", {"cor": cor}),
            (f"Show episodes with CoR {cor}", {"cor": cor}),
            (f"Fetch episodes for currency of record {cor}", {"cor": cor}),
            (f"Episodes with CoR {cor}", {"cor": cor}),
        ]

    # --- Variant: {pageSize, networkId, currencyOfRecord} ---
    all3_phrasings = [
        "Get {n} episodes from network {nid} with CoR {cor}",
        "Show {n} episodes from network {nid} with currency {cor}",
        "List {n} episodes on network {nid} for CoR {cor}",
        "Fetch {n} episodes from network {nid} with CoR {cor}",
        "Retrieve {n} episodes from network {nid} using currency of record {cor}",
        "Give me {n} episodes from network {nid} with CoR {cor}",
        "Pull {n} episodes on network {nid} for currency {cor}",
        "I need {n} episodes from network {nid} with currency of record {cor}",
    ]
    for nid in network_ids:
        for cor in cor_values:
            for n in [1, 5, 10, 20, 25, 50, 100, 200, 500]:
                for tmpl in all3_phrasings:
                    label = "episode" if n == 1 else "episodes"
                    q = (tmpl.replace("{n}", str(n))
                             .replace("{nid}", str(nid))
                             .replace("{cor}", str(cor))
                             .replace("episodes", label))
                    questions.append((q, {"page_size": n, "network_id": nid, "cor": cor}))
            # Naturalistic
            questions += [
                (f"Episodes from network {nid} with CoR {cor}", {"network_id": nid, "cor": cor}),
                (f"Get episodes from network {nid} using currency {cor}", {"network_id": nid, "cor": cor}),
                (f"List network {nid} episodes for CoR {cor}", {"network_id": nid, "cor": cor}),
                (f"Show network {nid} episodes using CoR {cor}", {"network_id": nid, "cor": cor}),
                (f"Get network {nid} episodes with currency of record {cor}", {"network_id": nid, "cor": cor}),
            ]

    # --- Variant: {pageSize, pageToken} ---
    for tok in page_tokens:
        for n in [1, 5, 10, 20, 25, 50, 100, 200, 500, 1000]:
            questions += [
                (f"Get {n} episodes starting from page token {tok}", {"page_token": tok, "page_size": n}),
                (f"Continue listing episodes, token {tok}, {n} per page", {"page_token": tok, "page_size": n}),
                (f"Fetch {n} episodes with page token {tok}", {"page_token": tok, "page_size": n}),
                (f"Paginate to the next batch, token {tok}, {n} episodes", {"page_token": tok, "page_size": n}),
                (f"Show {n} episodes from token {tok}", {"page_token": tok, "page_size": n}),
                (f"List {n} more episodes, token {tok}", {"page_token": tok, "page_size": n}),
                (f"Retrieve {n} episodes using page token {tok}", {"page_token": tok, "page_size": n}),
                (f"Give me {n} episodes starting at token {tok}", {"page_token": tok, "page_size": n}),
            ]
        questions += [
            (f"Get the next page of episodes, token {tok}", {"page_token": tok}),
            (f"Continue listing episodes from where I left off, token {tok}", {"page_token": tok}),
            (f"Next page of episodes, token {tok}", {"page_token": tok}),
            (f"Resume episode list at token {tok}", {"page_token": tok}),
            (f"Fetch more episodes, page token {tok}", {"page_token": tok}),
        ]

    # --- Variant: {pageSize, pageToken, networkId} ---
    for tok in page_tokens:
        for nid in network_ids:
            for n in [1, 5, 10, 20, 25, 50, 100, 200]:
                questions += [
                    (f"Get {n} episodes from network {nid}, page token {tok}", {"page_token": tok, "network_id": nid, "page_size": n}),
                    (f"Continue network {nid} episodes from token {tok}, {n} per page", {"page_token": tok, "network_id": nid, "page_size": n}),
                    (f"Fetch {n} more episodes from network {nid}, token {tok}", {"page_token": tok, "network_id": nid, "page_size": n}),
                    (f"Show {n} network {nid} episodes starting at token {tok}", {"page_token": tok, "network_id": nid, "page_size": n}),
                    (f"List {n} episodes on network {nid} from page token {tok}", {"page_token": tok, "network_id": nid, "page_size": n}),
                ]
            questions += [
                (f"Get page 2 of episodes for network {nid}, token {tok}", {"network_id": nid, "page_token": tok}),
                (f"Continue network {nid} episodes with page token {tok}", {"network_id": nid, "page_token": tok}),
                (f"Next page of network {nid} episodes, token {tok}", {"network_id": nid, "page_token": tok}),
                (f"Resume network {nid} episode list at token {tok}", {"network_id": nid, "page_token": tok}),
            ]

    # --- Variant: {pageSize, pageToken, currencyOfRecord} ---
    for tok in page_tokens:
        for cor in cor_values:
            for n in [1, 5, 10, 20, 25, 50, 100, 200]:
                questions += [
                    (f"Get {n} CoR {cor} episodes, page token {tok}", {"page_token": tok, "cor": cor, "page_size": n}),
                    (f"Continue CoR {cor} episodes from token {tok}, {n} per page", {"page_token": tok, "cor": cor, "page_size": n}),
                    (f"Fetch {n} more episodes with currency {cor}, token {tok}", {"page_token": tok, "cor": cor, "page_size": n}),
                    (f"Show {n} episodes with CoR {cor} starting at token {tok}", {"page_token": tok, "cor": cor, "page_size": n}),
                    (f"List {n} episodes for currency of record {cor} from page token {tok}", {"page_token": tok, "cor": cor, "page_size": n}),
                ]
            questions += [
                (f"Next page of CoR {cor} episodes, token {tok}", {"cor": cor, "page_token": tok}),
                (f"Continue currency {cor} episodes from token {tok}", {"cor": cor, "page_token": tok}),
                (f"Resume CoR {cor} episode list at token {tok}", {"cor": cor, "page_token": tok}),
                (f"Get more CoR {cor} episodes, page token {tok}", {"cor": cor, "page_token": tok}),
            ]

    # --- Variant: {pageSize, pageToken, networkId, currencyOfRecord} ---
    for tok in page_tokens:
        for nid in network_ids:
            for cor in cor_values:
                for n in [1, 5, 10, 20, 25, 50, 100]:
                    questions += [
                        (f"Get {n} episodes from network {nid} with CoR {cor}, token {tok}", {"page_token": tok, "network_id": nid, "cor": cor, "page_size": n}),
                        (f"Continue network {nid} CoR {cor} episodes from token {tok}, {n} per page", {"page_token": tok, "network_id": nid, "cor": cor, "page_size": n}),
                        (f"Fetch {n} more network {nid} episodes with currency {cor}, token {tok}", {"page_token": tok, "network_id": nid, "cor": cor, "page_size": n}),
                        (f"Show {n} episodes from network {nid} with CoR {cor} at token {tok}", {"page_token": tok, "network_id": nid, "cor": cor, "page_size": n}),
                    ]
                questions += [
                    (f"Next batch from network {nid} with CoR {cor}, token {tok}", {"network_id": nid, "cor": cor, "page_token": tok}),
                    (f"Continue network {nid} CoR {cor} episodes, token {tok}", {"network_id": nid, "cor": cor, "page_token": tok}),
                    (f"Get next page from network {nid} with currency {cor}, token {tok}", {"network_id": nid, "cor": cor, "page_token": tok}),
                ]

    return questions


def build_cmd(output: str, log: str, question_file: str, kwargs: dict) -> list[str]:
    cmd = [sys.executable, SCRIPT, "--question", question_file, "--output", output, "--log", log]
    if "page_size" in kwargs:
        cmd += ["--page-size", str(kwargs["page_size"])]
    if "page_token" in kwargs:
        cmd += ["--page-token", kwargs["page_token"]]
    if "network_id" in kwargs:
        cmd += ["--network-id", str(kwargs["network_id"])]
    if "cor" in kwargs:
        cmd += ["--cor", str(kwargs["cor"])]
    return cmd


def run_question(cmd: list[str], question: str, question_num: int, total: int) -> tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        last_line = result.stdout.strip().split("\n")[-1]
        return True, f"[{question_num:4d}/{total}] OK   — {last_line}"
    else:
        err = result.stderr.strip() or result.stdout.strip()
        return False, f"[{question_num:4d}/{total}] FAIL — {question[:60]!r}\n           {err[:120]}"


def main():
    cfg = load_config()
    output, log = resolve_paths(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)

    target = cfg["training"]["target_per_variant"]
    limits = cfg["limits"]
    workers = limits["workers"]
    per_tenant_rph = limits["per_tenant_rph"]
    target_rpm = (per_tenant_rph / 60) * 0.5 / workers

    # Count existing records per variant
    existing_counts = count_existing(output)

    # Get confirmed variants from config
    status = cfg.get("status") or {}
    config_variants = status.get("variants") or []
    if not config_variants:
        sys.exit("Error: no variants in config status. Run sweep.py first.")

    # Compute deficits
    variant_keys_map = {
        tuple(sorted(v["params"])): v
        for v in config_variants
        if v.get("confirmed")
    }

    total_deficit = 0
    print(f"\nVariant status (target: {target} each):")
    for vkey, v in sorted(variant_keys_map.items()):
        have = existing_counts.get(vkey, 0)
        deficit = max(0, target - have)
        total_deficit += deficit
        bar = f"{have}/{target}"
        status_str = "DONE" if deficit == 0 else f"need {deficit}"
        print(f"  {str(list(vkey)):<60} {bar:>8}  {status_str}")

    if total_deficit == 0:
        print(f"\nAll variants at target ({target}). Nothing to do.")
        return

    # Time estimate
    est_minutes = total_deficit / (target_rpm * workers)
    est_str = f"{est_minutes:.1f} min" if est_minutes < 60 else f"{est_minutes / 60:.1f} hr"
    print(f"\nTotal deficit: {total_deficit} records")
    print(f"Effective rate: ~{target_rpm * workers:.0f} req/min with {workers} workers")
    print(f"Estimated time: {est_str}")

    if est_minutes > 60:
        answer = input(f"\nEstimated time > 1 hour ({est_str}). Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Build all candidate questions, filter to deficit variants only
    all_questions = build_questions(cfg)

    # Group by variant key
    from collections import defaultdict as dd
    by_variant: dict[tuple, list] = dd(list)
    for q, kwargs in all_questions:
        # Determine which variant this maps to
        param_keys = ["pageSize"]  # always present
        if "network_id" in kwargs:
            param_keys.append("networkId")
        if "cor" in kwargs:
            param_keys.append("currencyOfRecord")
        if "page_token" in kwargs:
            param_keys.append("pageToken")
        vkey = tuple(sorted(param_keys))
        if vkey in variant_keys_map:
            by_variant[vkey].append((q, kwargs))

    # Select questions to run: up to deficit per variant, de-duplicated
    to_run: list[tuple[str, dict]] = []
    for vkey, v in sorted(variant_keys_map.items()):
        have = existing_counts.get(vkey, 0)
        deficit = max(0, target - have)
        if deficit == 0:
            continue
        candidates = by_variant.get(vkey, [])
        # Deduplicate by question text
        seen: set[str] = set()
        unique_candidates = []
        for q, kw in candidates:
            if q not in seen:
                seen.add(q)
                unique_candidates.append((q, kw))
        to_run.extend(unique_candidates[:deficit])

    if not to_run:
        print("\nNo candidate questions available. Expand build_questions() pool.")
        return

    # Deduplicate across variants (same question text)
    seen_global: set[str] = set()
    to_run_deduped = []
    for q, kw in to_run:
        if q not in seen_global:
            seen_global.add(q)
            to_run_deduped.append((q, kw))
    to_run = to_run_deduped

    print(f"\nRunning {len(to_run)} questions across {workers} workers...\n")

    passed = 0
    failed = 0
    total = len(to_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        qfiles = [os.path.join(tmpdir, f"q{i}.txt") for i in range(workers)]
        for qf in qfiles:
            open(qf, "w").close()

        if workers == 1:
            for i, (question, kwargs) in enumerate(to_run, 1):
                with open(qfiles[0], "w") as f:
                    f.write(question)
                cmd = build_cmd(str(output), str(log), qfiles[0], kwargs)
                ok, msg = run_question(cmd, question, i, total)
                print(msg)
                if ok:
                    passed += 1
                else:
                    failed += 1
        else:
            import itertools
            qfile_cycle = itertools.cycle(range(len(qfiles)))
            futures_map = {}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i, (question, kwargs) in enumerate(to_run, 1):
                    slot = next(qfile_cycle)
                    qf = os.path.join(tmpdir, f"q_{i}.txt")
                    with open(qf, "w") as f:
                        f.write(question)
                    cmd = build_cmd(str(output), str(log), qf, kwargs)
                    future = executor.submit(run_question, cmd, question, i, total)
                    futures_map[future] = i

                for future in as_completed(futures_map):
                    ok, msg = future.result()
                    print(msg)
                    if ok:
                        passed += 1
                    else:
                        failed += 1

    print(f"\nDone: {passed} passed, {failed} failed → {output}")


if __name__ == "__main__":
    main()
