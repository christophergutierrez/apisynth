#!/usr/bin/env python3
"""
Reads a question and CLI-specified API params, validates them against the
VideoAmp episodes endpoint, and writes a JSONL training record mapping the
question to the correct API call (not the response data).

Usage:
    python episode_qa.py --question question.txt --output training.jsonl

Token resolution order:
    1. VIDEOAMP_ACCESS_TOKEN env var
    2. `videoamp config get --key access_token` CLI
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

ENDPOINT = "GET /external/v1/content/episodes"
BASE_URL = "https://api.videoamp.dev/external/v1/content/episodes"


def get_token() -> str:
    token = os.environ.get("VIDEOAMP_ACCESS_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["videoamp", "config", "get", "--key", "access_token"],
            capture_output=True, text=True, check=True,
        )
        token = result.stdout.strip()
        if token:
            return token
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    sys.exit("Error: no access token found. Set VIDEOAMP_ACCESS_TOKEN or run `videoamp login`.")


def fetch_episodes(token: str, params: dict[str, Any]) -> dict:
    import urllib.request
    import urllib.parse

    url = BASE_URL
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def write_log(log_file: str, entry: dict) -> None:
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate episode API call training records.")
    parser.add_argument("--question", default="question.txt", help="File containing the natural language question.")
    parser.add_argument("--output", default="training.jsonl", help="Output JSONL file.")
    parser.add_argument("--log", default="episode_qa.log.jsonl", help="Log file for timing and errors.")
    parser.add_argument("--page-size", type=int, default=50, help="Number of episodes to fetch.")
    parser.add_argument("--page-token", default="", help="Pagination token for a specific page.")
    parser.add_argument("--network-id", type=int, help="Filter by network ID.")
    parser.add_argument("--program-id", type=int, help="Filter by program ID.")
    parser.add_argument("--cor", type=int, help="Filter by currency of record (e.g. 25).")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    def log_error(stage: str, err: Exception) -> None:
        write_log(args.log, {
            "status": "error",
            "started_at": started_at,
            "stage": stage,
            "error": type(err).__name__,
            "detail": str(err),
            "traceback": traceback.format_exc().strip(),
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        })

    try:
        with open(args.question) as f:
            question = f.read().strip()
    except OSError as e:
        log_error("read_question", e)
        sys.exit(f"Error reading question file: {e}")

    if not question:
        err = ValueError("question file is empty")
        log_error("read_question", err)
        sys.exit("Error: question file is empty.")

    # Build the params that represent the correct API call for this question.
    params: dict[str, Any] = {"pageSize": args.page_size}
    if args.page_token:
        params["pageToken"] = args.page_token
    if args.network_id:
        params["networkId"] = args.network_id
    if args.program_id:
        params["programId"] = args.program_id
    if args.cor:
        params["currencyOfRecord"] = args.cor

    try:
        token = get_token()
    except SystemExit as e:
        log_error("get_token", Exception(str(e)))
        raise

    # Validate the params by making the real API call.
    print(f"Validating API call (pageSize={args.page_size})...")
    t_fetch = time.perf_counter()
    try:
        data = fetch_episodes(token, params)
    except Exception as e:
        log_error("fetch_episodes", e)
        sys.exit(f"Error validating API call: {e}")
    fetch_seconds = round(time.perf_counter() - t_fetch, 3)

    episodes_returned = len(data.get("data", []))
    total_available = data.get("paging", {}).get("totalResults")

    # Training record: question → correct API call (not the response).
    record = {
        "question": question,
        "api_call": {
            "endpoint": ENDPOINT,
            "params": params,
        },
    }
    try:
        with open(args.output, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        log_error("write_output", e)
        sys.exit(f"Error writing output: {e}")

    elapsed = round(time.perf_counter() - t_start, 3)
    write_log(args.log, {
        "status": "ok",
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "fetch_seconds": fetch_seconds,
        "question_file": args.question,
        "output_file": args.output,
        "episodes_returned": episodes_returned,
        "total_available": total_available,
        "params": params,
    })

    print(f"Written to {args.output} ({elapsed}s total, {fetch_seconds}s fetch)")
    print(f"Validated: {episodes_returned} episodes returned (of {total_available:,} total)")
    print(f"Log: {args.log}")
    print(f"\nQ: {question}")
    print(f"\nAPI call: {ENDPOINT}")
    print(f"Params:   {json.dumps(params)}")


if __name__ == "__main__":
    main()
