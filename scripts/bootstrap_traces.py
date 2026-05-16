#!/usr/bin/env python3
"""
STaR bootstrap: run the trained model against prompts, verify against the
VideoAmp API, and write successful (prompt → thinking + api_call) pairs back
to training.jsonl format.

The model's own correct reasoning traces become training data — the system
improves itself with each production cycle.

Usage:
    # Run against the canonical 70 test prompts:
    python scripts/bootstrap_traces.py \
        --model api-thinking \
        --vllm-url http://192.168.2.103:8000 \
        --output data/videoamp/bootstrapped/training.jsonl

    # Run against a custom prompt file (one prompt per line):
    python scripts/bootstrap_traces.py \
        --model api-thinking \
        --vllm-url http://192.168.2.103:8000 \
        --prompts prompts.txt \
        --output data/videoamp/bootstrapped/training.jsonl

    # Dry-run: print results without writing:
    python scripts/bootstrap_traces.py --model api-thinking --dry-run

Requirements:
    - vLLM server running with the trained model
    - VIDEOAMP_ACCESS_TOKEN set (or videoamp CLI configured)
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Default prompt set — the 70 canonical test cases
# ---------------------------------------------------------------------------

CANONICAL_PROMPTS = [
    "Get program 42", "Get me", "Who am I?", "I need to see episode 99",
    "show me my measurements", "list all programs", "who is the current user",
    "show my account", "fetch program 7", "show all TV shows", "get episode 200",
    "find episode 15", "what are my ad reports", "show all measurements",
    "list all networks", "what networks are available", "show audiences",
    "what audiences do I have", "list all episodes", "browse episodes",
    "episode 5", "see episode 22", "pull up episode 42", "show episode 77",
    "episode number 100", "give me episode 33", "I want to see episode 8",
    "look at episode 11", "program 100", "what is program 55", "view program 12",
    "TV show 42", "series 7", "show me all shows", "all series", "browse shows",
    "what shows do you have", "my measurement", "my report", "show me my report",
    "list campaigns", "show my reports", "ad analytics", "reach and frequency data",
    "campaign performance", "my impressions", "TV channels", "all channels",
    "what channels exist", "channel 45", "network 12", "audience segments",
    "my segments", "demographic segments", "audience 101", "segment 55",
    "programs", "episodes", "measurements", "networks", "audiences",
    "my profile", "user information", "who owns this account", "show my details",
    "measurement 5", "get measurement 5", "network 7", "get network 7",
    "get audience 101",
]

SYSTEM = (
    "You are a VideoAmp API assistant. Given a natural language request, respond with the correct API call "
    "as a JSON object inside a code block. Think through the request before answering. "
    'For a single call use: {"endpoint": "GET /...", "params": {...}}. '
    "For a two-step call (when an ID must be fetched first) use: "
    '{"steps": [{"endpoint": "GET /...", "params": {}}, '
    '{"endpoint": "GET /.../{id}", "params": {"id": "{{steps.0.fieldName}}"}}]}.'
)

# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def call_model(vllm_url: str, model: str, prompt: str, max_tokens: int = 800) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{vllm_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1000
    content = d["choices"][0]["message"]["content"]
    usage = d.get("usage", {})
    return {"content": content, "ms": ms,
            "comp_tokens": usage.get("completion_tokens", 0)}


def extract(content: str) -> tuple[str, str]:
    """Return (thinking, answer) from model output."""
    think = ""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        think = m.group(1).strip()
        content = content[m.end():].strip()
    return think, content


def parse_api_call(answer: str) -> dict | None:
    cleaned = re.sub(r"```json\s*", "", answer).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API verification
# ---------------------------------------------------------------------------

CLI_MAP = {
    r"GET /external/v1/content/programs/\{": lambda p: ["videoamp", "content", "get-program", "--programId", str(p.get("programId", ""))],
    r"GET /external/v1/content/programs$":   lambda p: ["videoamp", "content", "list-programs"],
    r"GET /external/v1/content/episodes/\{": lambda p: ["videoamp", "content", "get-episode", "--episodeId", str(p.get("episodeId", ""))],
    r"GET /external/v1/content/episodes$":   lambda p: ["videoamp", "content", "list-episodes"],
    r"GET /external/v1/content/networks/\{": lambda p: ["videoamp", "content", "get-network", "--id", str(p.get("id", ""))],
    r"GET /external/v1/content/networks$":   lambda p: ["videoamp", "content", "list-networks"],
    r"GET /external/v1/measurements/\{":     lambda p: ["videoamp", "measurements", "get-ad", "--id", str(p.get("id", ""))],
    r"GET /external/v1/measurements$":       lambda p: ["videoamp", "measurements", "list-ad"],
    r"GET /v1/audiences/\{":                 lambda p: ["videoamp", "audiences", "get", "--id", str(p.get("id", ""))],
    r"GET /v1/audiences$":                   lambda p: ["videoamp", "audiences", "list"],
    r"GET /v1/me$":                          lambda p: ["videoamp", "me"],
    r"GET /external/v1/currency-of-record":  lambda p: ["videoamp", "currency-of-record", "list"],
    r"GET /v1/consents$":                    lambda p: ["videoamp", "consents", "list"],
}


def verify_api_call(api_call: dict) -> str:
    """Run the equivalent CLI command. Return '200' or error status."""
    if "steps" in api_call:
        # For chained calls, just verify step 0 (list)
        ep = api_call["steps"][0].get("endpoint", "")
        params = api_call["steps"][0].get("params", {})
    else:
        ep = api_call.get("endpoint", "")
        params = api_call.get("params", {})

    cmd = None
    for pattern, builder in CLI_MAP.items():
        if re.search(pattern, ep):
            cmd = builder(params)
            break

    if cmd is None:
        print(f"  WARNING: no CLI mapping for endpoint '{ep}' — skipping verification")
        return f"no-cli-mapping: {ep}"

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return "200" if r.returncode == 0 else f"exit:{r.returncode}"
    except Exception as e:
        return f"exception: {e}"
# ---------------------------------------------------------------------------
# Main bootstrap loop
# ---------------------------------------------------------------------------

def bootstrap_one(vllm_url: str, model: str, prompt: str, verify: bool) -> dict:
    try:
        result = call_model(vllm_url, model, prompt)
    except Exception as e:
        return {"prompt": prompt, "ok": False, "error": str(e)}

    thinking, answer = extract(result["content"])
    api_call = parse_api_call(answer)
    if api_call is None:
        return {"prompt": prompt, "ok": False, "error": "unparseable", "ms": result["ms"]}

    status = "skipped"
    if verify:
        status = verify_api_call(api_call)
        ok = status == "200"
    else:
        ok = True  # trust the model when not verifying

    return {
        "prompt": prompt,
        "ok": ok,
        "api_call": api_call,
        "thinking": thinking,
        "status": status,
        "ms": result["ms"],
        "comp_tokens": result["comp_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",     default="api-thinking")
    parser.add_argument("--vllm-url",  default="http://127.0.0.1:8000",
                        help="vLLM server URL (default: localhost:8000)")
    parser.add_argument("--prompts",   default=None,
                        help="File with one prompt per line (default: canonical 70)")
    parser.add_argument("--output",    default=None,
                        help="Output training.jsonl path")
    parser.add_argument("--verify",    action="store_true",
                        help="Verify each api_call against the live VideoAmp API")
    parser.add_argument("--workers",   type=int, default=5)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    if args.prompts:
        prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    else:
        prompts = CANONICAL_PROMPTS

    print(f"Model:   {args.model}")
    print(f"Server:  {args.vllm_url}")
    print(f"Prompts: {len(prompts)}")
    print(f"Verify:  {args.verify}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(bootstrap_one, args.vllm_url, args.model, p, args.verify): p
            for p in prompts
        }
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                p = futures[fut]
                print(f"  E  {p:<40s}  exception: {exc}")
                results.append({"prompt": p, "ok": False, "error": str(exc)})
                continue
            sym = "V" if r["ok"] else "X"
            status = r.get("status", "")
            ms = r.get("ms", 0)
            print(f"  {sym}  {r['prompt']:<40s}  {status:<8}  {ms:.0f}ms")
            results.append(r)

    passed = [r for r in results if r["ok"] and "api_call" in r]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(passed)}/{len(prompts)} passed")
    if failed:
        print(f"{len(failed)} failed:")
        for r in failed:
            print(f"  {r['prompt']!r}: {r.get('error','')}")

    if args.dry_run or not args.output:
        print("\nDry run — no output written.")
        return

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "a") as f:
        for r in passed:
            record = {
                "question": r["prompt"],
                "api_call": r["api_call"],
                "thinking": r["thinking"],
                "source": "bootstrap",
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    print(f"\nWrote {written} records → {out}")


if __name__ == "__main__":
    main()
