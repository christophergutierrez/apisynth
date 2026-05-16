#!/usr/bin/env python3
"""
STaR bootstrap: run the trained model against prompts, verify against the
live API, and write successful (prompt → thinking + api_call) pairs back
to training.jsonl format.

The model's own correct reasoning traces become training data — the system
improves itself with each production cycle.

Usage:
    # Run against vendor's canonical prompts:
    python scripts/bootstrap_traces.py \
        --model <adapter-name> \
        --vllm-url http://<host>:8000 \
        --vendor-dir apis/<vendor> \
        --output data/<vendor>/bootstrapped/training.jsonl

    # Run against a custom prompt file (one prompt per line):
    python scripts/bootstrap_traces.py \
        --model <adapter-name> \
        --vllm-url http://<host>:8000 \
        --prompts prompts.txt \
        --output data/<vendor>/bootstrapped/training.jsonl

    # Dry-run: print results without writing:
    python scripts/bootstrap_traces.py --model <adapter-name> --dry-run

Requirements:
    - vLLM server running with the trained model
    - auth token set via env_var or cli_fallback (see config.yaml)
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

# ---------------------------------------------------------------------------
# Default prompt set — override with --prompts or --vendor-dir
# A minimal generic fallback; real prompts live in apis/<vendor>/canonical_prompts.txt
# ---------------------------------------------------------------------------

CANONICAL_PROMPTS = [
    "list all items",
    "get item 1",
    "show me item 42",
    "what items are available",
    "find item 7",
]

SYSTEM = (
    "You are an API assistant. Given a natural language request, respond with the correct API call "
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

# ---------------------------------------------------------------------------
# API verification
# ---------------------------------------------------------------------------

# CLI_MAP is loaded from apis/<vendor>/cli_verification.yaml by load_vendor_config().
# Each entry maps an endpoint pattern (regex) to a CLI command list.
# See apis/example/cli_verification.yaml for the format.
CLI_MAP: dict = {}


def load_vendor_config(vendor_dir: str) -> None:
    """Load vendor-specific prompts and CLI verification map from <vendor_dir>.

    Reads:
      <vendor_dir>/canonical_prompts.txt  — one prompt per line
      <vendor_dir>/cli_verification.yaml — CLI command mappings for --verify

    Call before main() or pass --vendor-dir on the CLI.
    """
    global CANONICAL_PROMPTS, CLI_MAP
    from pathlib import Path as _Path
    vdir = _Path(vendor_dir).expanduser()

    prompts_file = vdir / "canonical_prompts.txt"
    if prompts_file.exists():
        CANONICAL_PROMPTS = [
            l.strip() for l in prompts_file.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]

    cli_file = vdir / "cli_verification.yaml"
    if cli_file.exists():
        try:
            import yaml
        except ImportError:
            import sys; sys.exit("PyYAML required: pip install pyyaml")
        data = yaml.safe_load(cli_file.read_text()) or {}
        new_map = {}
        for entry in data.get("cli_map", []):
            pattern = entry["pattern"]
            cmd_template = entry["command"]
            # Build a lambda that substitutes {param} placeholders from params dict
            def make_builder(tmpl):
                def builder(p):
                    return [str(p.get(a[1:-1], a)) if a.startswith("{") and a.endswith("}") else a
                            for a in tmpl]
                return builder
            new_map[pattern] = make_builder(cmd_template)
        CLI_MAP = new_map


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
                        help="Verify each api_call against the live API (requires cli_verification.yaml)")
    parser.add_argument("--workers",   type=int, default=5)
    parser.add_argument("--vendor-dir", default=None,
                        help="Path to vendor directory (e.g. apis/videoamp) containing "
                             "canonical_prompts.txt and cli_verification.yaml")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    if args.vendor_dir:
        load_vendor_config(args.vendor_dir)

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
