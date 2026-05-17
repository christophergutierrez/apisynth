#!/usr/bin/env python3
"""3-tier evaluation rubric for structured API call outputs.

Scores predictions against a holdout set on three independent tiers:

  Tier 1 — format:      is api_call valid JSON with the required structure?
  Tier 2 — param_f1:   F1 over parameter names (precision + recall)
  Tier 3 — executable: optional live API validation (requires --executable flag)

Usage:
    python scripts/eval.py \\
        --predictions data/<vendor>/<endpoint>/holdout.jsonl \\
        --holdout     data/<vendor>/<endpoint>/holdout.jsonl \\
        [--executable --config apis/<vendor>/<endpoint>/config.yaml]

When --predictions and --holdout are the same file the score is a self-
consistency check (should be 1.0 for format and param_f1).
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_token, PYYAML_REQUIRED


# ── Tier 1: format validity ────────────────────────────────────────────────

def format_score(api_call: object) -> float:
    """Return 1.0 if api_call has the required structure, 0.0 otherwise."""
    if not isinstance(api_call, dict):
        return 0.0
    if "steps" in api_call:
        steps = api_call["steps"]
        if not isinstance(steps, list) or len(steps) < 1:
            return 0.0
        for step in steps:
            if not isinstance(step, dict) or "endpoint" not in step or "params" not in step:
                return 0.0
        return 1.0
    if "endpoint" not in api_call or "params" not in api_call:
        return 0.0
    if not isinstance(api_call["params"], dict):
        return 0.0
    return 1.0


# ── Tier 2: param F1 ───────────────────────────────────────────────────────

def param_f1(predicted: dict, expected: dict) -> float:
    """F1 over parameter *names* between predicted and expected api_call.

    For chained calls, computes F1 over the union of all step param names.
    Returns 1.0 when both have no params (correct no-op).
    """
    def _param_names(api_call: dict) -> set:
        if "steps" in api_call:
            names = set()
            for step in api_call.get("steps", []):
                names |= set(step.get("params", {}).keys())
            return names
        return set(api_call.get("params", {}).keys())

    pred_names = _param_names(predicted)
    exp_names = _param_names(expected)

    if not pred_names and not exp_names:
        return 1.0

    tp = len(pred_names & exp_names)
    precision = tp / len(pred_names) if pred_names else 0.0
    recall = tp / len(exp_names) if exp_names else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Tier 3: executability ──────────────────────────────────────────────────

def executable_score(api_call: dict, cfg: dict, token: str) -> bool | None:
    """Return True if the api_call executes successfully against the live API.

    Returns None if the call cannot be executed (e.g. chained calls, missing config).
    """
    if "steps" in api_call:
        return None  # Only validates step 0 for now

    path_params_cfg = cfg.get("path_params") or {}
    path_values = {}
    query_params = {}
    for k, v in api_call.get("params", {}).items():
        if k in path_params_cfg:
            path_values[k] = v
        else:
            query_params[k] = v

    ep = cfg["endpoint"]
    base = ep.get("base_url", "")
    if not base.startswith("http"):
        return None

    if path_values:
        domain = "/".join(base.split("/")[:3])
        path = ep.get("path", "")
        for k, v in path_values.items():
            path = path.replace(f"{{{k}}}", str(v))
        url = domain + path
    else:
        url = base

    if query_params:
        url = f"{url}?{urllib.parse.urlencode(query_params)}"

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        return e.code < 400
    except Exception:
        return False


# ── Scoring ────────────────────────────────────────────────────────────────

def score_record(
    predicted_api_call: object,
    expected_api_call: dict,
    cfg: dict | None = None,
    token: str | None = None,
    run_executable: bool = False,
) -> dict:
    """Score a single prediction against the expected api_call."""
    fmt = format_score(predicted_api_call)

    if fmt == 0.0 or not isinstance(predicted_api_call, dict):
        f1 = 0.0
        exe = False if run_executable else None
    else:
        f1 = param_f1(predicted_api_call, expected_api_call)
        if run_executable and cfg and token:
            exe = executable_score(predicted_api_call, cfg, token)
        else:
            exe = None

    composite = (fmt + f1) / 2
    if exe is not None:
        composite = (fmt + f1 + (1.0 if exe else 0.0)) / 3

    band = _band(composite)

    return {
        "format_score": fmt,
        "param_f1": round(f1, 4),
        "executable": exe,
        "composite_score": round(composite, 4),
        "band": band,
    }


def _band(score: float) -> str:
    if score >= 0.9:
        return "GOLD"
    if score >= 0.7:
        return "SILVER"
    if score >= 0.4:
        return "BRONZE"
    return "FAIL"


# ── Main ───────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="JSONL file of model predictions (same format as training.jsonl)")
    parser.add_argument("--holdout", required=True, type=Path,
                        help="JSONL holdout file with ground-truth api_call fields")
    parser.add_argument("--executable", action="store_true",
                        help="Also run Tier 3 live API validation (requires --config)")
    parser.add_argument("--config", type=Path,
                        help="config.yaml for Tier 3 validation")
    parser.add_argument("--output", type=Path,
                        help="Write scored records to this JSONL file")
    args = parser.parse_args()

    cfg = None
    token = None
    if args.executable:
        if not args.config:
            sys.exit("--executable requires --config")
        try:
            import yaml
        except ImportError:
            sys.exit(PYYAML_REQUIRED)
        cfg = yaml.safe_load(args.config.read_text())
        token = get_token(cfg)

    preds = load_records(args.predictions)
    holdout = load_records(args.holdout)

    if len(preds) != len(holdout):
        print(f"Warning: predictions ({len(preds)}) and holdout ({len(holdout)}) have different lengths.")
        print("Scoring up to min(len).")

    results = []
    for pred, gold in zip(preds, holdout):
        scored = score_record(
            pred.get("api_call"),
            gold.get("api_call", {}),
            cfg=cfg,
            token=token,
            run_executable=args.executable,
        )
        scored["question"] = pred.get("question", "")
        scored["conventions_tested"] = gold.get("conventions_tested", [])
        results.append(scored)

    # Summary
    n = len(results)
    avg_fmt = sum(r["format_score"] for r in results) / n
    avg_f1 = sum(r["param_f1"] for r in results) / n
    avg_comp = sum(r["composite_score"] for r in results) / n
    bands = {}
    for r in results:
        bands[r["band"]] = bands.get(r["band"], 0) + 1

    print(f"\n{'─'*50}")
    print(f"Records evaluated:  {n}")
    print(f"Avg format score:   {avg_fmt:.3f}")
    print(f"Avg param F1:       {avg_f1:.3f}")
    print(f"Avg composite:      {avg_comp:.3f}")
    print(f"Band distribution:  {bands}")
    print(f"{'─'*50}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Scored records written to: {args.output}")


if __name__ == "__main__":
    main()
