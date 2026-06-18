#!/usr/bin/env python3
"""3-tier evaluation rubric for structured API call outputs.

Scores predictions against a holdout set on three independent tiers:

  Tier 1 — format:      is api_call valid JSON with the required structure?
  Tier 2 — param_f1:   F1 over parameter names (precision + recall)
  Tier 3 — executable: optional live API validation (requires --executable flag)

Phase-3 code records are also supported (--mode code / --mode auto).

Usage:
    python scripts/eval.py \\
        --predictions data/<vendor>/<endpoint>/holdout.jsonl \\
        --holdout     data/<vendor>/<endpoint>/holdout.jsonl \\
        [--executable --config apis/<vendor>/<endpoint>/config.yaml]
    python scripts/eval.py \\
        --predictions data/repos/<repo>/holdout.jsonl \\
        --holdout     data/repos/<repo>/holdout.jsonl \\
        --mode code [--check-signature]

When --predictions and --holdout are the same file the score is a self-
consistency check (should be 1.0 for format and param_f1 / field_accuracy).
"""

import argparse
import ast
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


# ── Code-unit evaluation (Phase 3, additive / opt-in) ─────────────────────
#
# Mirrors the 3-tier rubric for API calls, adapted for code-unit output dicts.
# The wrap-as-def contract for signature validation mirrors the form pinned in
# generate_from_code.py milestone 3.2 (_signature_well_formed).

_CODE_UNIT_TYPES = frozenset({"function", "method", "class", "api_call"})
_CODE_REQUIRED_KEYS = frozenset({"unit", "name", "file", "signature"})


def code_format_score(output: object) -> float:
    """Return 1.0 iff output is a well-formed code-unit dict, 0.0 otherwise.

    Requires output to be a dict with keys 'unit', 'name', 'file', 'signature'
    where 'unit' is one of {function, method, class, api_call} and name/file/
    signature are non-empty strings.  If the optional 'class' key is present it
    must be a string.
    """
    if not isinstance(output, dict):
        return 0.0
    if not _CODE_REQUIRED_KEYS.issubset(output.keys()):
        return 0.0
    if output["unit"] not in _CODE_UNIT_TYPES:
        return 0.0
    for key in ("name", "file", "signature"):
        if not isinstance(output[key], str) or not output[key]:
            return 0.0
    if "class" in output and not isinstance(output["class"], str):
        return 0.0
    return 1.0


def code_field_accuracy(predicted: object, expected: dict) -> dict:
    """Compute exact-match field accuracy between predicted and expected outputs.

    Always compares: unit, name, file, signature.
    Also compares 'class' when expected has a truthy 'class' key.

    Returns a dict with per-field boolean matches (e.g. 'unit_match'), a
    computed 'field_accuracy' = (matching fields) / (compared fields) rounded
    to 4 decimal places, and the per-field boolean values.
    """
    always_fields = ("unit", "name", "file", "signature")

    if not isinstance(predicted, dict):
        result: dict = {f"{k}_match": False for k in always_fields}
        if expected.get("class"):
            result["class_match"] = False
        result["field_accuracy"] = 0.0
        return result

    fields = list(always_fields)
    if expected.get("class"):
        fields.append("class")

    matches = {f"{k}_match": (predicted.get(k) == expected.get(k)) for k in fields}
    accuracy = sum(matches.values()) / len(fields)
    return {**matches, "field_accuracy": round(accuracy, 4)}


def code_signature_valid(output: object) -> bool | None:
    """Return True if the predicted output's signature is well-formed, else False.

    Validation form: wrap as ``def {sig}: pass`` and attempt ``ast.parse``.
    This mirrors the contract in generate_from_code._signature_well_formed
    (milestone 3.2).  Returns None when output is not a dict or has no
    non-empty 'signature' key (absence is not a malformation).
    """
    if not isinstance(output, dict):
        return None
    sig = output.get("signature")
    if not isinstance(sig, str) or not sig:
        return None
    try:
        ast.parse(f"def {sig}: pass")
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def score_code_record(
    predicted_output: object,
    expected_output: dict,
    check_signature: bool = False,
) -> dict:
    """Score a single code-unit prediction against the expected output.

    Tiers:
      1. code_format_score  — structural validity (always run)
      2. code_field_accuracy — exact-match per field (skipped on format fail)
      3. code_signature_valid — AST well-formedness (only when check_signature=True)

    Composite = mean of available tiers; banded via _band (reused as-is).
    """
    fmt = code_format_score(predicted_output)

    always_fields = ("unit", "name", "file", "signature")
    if fmt == 0.0 or not isinstance(predicted_output, dict):
        field_booleans = {f"{k}_match": False for k in always_fields}
        if isinstance(expected_output, dict) and expected_output.get("class"):
            field_booleans["class_match"] = False
        field_acc = 0.0
        sig_valid: bool | None = False if check_signature else None
    else:
        fa_dict = code_field_accuracy(predicted_output, expected_output)
        field_acc = fa_dict["field_accuracy"]
        field_booleans = {k: v for k, v in fa_dict.items() if k != "field_accuracy"}
        sig_valid = code_signature_valid(predicted_output) if check_signature else None

    composite = (fmt + field_acc) / 2
    if sig_valid is not None:
        composite = (fmt + field_acc + (1.0 if sig_valid else 0.0)) / 3

    band = _band(round(composite, 4))

    return {
        "format_score": fmt,
        "field_accuracy": round(field_acc, 4),
        **field_booleans,
        "signature_valid": sig_valid,
        "composite_score": round(composite, 4),
        "band": band,
    }


def _is_code_record(record: dict) -> bool:
    """Return True if record should be evaluated as a code record.

    Criteria: record has type == "code", OR has an "output" dict without
    an "api_call" key.
    """
    if record.get("type") == "code":
        return True
    if isinstance(record.get("output"), dict) and "api_call" not in record:
        return True
    return False


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
                        help="Also run Tier 3 live API validation (requires --config). API path only.")
    parser.add_argument("--config", type=Path,
                        help="config.yaml for Tier 3 validation (API path only)")
    parser.add_argument("--output", type=Path,
                        help="Write scored records to this JSONL file")
    parser.add_argument("--mode", choices=["auto", "api", "code"], default="auto",
                        help="Scoring mode: 'api' for Phase-2 api_call records, 'code' for Phase-3 "
                             "code-unit records, 'auto' to dispatch per record (default: auto)")
    parser.add_argument("--check-signature", action="store_true",
                        help="Code path: run Tier-3 AST signature well-formedness check (offline)")
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

    api_results = []
    code_results = []

    for pred, gold in zip(preds, holdout):
        # Determine which path to use for this record.
        if args.mode == "api":
            use_code = False
        elif args.mode == "code":
            use_code = True
        else:  # auto
            use_code = _is_code_record(gold)

        if use_code:
            scored = score_code_record(
                pred.get("output"),
                gold.get("output", {}),
                check_signature=args.check_signature,
            )
            scored["question"] = pred.get("question", "")
            code_results.append(scored)
        else:
            scored = score_record(
                pred.get("api_call"),
                gold.get("api_call", {}),
                cfg=cfg,
                token=token,
                run_executable=args.executable,
            )
            scored["question"] = pred.get("question", "")
            scored["conventions_tested"] = gold.get("conventions_tested", [])
            api_results.append(scored)

    results = api_results + code_results

    if not results:
        print("No records to evaluate.")
        return

    print(f"\n{'─'*50}")

    # API summary
    if api_results:
        n = len(api_results)
        avg_fmt = sum(r["format_score"] for r in api_results) / n
        avg_f1 = sum(r["param_f1"] for r in api_results) / n
        avg_comp = sum(r["composite_score"] for r in api_results) / n
        bands: dict = {}
        for r in api_results:
            bands[r["band"]] = bands.get(r["band"], 0) + 1
        print(f"API records evaluated: {n}")
        print(f"Avg format score:      {avg_fmt:.3f}")
        print(f"Avg param F1:          {avg_f1:.3f}")
        print(f"Avg composite:         {avg_comp:.3f}")
        print(f"Band distribution:     {bands}")

    # Code summary
    if code_results:
        if api_results:
            print()
        n = len(code_results)
        avg_fmt = sum(r["format_score"] for r in code_results) / n
        avg_fa = sum(r["field_accuracy"] for r in code_results) / n
        avg_comp = sum(r["composite_score"] for r in code_results) / n
        bands = {}
        for r in code_results:
            bands[r["band"]] = bands.get(r["band"], 0) + 1
        print(f"Code records evaluated: {n}")
        print(f"Avg format score:       {avg_fmt:.3f}")
        print(f"Avg field accuracy:     {avg_fa:.3f}")
        print(f"Avg composite:          {avg_comp:.3f}")
        print(f"Band distribution:      {bands}")

    print(f"{'─'*50}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Scored records written to: {args.output}")


if __name__ == "__main__":
    main()
