#!/usr/bin/env python3
"""Generate and/or score code-path predictions with apisynth's code rubric."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import score_code_record  # noqa: E402


JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_output(text: str) -> object:
    match = JSON_BLOCK_RE.search(text or "")
    payload = match.group(1) if match else text
    try:
        return json.loads(payload)
    except Exception:
        # Fall back to the first plausible JSON object in a chatty response.
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(payload[start:end + 1])
            except Exception:
                pass
    return None


def generate_predictions(
    holdout_messages: list[dict],
    model: str,
    vllm_url: str,
    max_tokens: int,
    retrieval_index: dict | None = None,
) -> list[dict]:
    from openai import OpenAI

    if retrieval_index is not None:
        from code_retrieval import enrich_messages

    client = OpenAI(base_url=f"{vllm_url.rstrip('/')}/v1", api_key="none")
    predictions: list[dict] = []
    for i, record in enumerate(holdout_messages):
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in record["messages"]
            if m["role"] != "assistant"
        ]
        if retrieval_index is not None:
            messages = enrich_messages(messages, retrieval_index)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            seed=42,
        )
        generated = resp.choices[0].message.content or ""
        predictions.append({
            "question": record.get("label") or messages[-1]["content"],
            "generated": generated,
            "output": parse_output(generated),
        })
        if (i + 1) % 25 == 0 or i + 1 == len(holdout_messages):
            print(f"generated {i + 1}/{len(holdout_messages)}", flush=True)
    return predictions


def score_predictions(predictions: list[dict], holdout_gold: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for i, (pred, gold) in enumerate(zip(predictions, holdout_gold)):
        expected = gold.get("output", {})
        output = pred.get("output")
        score = score_code_record(output, expected, check_signature=True)
        scored.append({
            "index": i,
            "question": gold.get("question", pred.get("question", "")),
            "expected": expected,
            "predicted": output,
            "generated": pred.get("generated", ""),
            **score,
        })
    return scored


def print_summary(scored: list[dict]) -> None:
    n = len(scored)
    if n == 0:
        print("No records scored.")
        return
    avg_format = sum(r["format_score"] for r in scored) / n
    avg_field = sum(r["field_accuracy"] for r in scored) / n
    sig_valid = sum(1 for r in scored if r.get("signature_valid") is True) / n
    composite = sum(r["composite_score"] for r in scored) / n
    bands: dict[str, int] = {}
    for r in scored:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    print("\nCode holdout evaluation")
    print(f"records={n}")
    print(f"format_score={avg_format:.4f}")
    print(f"field_accuracy={avg_field:.4f}")
    print(f"signature_valid={sig_valid:.4f}")
    print(f"composite_score={composite:.4f}")
    print(f"bands={bands}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path,
                        help="Raw apisynth code holdout JSONL.")
    parser.add_argument("--holdout-messages", type=Path,
                        help="Prepared holdout messages JSONL, required for generation.")
    parser.add_argument("--predictions", type=Path,
                        help="Existing or output prediction JSONL.")
    parser.add_argument("--scored-out", type=Path)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retrieval-index", type=Path, nargs="+", default=None,
                        help="JSONL file(s) to build retrieval index from (enables RAG).")
    parser.add_argument("--use-gold", action="store_true",
                        help="Score holdout outputs against themselves; rubric self-check.")
    args = parser.parse_args()

    holdout = load_jsonl(args.holdout.expanduser())

    retrieval_index = None
    if args.retrieval_index:
        from code_retrieval import build_index_from_jsonl
        index_paths = [p.expanduser() for p in args.retrieval_index]
        retrieval_index = build_index_from_jsonl(*index_paths)
        print(f"retrieval index: {sum(len(v) for v in retrieval_index.values())} name→file entries "
              f"from {len(index_paths)} file(s)")

    if args.use_gold:
        predictions = [
            {
                "question": record.get("question", ""),
                "generated": "",
                "output": record.get("output"),
            }
            for record in holdout
        ]
        if args.predictions:
            write_jsonl(args.predictions.expanduser(), predictions)
    elif args.vllm_url:
        if not args.model:
            raise SystemExit("--model is required with --vllm-url")
        if not args.holdout_messages:
            raise SystemExit("--holdout-messages is required with --vllm-url")
        predictions = generate_predictions(
            load_jsonl(args.holdout_messages.expanduser()),
            model=args.model,
            vllm_url=args.vllm_url,
            max_tokens=args.max_tokens,
            retrieval_index=retrieval_index,
        )
        if args.predictions:
            write_jsonl(args.predictions.expanduser(), predictions)
    else:
        if not args.predictions:
            raise SystemExit("Provide --predictions, or --vllm-url plus --model.")
        predictions = load_jsonl(args.predictions.expanduser())

    if len(predictions) != len(holdout):
        print(
            f"Warning: predictions={len(predictions)} holdout={len(holdout)}; "
            "scoring min length.",
            file=sys.stderr,
        )

    scored = score_predictions(predictions, holdout)
    print_summary(scored)

    if args.scored_out:
        write_jsonl(args.scored_out.expanduser(), scored)
        print(f"scored_out={args.scored_out.expanduser()}")


if __name__ == "__main__":
    main()
