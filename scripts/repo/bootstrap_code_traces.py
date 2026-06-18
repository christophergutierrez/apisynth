#!/usr/bin/env python3
"""
STaR-style self-bootstrap for CODE-path training traces.

Mirrors scripts/bootstrap_traces.py (the API-path analog) but targets the
code-unit output format: {unit, name, file, signature[, class]}.  Verification
is done offline with the Phase-3 deterministic validators from scripts/eval.py
— no live execution sandbox.

The model's own correct reasoning traces become training data, improving itself
with each production cycle.

Usage:
    # Run against a code training.jsonl (strict gold-map verification):
    python scripts/repo/bootstrap_code_traces.py \\
        --model <adapter-name> \\
        --vllm-url http://<host>:8000 \\
        --input data/repos/<repo>/training.jsonl \\
        --output data/repos/<repo>/bootstrapped/training.jsonl

    # Run against a free-form prompt file (format+signature verification only):
    python scripts/repo/bootstrap_code_traces.py \\
        --model <adapter-name> \\
        --vllm-url http://<host>:8000 \\
        --prompts prompts.txt \\
        --output data/repos/<repo>/bootstrapped/training.jsonl

    # Dry-run: print results without writing:
    python scripts/repo/bootstrap_code_traces.py --model <adapter-name> --dry-run

Requirements:
    - vLLM server running with the trained model
    - auth token set via env_var or cli_fallback as needed
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make scripts/ importable so we can reach eval.py from scripts/repo/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval import code_field_accuracy, code_format_score, code_signature_valid  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — on-disk key names for the CODE record shape
# ---------------------------------------------------------------------------

FIELD_TYPE = "type"
FIELD_QUESTION = "question"
FIELD_THINKING = "thinking"
FIELD_OUTPUT = "output"
FIELD_SOURCE = "source"

CODE_TYPE = "code"
SOURCE_BOOTSTRAP = "bootstrap"

# ---------------------------------------------------------------------------
# System prompt — instructs the model to think then return a code-unit JSON
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a code-documentation assistant. Given a natural language question about "
    "a code unit (function, method, class, or API call), respond with a JSON object "
    "describing that code unit inside a ```json code block. Think through the question "
    "before answering. "
    "Return exactly: "
    '{"unit": "<function|method|class|api_call>", '
    '"name": "<identifier>", '
    '"file": "<path/to/file.py>", '
    '"signature": "<signature_string>"} '
    'and optionally "class": "<ClassName>" when the unit is a method.'
)

# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def call_model(
    vllm_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.3,
) -> dict:
    """POST to a vLLM/OpenAI-compatible /v1/chat/completions endpoint.

    Returns a dict with keys: content (str), ms (float), comp_tokens (int).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
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
    return {"content": content, "ms": ms, "comp_tokens": usage.get("completion_tokens", 0)}


# ---------------------------------------------------------------------------
# Extract / parse
# ---------------------------------------------------------------------------


def extract(content: str) -> tuple[str, str]:
    """Return (thinking, answer) from model output.

    Strips the <think>…</think> block (if present) and returns the remainder
    as the answer — mirroring the API-path analog.
    """
    think = ""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        think = m.group(1).strip()
        content = content[m.end():].strip()
    return think, content


def parse_code_output(answer: str) -> dict | None:
    """Strip ```json fences and parse as JSON.  Returns None on failure."""
    cleaned = re.sub(r"```json\s*", "", answer).replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            return None
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_code_output(predicted: object, gold: dict | None) -> bool:
    """Verify a predicted code-unit dict against optional gold.

    Strict path (gold is a dict):
        PASS iff code_format_score == 1.0
              AND code_signature_valid is True
              AND code_field_accuracy["field_accuracy"] == 1.0

    Fallback path (gold is None — free-form prompt):
        PASS iff code_format_score == 1.0
              AND code_signature_valid is True
    """
    if code_format_score(predicted) != 1.0:
        return False
    if code_signature_valid(predicted) is not True:
        return False
    if gold is not None:
        if code_field_accuracy(predicted, gold)["field_accuracy"] != 1.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Gold-map builder
# ---------------------------------------------------------------------------


def build_gold_map(input_path: str) -> dict[str, dict]:
    """Read a code training.jsonl and build a {question -> output} map.

    First-seen order wins when multiple records share the same question.
    Only records with type=="code" and a dict output are included.
    """
    gold_map: dict[str, dict] = {}
    p = Path(input_path).expanduser()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get(FIELD_TYPE) != CODE_TYPE:
            continue
        question = record.get(FIELD_QUESTION, "")
        output = record.get(FIELD_OUTPUT)
        if question and isinstance(output, dict) and question not in gold_map:
            gold_map[question] = output
    return gold_map


# ---------------------------------------------------------------------------
# Deduplication helpers (mirrored verbatim from the API-path analog)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _dedup_by_embedding(
    candidates: list[dict],
    output_path: str | None,
    threshold: float,
) -> tuple[list[dict], int]:
    """Remove candidates whose question is near-duplicate of existing training records.

    Embeds all questions using sentence-transformers (all-MiniLM-L6-v2).
    Compares each candidate against existing output records + already-accepted
    candidates.  Returns (kept_candidates, num_removed).

    If sentence-transformers is not installed, skips dedup with a warning.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Warning: sentence-transformers not installed — skipping dedup. "
            "Run: pip install sentence-transformers"
        )
        return candidates, 0

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load existing questions from the output file
    existing_questions: list[str] = []
    if output_path:
        p = Path(output_path).expanduser()
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        existing_questions.append(json.loads(line).get(FIELD_QUESTION, ""))
                    except json.JSONDecodeError:
                        pass

    reference_embeddings = (
        model.encode(existing_questions).tolist() if existing_questions else []
    )

    kept: list[dict] = []
    removed = 0
    for r in candidates:
        q = r.get(FIELD_QUESTION, "")
        emb = model.encode([q]).tolist()[0]
        is_dup = any(
            _cosine_similarity(emb, ref) >= threshold
            for ref in reference_embeddings
        )
        if is_dup:
            removed += 1
        else:
            kept.append(r)
            reference_embeddings.append(emb)

    return kept, removed


# ---------------------------------------------------------------------------
# Bootstrap worker
# ---------------------------------------------------------------------------


def bootstrap_one(
    vllm_url: str,
    model: str,
    prompt: str,
    gold: dict | None,
    temperature: float = 0.3,
) -> dict:
    """Run inference on a single prompt and verify the result.

    Returns a result dict; 'ok' key indicates success.
    """
    try:
        result = call_model(vllm_url, model, prompt, temperature=temperature)
    except Exception as e:
        return {FIELD_QUESTION: prompt, "ok": False, "error": str(e)}

    thinking, answer = extract(result["content"])
    predicted = parse_code_output(answer)
    if predicted is None:
        return {
            FIELD_QUESTION: prompt,
            "ok": False,
            "error": "unparseable",
            "ms": result["ms"],
        }

    ok = verify_code_output(predicted, gold)

    return {
        FIELD_QUESTION: prompt,
        "ok": ok,
        FIELD_OUTPUT: predicted,
        FIELD_THINKING: thinking,
        "ms": result["ms"],
        "comp_tokens": result["comp_tokens"],
    }


# ---------------------------------------------------------------------------
# Main bootstrap loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="code-thinking")
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
        help="vLLM server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Code training.jsonl — each record's question becomes a prompt and "
             "its output is the gold for strict verification",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="File with one prompt per line — free-form, no gold (format+signature check only)",
    )
    parser.add_argument("--output", default=None, help="Output training.jsonl path")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3). Use >0 to prevent trace collapse.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for deduplication against existing "
             "training records (default: 0.95). Set to 1.0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Build gold map and prompts list
    gold_map: dict[str, dict] = {}
    prompts: list[str] = []

    if args.input:
        gold_map = build_gold_map(args.input)
        prompts = list(gold_map.keys())
        print(f"Input:       {args.input} ({len(prompts)} prompts with gold)")

    if args.prompts:
        extra = [
            line.strip()
            for line in Path(args.prompts).read_text().splitlines()
            if line.strip()
        ]
        # Preserve first-seen order; prompts from --input take priority
        seen = set(prompts)
        for p in extra:
            if p not in seen:
                prompts.append(p)
                seen.add(p)
        print(f"Prompts:     {args.prompts} ({len(extra)} lines)")

    if not prompts:
        print("No prompts — supply --input or --prompts. Exiting.")
        return

    print(f"Model:       {args.model}")
    print(f"Server:      {args.vllm_url}")
    print(f"Prompts:     {len(prompts)}")
    print(f"Temperature: {args.temperature}")
    print(f"Dedup >=:    {args.dedup_threshold}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                bootstrap_one,
                args.vllm_url,
                args.model,
                p,
                gold_map.get(p),
                args.temperature,
            ): p
            for p in prompts
        }
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                p = futures[fut]
                print(f"  E  {p:<40s}  exception: {exc}")
                results.append({FIELD_QUESTION: p, "ok": False, "error": str(exc)})
                continue
            sym = "V" if r["ok"] else "X"
            ms = r.get("ms", 0)
            print(f"  {sym}  {r[FIELD_QUESTION]:<40s}  {ms:.0f}ms")
            results.append(r)

    passed = [r for r in results if r["ok"] and FIELD_OUTPUT in r]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(passed)}/{len(prompts)} passed")
    if failed:
        print(f"{len(failed)} failed:")
        for r in failed:
            print(f"  {r[FIELD_QUESTION]!r}: {r.get('error', '')}")

    # Dedup against existing training data + already-accepted bootstrap records
    dedup_candidates = passed
    deduped_count = 0
    if args.dedup_threshold < 1.0 and passed:
        dedup_candidates, deduped_count = _dedup_by_embedding(
            passed, args.output, args.dedup_threshold
        )
        print(
            f"Dedup: {deduped_count} removed (similarity >= {args.dedup_threshold}), "
            f"{len(dedup_candidates)} kept"
        )

    if args.dry_run or not args.output:
        print("\nDry run — no output written.")
        return

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "a", encoding="utf-8") as f:
        for r in dedup_candidates:
            record = {
                FIELD_TYPE: CODE_TYPE,
                FIELD_QUESTION: r[FIELD_QUESTION],
                FIELD_THINKING: r[FIELD_THINKING],
                FIELD_OUTPUT: r[FIELD_OUTPUT],
                FIELD_SOURCE: SOURCE_BOOTSTRAP,
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    print(f"\nWrote {written} records -> {out}")


if __name__ == "__main__":
    main()
