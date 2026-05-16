# Architecture

How apisynth works and why it is designed this way.

---

## Overview

apisynth is a data generation pipeline. It takes a REST API and produces training data
that teaches a language model to translate natural language into API calls.

```
config.yaml          ← describes the endpoint (URL, params, auth, targets)
    │
    ▼
sweep.py             ← discovers which param values return real data
    │
    ▼
run.py               ← generates (question, api_call) pairs, validates each via live API
    │
    ▼
add_thinking.py      ← enriches each record with a structured reasoning trace
    │
    ▼
training.jsonl       ← final supervised fine-tuning dataset
    │
    ▼
(external: trainLLM) ← fine-tunes the LLM on the dataset
    │
    ▼
bootstrap_traces.py  ← runs trained model, verifies outputs, feeds successful traces back
```

The router (gen_router_data.py + train_router.py) is a separate, lighter-weight
classifier used for intent routing before the LLM generates the API call. It is
optional when the LLM handles routing internally (e.g. with thinking traces).

---

## Why each component exists

### sweep.py — live param discovery

Config files describe params in terms of sweep ranges (e.g., integer IDs 1–10000).
The sweep confirms which values actually exist in the API and saves them in `config.yaml`
under `status.valid_values`. This matters because:

- Training records with invalid IDs would be unreachable at inference time
- The model learns from real IDs and real combinations, not random integers
- Sweep results are cached — rerun only when the dataset changes

Chained endpoints (e.g., get-episode requires knowing a valid episodeId) collect parent
IDs first by calling the list endpoint, then sweep the child endpoint.

### run.py — API-validated training records

Every record written by `run.py` has been confirmed by a live API call. This means:

- No hallucinated param combinations
- Realistic distribution of param values
- No training records for endpoints the account can't access

The question generation in `gen_questions()` uses phrasing templates to produce
varied natural language for the same underlying API call. The same API call might appear
as "Get program 42", "fetch program 42", "program 42 details", etc.

### add_thinking.py — deterministic reasoning traces

Rather than using a teacher LLM to generate reasoning (which can hallucinate), traces
are generated deterministically from the known-correct `api_call`. The process is:

1. Infer entity type and operation (list vs by-ID) from the endpoint pattern
2. Detect synonym vocabulary in the question ("campaigns" → measurements)
3. Look up domain-specific rejection notes (e.g., "this API has no /campaigns endpoint — use /measurements instead")
4. Emit a structured trace: Entity → Scope → Use (correct endpoint) → NOT (wrong endpoint)

This guarantees the trace is always consistent with the correct answer and includes
explicit rejection of the model's likely pretrained priors for wrong endpoints.

Manual overrides (`_MANUAL` dict) provide hand-crafted traces for cases the templates
handle poorly — typically complex synonym chains or disambiguation between two real endpoints.

### bootstrap_traces.py — self-improvement (STaR loop)

After the model is trained, it generates its own reasoning traces. The STaR (Self-Taught
Reasoner) loop:

1. Run model against prompts (canonical test cases + organic production logs)
2. Verify each generated API call against the live API (`--verify` flag)
3. Write successful (prompt → thinking + api_call) pairs back to training data
4. Retrain — the model learns from its own correct reasoning

As production logs accumulate, this loop makes the model self-correcting without
manual data generation. The organic logs contain real user phrasings that synthetic
data cannot anticipate.

### The router classifier

`train_router.py` trains a logistic-regression classifier on top of `all-MiniLM-L6-v2`
sentence embeddings. It is used when a separate routing step precedes the LLM generation
call. Design choices:

- **Logistic regression over neural classifiers**: fast, interpretable, low memory, no GPU
- **Sentence embeddings over bag-of-words**: handles paraphrase and synonym variation
- **Separate from LLM**: allows routing to be swapped independently; also cheaper to run

When the LLM uses thinking traces to do its own routing (the thinking-model approach),
the external router is optional.

---

## Data flow in detail

```
config.yaml status section (auto-populated by sweep.py):
  status:
    programId:
      valid_values: [1, 2, 5, 7, ...]    ← real IDs confirmed by live API
      swept_through: 500
    variants:
      - params: {pageSize: 10}
        confirmed: true
      - params: {pageSize: 10, networkId: 3}
        confirmed: true

training.jsonl (written by run.py, enriched by add_thinking.py):
  {"question": "Get program 42",
   "thinking": "Entity: program\nScope: single item...\nUse: GET /external/v1/content/programs/{programId}",
   "api_call": {"endpoint": "GET /external/v1/content/programs/{programId}", "params": {"programId": 42}}}

router_train.jsonl (written by gen_router_data.py):
  {"question": "list all programs", "route_key": "acme/api/programs"}
  {"question": "Get program 42",    "route_key": "acme/api/program"}
```

---

## Why thinking traces are generated backwards

Standard CoT pipeline (forward):
```
question → LLM generates trace → check if trace leads to correct answer → keep if correct
```

apisynth pipeline (backward):
```
question + correct answer (known from live API validation) → generate trace that leads to it
```

The backward approach guarantees the trace is consistent with the correct answer.
Forward-generated traces can be incorrect even when they reach the right endpoint by
coincidence — or the trace can be correct but the model ignores it (the "thinks right,
outputs wrong" failure mode).

---

## Directory structure rationale

```
apis/<vendor>/<endpoint>/    one directory per endpoint
  config.yaml               endpoint description + sweep results (auto-updated)
  config.example.yaml       template for new endpoints (not auto-updated)

data/<vendor>/<endpoint>/   one directory per endpoint (gitignored)
  training.jsonl            generated + validated training records
  holdout.jsonl             held-out evaluation records
  dpo.jsonl                 DPO preference pairs (if applicable)

data/<vendor>/router/       router training data (gitignored)
  router_train.jsonl
  router_test.jsonl
  router_classifier.joblib  trained classifier artifact

scripts/                    generic scripts (no vendor-specific logic)
apis/<vendor>/              vendor-specific configs and wrappers
```

Config files are committed; training data is not (data/ is gitignored). This keeps
the repo lightweight while preserving the configuration that produces the data.
