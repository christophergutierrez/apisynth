# Architecture

How apisynth works and why it is designed this way.

---

## Overview

apisynth is a data generation pipeline. It takes a REST API and produces training data
that teaches a language model to translate natural language into API calls.

```
config.yaml             ← describes the endpoint (URL, params, auth, targets)
    │
    ▼
pipeline.py             ← orchestrates all steps below (or run individually)
    │
    ├── sweep.py        ← discovers which param values return real data
    │
    ├── run.py          ← generates (question, api_call) pairs, validates each via live API
    │                      each record now includes schema + intent_category fields
    │
    ├── add_thinking.py ← enriches each record with a structured reasoning trace
    │
    ├── enrich_schema.py← adds schema + intent_category to pre-existing records (no API)
    │
    ├── evolve_questions.py ← diversifies question phrasings via LLM mutation (Claude API)
    │
    └── gen_router_data + train_router ← trains intent classifier
    │
    ▼
training.jsonl          ← final supervised fine-tuning dataset
    │
    ▼
(external: trainLLM)    ← fine-tunes the LLM on the dataset
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

## The code/repo path — validators as the verifier

apisynth has a second data-generation path that targets **source-code repositories** instead of a
live REST API. Its design mirrors the API path component-for-component, with one substitution at the
core: where the API path uses **the live API as the verifier**, the code path uses **deterministic
structural validators** (`eval.py`'s code rubric). The full guide is
[REPO_SOURCES.md](REPO_SOURCES.md); the architecture rationale:

```
repo.yaml               ← describes the repository (path/url, include globs, extraction units)
    │
    ▼
repo_pipeline.py        ← orchestrates scan → generate (the pipeline.py analog)
    │
    ├── scan_repo.py        ← ast-extracts code units {function, method, class, api_call}
    │
    └── generate_from_code.py ← one record per unit: question + deterministic thinking + output,
                                then a SHA-256 deterministic train/holdout split
    │
    ▼
training.jsonl ({type:"code", ...})   ← optionally augmented, all verified offline:
    ├── gen_code_dpo.py          ← preference pairs; validators decide chosen vs rejected
    ├── evolve_code_questions.py ← deterministic template paraphrase (NO LLM, unlike the API path)
    ├── gen_code_router_data.py + train_code_router.py ← route by target file path
    └── bootstrap_code_traces.py ← STaR: model proposes, validators verify (no exec sandbox)
```

**Why validators instead of a live API.** The "answer" for a code question is a structural fact —
which function/method/class a question is about — that can be checked exactly against the scanned
ground truth. So the same role the live API plays for the API path (rejecting hallucinated
parameter combinations) is played here by three offline checks: format validity
(`code_format_score`), AST well-formedness of the signature (`code_signature_valid`), and
exact field match against the scanned unit (`code_field_accuracy`). This keeps the entire path
**offline and key-free** — the only network call is STaR bootstrap's model inference, and even those
outputs are verified offline. It also keeps everything **deterministic**: SHA-256 of `<file>:<name>`
for hashing, fixed `SEED = 42` for shuffles, and no unseeded randomness anywhere.

**Where it diverges from the API path.** Two stages are deliberately different:
- `evolve_code_questions.py` is **fully deterministic template-based** paraphrase — it does *not*
  call an LLM, whereas the API path's `evolve_questions.py` calls Claude.
- `bootstrap_code_traces.py` has **no live execution sandbox**; the Phase-3 validators are the
  execution-feedback stand-in (strict field-accuracy match against gold, or format+signature for
  free-form prompts).

The backward-trace principle (generate the trace from the known-correct answer) carries over
unchanged: traces are built from each unit's `output`, so reasoning is always consistent with the
correct code unit.

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
