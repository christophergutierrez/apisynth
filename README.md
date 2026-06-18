# apisynth

Generate validated training data mapping natural-language questions to structured API calls.

Each record pairs a question with the API call that satisfies it:
```json
{
  "question": "Get 10 episodes from network 5",
  "thinking": "Entity: episodes\nScope: list\nRequested count: 10\nFilters: networkId=5\nEndpoint: GET /external/v1/content/episodes\nParams: {\"pageSize\": 10, \"networkId\": 5}",
  "api_call": {"endpoint": "GET /external/v1/content/episodes", "params": {"pageSize": 10, "networkId": 5}}
}
```

The training data teaches an LLM *intent → API parameters*, not API responses. The optional `thinking` field contains a structured reasoning trace added by `add_thinking.py`.

## What it does and why

apisynth is an **agentic tool-use data pipeline**. The goal is to produce fine-tuning data that teaches a model to call APIs from natural language — the same problem addressed by Gorilla LLM and ToolBench, but designed to run against any REST API you have access to.

The unifying design principle: **use the live API as the verifier at every stage**. Sweep, run, eval, DPO, and STaR all use real API responses as ground truth rather than human labels or a teacher model. If the API rejects a parameter combination, it never appears in the dataset.

The pipeline implements several established methods:

| Method | Where | What it does |
|--------|-------|--------------|
| **SFT with live verification** | `run.py` | Generates (question, api_call) pairs; every record is validated by a real API call before being written |
| **Chain-of-thought traces** | `add_thinking.py` | Adds structured reasoning traces, but generated *deterministically from the ground-truth answer* — not distilled from a teacher model, so the reasoning is always consistent with the correct call |
| **STaR** (Zelikman et al. 2022) | `bootstrap_traces.py` | Runs the trained model, keeps outputs the API accepts, feeds them back as new training data — iterative self-improvement using execution as the reward signal |
| **DPO** (Rafailov et al. 2023) | `gen_dpo.py` | Generates (chosen, rejected) preference pairs using the live API validator as the judge |
| **Paraphrase augmentation** | `evolve_questions.py` | Uses Claude to rewrite questions along multiple axes (formality, verbosity, synonyms) — same API call, broader surface-form coverage |
| **Intent routing** | `train_router.py` | Trains a lightweight logistic-regression classifier to route to the right endpoint before the heavier LLM runs |

### Two sources: APIs and code repositories

apisynth has a second, parallel path that generates the same kinds of training data from a
**source-code repository** instead of a live API. It teaches a model *natural language → code unit*
(which function/method/class/API-call a question targets). The design mirrors the API path
component-for-component, but swaps the verifier: where the API path uses the **live API** as ground
truth, the code path uses **deterministic structural validators** (`scripts/eval.py`). This makes
the whole code path offline and key-free (the one exception is STaR bootstrap, which calls a model
server but still verifies offline). The scripts live in `scripts/repo/` and `scripts/repo_pipeline.py`.

```bash
python scripts/repo_pipeline.py --repo-dir repos/example   # scan a repo → training + holdout
```

See [docs/REPO_SOURCES.md](docs/REPO_SOURCES.md) for the full code-path guide.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pipeline works, why each component exists, data flow |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | Step-by-step walkthrough for adding a new vendor |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors and how to fix them |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | Schema for training.jsonl, holdout, router, and DPO files (API and code paths) |
| [docs/REPO_SOURCES.md](docs/REPO_SOURCES.md) | The code/repo path: generating training data from a source repository |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Tuning workers, estimating timings, resuming interrupted runs |

## Prerequisites

```bash
pip install -r requirements.txt             # all scripts (pinned versions)
# or manually:
pip install pyyaml                          # all scripts
pip install sentence-transformers scikit-learn joblib  # routers (train_router.py, train_code_router.py) + bootstrap dedup
# vLLM server required for bootstrap_traces.py / bootstrap_code_traces.py (see their --vllm-url flag)
```

> **Note:** The router classifier artifact (`router_classifier.joblib`) is sensitive to the
> sklearn version used to train it. Use `requirements.txt` to ensure the version used at
> inference matches the version used to generate the file.

## Repository structure

```
apisynth/
├── scripts/              # Generic, reusable scripts
│   ├── pipeline.py       # Full pipeline orchestrator: sweep→run→thinking→enrich→evolve→router
│   ├── sweep.py          # Discover valid param values; confirm variant combinations
│   ├── run.py            # Generate training JSONL via real API calls
│   ├── add_thinking.py   # Add structured thinking traces to training records (no API calls)
│   ├── enrich_schema.py  # Add schema + intent_category fields to existing records (no API calls)
│   ├── evolve_questions.py  # Diversify questions via LLM mutation (Claude API)
│   ├── eval.py           # 3-tier evaluation: format validity, param F1, executability
│   ├── gen_dpo.py        # Generate on-policy DPO preference pairs via live API validator
│   ├── bootstrap_traces.py  # STaR: run trained model, verify outputs, write new traces
│   ├── gen_router_data.py  # Generate router classifier training data (no API calls)
│   ├── train_router.py   # Train a logistic-regression intent router
│   ├── utils.py          # Shared utilities (humanize, PAGE_SIZES, extract_schema, infer_intent)
│   ├── repo_pipeline.py  # Code-path orchestrator: scan→generate (see docs/REPO_SOURCES.md)
│   └── repo/             # Code/repo path — generate training data from a source repository
│       ├── scan_repo.py            # AST-extract code units (function/method/class/api_call)
│       ├── generate_from_code.py   # Units → {type:"code", question, thinking, output} records
│       ├── loader.py               # Parse repo.yaml into a RepoConfig
│       ├── gen_code_dpo.py         # DPO pairs verified by eval.py validators (no API)
│       ├── evolve_code_questions.py  # Deterministic template paraphrase (no LLM)
│       ├── gen_code_router_data.py # Router data keyed by target file path
│       ├── train_code_router.py    # Train the code-unit intent router
│       └── bootstrap_code_traces.py  # STaR: model proposes, validators verify
├── repos/               # Code-path repo configs (repo.yaml per source repository)
│   └── example/repo.yaml
├── apis/
│   └── <vendor>/
│       ├── <endpoint>/
│       │   └── config.yaml   # Endpoint definition, param metadata, sweep status
│       └── generate_holdouts.py  # Generate static holdout evaluation sets
└── data/                 # Generated output (gitignored)
    └── <vendor>/
        └── <endpoint>/
            ├── training.jsonl   # Generated by run.py + enriched by pipeline
            ├── holdout.jsonl    # Generated by generate_holdouts.py (static evaluation set)
            └── dpo.jsonl        # DPO preference data (if generated)
        └── router/
            ├── router_train.jsonl
            ├── router_test.jsonl
            └── router_classifier.joblib
```

## config.yaml schema

Each endpoint directory contains a `config.yaml` describing the API and its parameters. Key sections:

| Section | Purpose |
|---------|---------|
| `endpoint` | Method, path, base URL, vendor name |
| `auth` | Token env var + CLI fallback command |
| `limits` | Worker count, rate limits |
| `params` | Query parameters with type, values, sweep config |
| `path_params` | Path parameters (e.g. `{id}`) with sweep config |
| `parent` | For nested list endpoints — how to fetch parent IDs |
| `skip_params` | Params to exclude from variant dimensions (vendor-specific) |
| `training` | Target records per variant, output path template |
| `status` | Auto-populated by sweep.py: valid values, confirmed variants |

## Workflow

### Run the full pipeline (recommended)

```bash
python scripts/pipeline.py --vendor-dir apis/<vendor>
```

Runs all steps in sequence, skipping anything already done. Prints a status
table before and after showing record counts, schema coverage, and router state.

```bash
# Common options
python scripts/pipeline.py --vendor-dir apis/<vendor> --skip-evolve   # skip LLM question mutation
python scripts/pipeline.py --vendor-dir apis/<vendor> --skip-router   # skip router training
python scripts/pipeline.py --vendor-dir apis/<vendor> --from-step enrich  # resume mid-pipeline
python scripts/pipeline.py --vendor-dir apis/<vendor> --dry-run       # preview without running

# Evolve options (Claude API)
python scripts/pipeline.py --vendor-dir apis/<vendor> \
    --evolve-per-record 2 \
    --evolve-sample 50     # only evolve a random sample of 50 records per endpoint
```

**Pipeline steps in order:**

| Step | Script | API calls? | What it does |
|------|--------|-----------|--------------|
| `sweep` | `sweep.py` | Yes | Discover valid param values; confirm variant combinations |
| `run` | `run.py` | Yes | Generate training records via live API validation |
| `thinking` | `add_thinking.py` | No | Add structured reasoning traces |
| `enrich` | `enrich_schema.py` | No | Add `schema` + `intent_category` fields |
| `evolve` | `evolve_questions.py` | Claude API | Diversify questions via LLM mutation |
| `router` | `gen_router_data.py` + `train_router.py` | No | Build + train intent classifier |

---

### Individual steps (reference)

### 1. Sweep — discover valid values and confirm variants
```bash
python scripts/sweep.py --config apis/<vendor>/<endpoint>/config.yaml
```
Sweeps integer params, confirms variant combinations via live API calls, and writes results back to the `status` section of `config.yaml`.

### 2. Generate training data
```bash
python scripts/run.py --config apis/<vendor>/<endpoint>/config.yaml
# Output: data/<vendor>/<endpoint>/training.jsonl
```
Generates question/param pairs for each confirmed variant, validates via real API calls, writes JSONL records. Each record includes `schema` and `intent_category` fields.

### 3. Add thinking traces
```bash
python scripts/add_thinking.py --input-dir data/<vendor>
# Enriches training.jsonl records with a "thinking" field (deterministic, no API calls)
```
Use `--force` to re-generate traces for records that already have one, `--dry-run --sample 5` to preview traces without writing.

### 4. Enrich schema and intent
```bash
python scripts/enrich_schema.py --vendor-dir apis/<vendor>
# Adds schema + intent_category to existing records (no API calls, safe to re-run)
```
Use when enriching pre-existing records that were generated before these fields were added.

### 5. Evolve questions (optional, Claude API)
```bash
python scripts/evolve_questions.py --input-dir data/<vendor> --per-record 1 --sample 50
# Appends LLM-mutated question variants to training.jsonl files
```
Requires `ANTHROPIC_API_KEY`. Uses Haiku for simple mutations, Sonnet for complexity axis.

### 6. Generate DPO pairs (optional)
```bash
python scripts/gen_dpo.py --config apis/<vendor>/<endpoint>/config.yaml
# Output: data/<vendor>/<endpoint>/dpo.jsonl
```

### 7. Evaluate predictions
```bash
python scripts/eval.py \
    --predictions data/<vendor>/<endpoint>/holdout.jsonl \
    --holdout     data/<vendor>/<endpoint>/holdout.jsonl
```
Scores predictions on format validity, parameter F1, and optionally live executability.

### 8. Generate router data
```bash
python scripts/gen_router_data.py --apis-dir apis/<vendor>
# Output: data/<vendor>/router/router_{train,test}.jsonl
```
Generates (question, route_key) pairs for all endpoints in a vendor directory — no API calls required.

### 9. Train the router
```bash
python scripts/train_router.py --data-dir data/<vendor>/router
# Output: data/<vendor>/router/router_classifier.joblib
```

### 10. STaR bootstrap (optional)
```bash
python scripts/bootstrap_traces.py \
    --model <model-name> \
    --vllm-url http://<host>:8000 \
    --output data/<vendor>/bootstrapped/training.jsonl \
    --verify          # recommended: validates each model output against the live API
    --temperature 0.3 # non-zero temperature prevents trace collapse
# Requires a vLLM server running the trained model
```
Runs the trained model against prompts, verifies API calls, and writes successful (prompt → thinking + api_call) pairs as new training records.

### 11. Generate holdout evaluation sets
```bash
python apis/<vendor>/generate_holdouts.py
# Output: data/<vendor>/<endpoint>/holdout.jsonl (30 records per endpoint, kept out of training)
```

## Adding a new vendor

1. Create `apis/<vendor>/` and add subdirectories for each endpoint
2. Write a `config.yaml` per endpoint following the schema above
3. Run the full pipeline:
   ```bash
   python scripts/pipeline.py --vendor-dir apis/<vendor>
   ```
   This handles sweep → run → thinking → enrich → evolve → router in sequence,
   skipping steps that are already complete.

> **Note on committing vendor files:** `apis/` is gitignored by default — vendor configs
> are intended to be local-only. If your vendor includes Python scripts that override
> pipeline behavior (e.g., a custom `run.py` or `generate_holdouts.py`), add a gitignore
> exception to track those files:
> ```
> !apis/<vendor>/
> !apis/<vendor>/**
> ```
> See `apis/videoamp/` as the reference implementation.

## Auth

The `auth` section of each config specifies:
- `env_var`: environment variable name for the access token
- `cli_fallback`: shell command to retrieve the token if the env var is not set

> **Note:** `cli_fallback` is tokenized with `shlex.split` — use simple commands without shell quoting (e.g., `<vendor> config get --key token`). If the command requires special shell syntax, set `env_var` directly instead.

## Rate limits

Configure `limits.workers` in each config. Two workers is a safe default for most APIs.

## skip_params

Params listed in a config's `skip_params` are excluded from variant dimensions during sweep and from filter cycling during question generation. Use this for:
- Free-text search params (`name`, `query` — already in the base set)
- Date/time filters (`createdAt`, `updatedAt`)
- Boolean flags that don't form meaningful training variants
- Array ID filters (`networkIds`, `audienceIds`)
