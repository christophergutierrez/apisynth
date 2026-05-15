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

## Prerequisites

```bash
pip install pyyaml                          # all scripts
pip install sentence-transformers scikit-learn joblib  # train_router.py only
# vLLM server required for bootstrap_traces.py (see its --vllm-url flag)
```

## Repository structure

```
apisynth/
├── scripts/              # Generic, reusable scripts
│   ├── sweep.py          # Discover valid param values; confirm variant combinations
│   ├── run.py            # Generate training JSONL via real API calls
│   ├── add_thinking.py   # Add structured thinking traces to training records (no API calls)
│   ├── bootstrap_traces.py  # STaR: run trained model, verify outputs, write new traces
│   ├── gen_router_data.py  # Generate router classifier training data (no API calls)
│   ├── train_router.py   # Train a logistic-regression intent router
│   └── utils.py          # Shared utilities (humanize, PAGE_SIZES, skip sets)
├── apis/
│   └── <vendor>/
│       ├── <endpoint>/
│       │   └── config.yaml   # Endpoint definition, param metadata, sweep status
│       ├── sweep.sh          # Vendor wrapper: calls scripts/sweep.py
│       ├── run.sh            # Vendor wrapper: calls scripts/run.py
│       ├── gen_router_data.sh  # Vendor wrapper: calls scripts/gen_router_data.py
│       └── train_router.sh   # Vendor wrapper: calls scripts/train_router.py
└── data/                 # Generated output (gitignored)
    └── <vendor>/
        └── <endpoint>/
            └── training.jsonl
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
Generates question/param pairs for each confirmed variant, validates via real API calls, writes JSONL records.

### 3. Generate router data
```bash
python scripts/gen_router_data.py --apis-dir apis/<vendor>
# Output: data/<vendor>/router/router_{train,test}.jsonl
```
Generates (question, route_key) pairs for all endpoints in a vendor directory — no API calls required.

### 4. Train the router
```bash
python scripts/train_router.py --data-dir data/<vendor>/router
# Output: data/<vendor>/router/router_classifier.joblib
```

### 5. Add thinking traces
```bash
python scripts/add_thinking.py --input-dir data/<vendor>
# Enriches training.jsonl records with a "thinking" field (deterministic, no API calls)
```
Use `--force` to re-generate traces for records that already have one, `--dry-run --sample 5` to preview traces without writing.

### 6. STaR bootstrap (optional)
```bash
python scripts/bootstrap_traces.py \
    --model <model-name> \
    --vllm-url http://<host>:8000 \
    --output data/<vendor>/bootstrapped/training.jsonl
# Requires a vLLM server running the trained model
# Use --verify to validate each generated call against the live API
```
Runs the trained model against prompts, verifies API calls, and writes successful (prompt → thinking + api_call) pairs as new training records.

## Adding a new vendor

1. Create `apis/<vendor>/` and add subdirectories for each endpoint
2. Write a `config.yaml` per endpoint following the schema above
3. Run `sweep.py` to populate the `status` section
4. Run `run.py` to generate training records
5. Optionally create vendor-specific wrapper scripts (`sweep.sh`, `run.sh`, etc.)

## Auth

The `auth` section of each config specifies:
- `env_var`: environment variable name for the access token
- `cli_fallback`: shell command to retrieve the token if the env var is not set

## Rate limits

Configure `limits.workers` in each config. Two workers is a safe default for most APIs.

## skip_params

Params listed in a config's `skip_params` are excluded from variant dimensions during sweep and from filter cycling during question generation. Use this for:
- Free-text search params (`name`, `query` — already in the base set)
- Date/time filters (`createdAt`, `updatedAt`)
- Boolean flags that don't form meaningful training variants
- Array ID filters (`networkIds`, `audienceIds`)
