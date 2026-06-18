# Repo Sources — the code/repo path

How apisynth generates training data from a **source-code repository** instead of a live REST API.

This is the canonical guide for the code path. For the API path see [README.md](../README.md),
[ARCHITECTURE.md](ARCHITECTURE.md), and [TUTORIAL.md](TUTORIAL.md). Record schemas for both paths
live in [DATA_FORMATS.md](DATA_FORMATS.md).

---

## What it is

The API path teaches a model *natural language → API call*, using a **live API as the verifier** —
a record is kept only if a real request accepts it. The code path teaches a model
*natural language → code unit* (which function/method/class/API-call a question is about), using
**deterministic structural validators as the verifier**. A repository is scanned with Python's
`ast` module into code units; questions and reasoning traces are generated from those units; and
every preference pair / bootstrap trace is checked against the validators in
[`scripts/eval.py`](../scripts/eval.py) rather than against a network service.

The practical consequence: **the whole code path is offline and needs no API key** — the one
exception is STaR bootstrap, which calls a model server to generate new traces (but still verifies
them offline). Determinism is preserved end-to-end (no unseeded randomness; SHA-256 of
`"<file>:<name>"` where hashing is needed; fixed `SEED = 42` for shuffles).

### The two paths side by side

| Stage | API path (live-API verifier) | Code path (validators-as-verifier) |
|-------|------------------------------|-------------------------------------|
| Discovery | `sweep.py` (probe the API) | `scan_repo.py` (AST extraction) |
| Generate + thinking | `run.py` + `add_thinking.py` | `generate_from_code.py` |
| Orchestrator | `pipeline.py` | `repo_pipeline.py` |
| DPO pairs | `gen_dpo.py` | `gen_code_dpo.py` |
| Question evolution | `evolve_questions.py` (Claude API) | `evolve_code_questions.py` (deterministic, **no LLM**) |
| Router data + train | `gen_router_data.py` + `train_router.py` | `gen_code_router_data.py` + `train_code_router.py` |
| STaR bootstrap | `bootstrap_traces.py` (live API verify) | `bootstrap_code_traces.py` (validator verify) |
| Verifier | live API response | `eval.py` code rubric |

All code-path scripts live in [`scripts/repo/`](../scripts/repo/) except the orchestrator
[`scripts/repo_pipeline.py`](../scripts/repo_pipeline.py) and the shared rubric in
[`scripts/eval.py`](../scripts/eval.py).

---

## Record formats (summary)

The code path uses a `type: "code"` discriminator. Full specs are in
[DATA_FORMATS.md](DATA_FORMATS.md#code-path-formats).

```json
{
  "type": "code",
  "question": "How do I use `scan_repo`?",
  "thinking": "Entity: function scan_repo\nFile: scripts/repo/scan_repo.py\n...",
  "output": {"unit": "function", "name": "scan_repo",
             "file": "scripts/repo/scan_repo.py", "signature": "scan_repo(...)"}
}
```

- `output.unit` ∈ `{function, method, class, api_call}`; method units also carry `class`.
- DPO records are `{type, question, chosen, rejected}` (chosen/rejected are `output` dicts).
- Router records are `{question, route_key}` where `route_key` is the unit's relative file path.
- Bootstrap records add `source: "bootstrap"`; evolved records add `source: "evol"`.

---

## Quick start

```bash
# 1. Point a repo.yaml at a source tree (an example ships in repos/example/).
# 2. Scan → generate training + holdout in one step:
python scripts/repo_pipeline.py --repo-dir repos/example

# Output: data/repos/example-repo/training.jsonl + holdout.jsonl
```

That single command covers discovery (scan) and generation (questions + thinking + the
deterministic train/holdout split). The remaining stages — DPO, evolution, router, bootstrap —
are optional augmentations run separately, mirroring the API path.

---

## repo.yaml schema

A `repo.yaml` describes one repository. It may be written flat or wrapped under a `repo:` key;
the `extraction`, `generation`, and `output` sections are top-level siblings of `repo:`. Parsed by
[`scripts/repo/loader.py`](../scripts/repo/loader.py) into a `RepoConfig`.

```yaml
repo:
  name: example-repo              # required; used as the output sub-directory (path-safe)
  path: ../../tests/fixtures/sample_repo   # local path (resolved relative to repo.yaml)
  # url / branch / commit:        # alternative to path — clone a git repo instead
  language: python                # default: python
  include: ["**/*.py"]            # default: ["**/*.py"]
  exclude: []                     # default: []
extraction:
  units: [functions, classes]     # default: [functions, classes]; also methods, api_calls
generation:
  target_records: 500             # default: 500 (deterministic cap by SHA-256 ordering)
  holdout_ratio: 0.15             # default: 0.15
  holdout_strategy: hash          # default: hash (per-unit threshold); or "stratified" (per type)
  thinking_style: deterministic   # "deterministic"→linear traces (default); "hybrid"→QOC traces
```

Either `path` (a local directory) **or** `url` (a git remote, cloned to a temp dir) must be set.
A relative `path` is resolved against the `repo.yaml`'s own directory.

| Field | Default | Notes |
|-------|---------|-------|
| `name` | — (required) | Output sub-directory; rejected if it contains `/`, `\`, `..`, is absolute, or `.`/empty |
| `path` / `url` | — | One is required; `url` clones to a temp dir (supports `branch`/`commit`) |
| `include` / `exclude` | `["**/*.py"]` / `[]` | Glob patterns for file selection |
| `extraction.units` | `[functions, classes]` | Plural names; mapped to unit types `function`/`class`/`method`/`api_call` |
| `generation.target_records` | `500` | Deterministic cap (sort by SHA-256 of `<file>:<name>`, take first N) |
| `generation.holdout_ratio` | `0.15` | Fraction held out |
| `generation.holdout_strategy` | `hash` | `hash` = per-unit SHA-256 threshold (~ratio); `stratified` = exact `round(ratio·n)` per unit type |
| `generation.thinking_style` | `deterministic` | `deterministic` → linear; `hybrid` → QOC |

---

## Stage 1 — scan → generate (`repo_pipeline.py`)

```bash
python scripts/repo_pipeline.py --repo-dir repos/example
python scripts/repo_pipeline.py --repo-dir repos/example --dry-run        # counts only, no files
python scripts/repo_pipeline.py --repo-dir repos/example --data-dir /tmp/out  # override output root
```

`repo_pipeline.run_pipeline()` calls `generate_from_repo(config)` in-process: it scans the repo
([`scan_repo.py`](../scripts/repo/scan_repo.py), a raw `ast` extractor that returns every
`function`/`method`/`class`/`api_call` unit), filters by `extraction.units`, caps to
`target_records`, generates one record per unit with a deterministic thinking trace, then splits
into train/holdout. Output goes to `data/repos/<name>/{training,holdout}.jsonl`.

- **Skip-if-done:** if both output files already exist and are non-empty, the run is skipped
  (delete the outputs to regenerate). No checkpoint files.
- **Deterministic split:** a unit is holdout iff `sha256("<file>:<name>")` falls in the holdout
  band; stable across processes and `PYTHONHASHSEED`. The partition is complete and disjoint.

`scan_repo.py` has no CLI of its own — it is a library used by `repo_pipeline.py` and
`generate_from_code.py`. To run scan→generate without the orchestrator (e.g. to use
`--output-dir`), call `generate_from_code.py` directly:

```bash
python scripts/repo/generate_from_code.py repos/example/repo.yaml --output-dir data/repos --dry-run
```

---

## Stage 2 — DPO preference pairs (`gen_code_dpo.py`)

```bash
python scripts/repo/gen_code_dpo.py \
  --input  data/repos/example-repo/training.jsonl \
  --output data/repos/example-repo/dpo.jsonl
```

For each code record the correct `output` is the **chosen** side. Deterministically corrupted
variants (wrong unit type, perturbed name, wrong file, garbled signature, dropped key,
wrong/missing class) are candidate **rejected** sides. A candidate is kept only if it is a
*strictly-worse* answer than gold — it differs from chosen AND is either malformed (fails the
structural/signature verifier) or semantically wrong (`code_field_accuracy < 1.0`). The first
surviving candidate per record is written. Output `dpo.jsonl` is **appended** to.

The verifier here is `code_format_score` + `code_signature_valid` + `code_field_accuracy` from
`eval.py` — no live API. Use `--dry-run` to preview pairs without writing.

---

## Stage 3 — question evolution (`evolve_code_questions.py`)

```bash
python scripts/repo/evolve_code_questions.py --input-dir data/repos/example-repo --per-record 2
python scripts/repo/evolve_code_questions.py --input data/repos/example-repo/training.jsonl \
  --per-record 2 --dry-run --sample 5
```

Unlike the API path's `evolve_questions.py` (which calls Claude), this is **fully deterministic and
offline** — no LLM, no network, no API key. It produces alternate framings of the same ask along
three axes:

- `paraphrase` — an alternate natural phrasing
- `context` — adds file/module and (for methods) class context
- `task_pattern` — reframes as an implement/use task

Evolved records carry all original fields plus `source: "evol"`, `evol_axis`, and
`evol_seed` (the original question). `output`/`thinking` are preserved unchanged. Records already
tagged `evol` or `bootstrap` are skipped. New records are **appended to the same training.jsonl**.
`--per-record` is 1–3 (default 2); `--sample N` evolves a random sample (seeded, default seed 42).

> `--input-dir` accepts either a single repo dir (`training.jsonl` directly inside) or a
> parent-of-repos dir (`<repo>/training.jsonl`). The `--force` flag is a placeholder (no-op),
> kept for parity with the API-path analog.

---

## Stage 4 — router data + classifier (`gen_code_router_data.py`, `train_code_router.py`)

```bash
# Build (question, route_key) pairs — route_key is the unit's relative file path:
python scripts/repo/gen_code_router_data.py \
  --input-dir data/repos/example-repo \
  --out-dir   data/repos/example-repo/router

# Train a logistic-regression classifier on sentence embeddings:
python scripts/repo/train_code_router.py --data-dir data/repos/example-repo/router
```

`gen_code_router_data.py` emits `{question, route_key}` where `route_key = output["file"]` — the
"which file does this question target" analog of the API router's `vendor/api/name`. It accepts
`--input` and/or `--input-dir` (directory mode globs `training.jsonl` only, at any depth, so a
sibling `holdout.jsonl` never leaks into the split). Records are shuffled with a fixed `SEED = 42`
and split `TRAIN_RATIO = 0.8` into `router_train.jsonl` / `router_test.jsonl`
(default out-dir: `data/repos/router`).

`train_code_router.py` embeds questions with `all-MiniLM-L6-v2` and trains a `LogisticRegression`
(`--C`, default 4.0), saving `<data-dir>/code_router_classifier.joblib` (override with `--out`).
The heavy ML deps (`sentence-transformers`, `scikit-learn`, `joblib`) are imported inside `main`,
so the module imports cleanly without them.

> With a global shuffle on a small corpus, a `route_key` appearing in very few records can land
> entirely in the test split; `train_code_router.py` exits with a clear message if the test set
> contains labels unseen in training. Add more records per file (or re-seed) to avoid it.

---

## Stage 5 — STaR bootstrap (`bootstrap_code_traces.py`)

```bash
# Strict: each record's question is a prompt, its output is the gold to match exactly.
python scripts/repo/bootstrap_code_traces.py \
  --model <adapter-name> --vllm-url http://<host>:8000 \
  --input  data/repos/example-repo/training.jsonl \
  --output data/repos/example-repo/bootstrapped/training.jsonl

# Free-form: prompts file, no gold (format+signature verification only).
python scripts/repo/bootstrap_code_traces.py \
  --model <adapter-name> --prompts prompts.txt \
  --output data/repos/example-repo/bootstrapped/training.jsonl
```

This is the one code-path stage that calls a model: it runs a trained model (vLLM/OpenAI-compatible
`/v1/chat/completions`) against prompts, extracts the `<think>` block and the JSON output, and
**verifies offline** against the `eval.py` rubric:

- **Strict** (gold available from `--input`): keep iff `code_format_score == 1.0` **and**
  `code_signature_valid` is True **and** `code_field_accuracy == 1.0`.
- **Fallback** (free-form `--prompts`, no gold): keep iff format complete and signature valid.

Surviving traces are deduplicated by question embedding (`--dedup-threshold`, default 0.95; set to
1.0 to disable) against existing output records, then written with `source: "bootstrap"`. There is
**no live execution sandbox** — the deterministic validators are the execution-feedback stand-in.
Key flags: `--temperature` (default 0.3, keep > 0 to prevent trace collapse), `--workers`
(default 5), `--dry-run`.

---

## Worked example

The in-tree example at [`repos/example/repo.yaml`](../repos/example/repo.yaml) points at the
fixture repo `tests/fixtures/sample_repo` and extracts functions + classes:

```bash
# Scan → generate
python scripts/repo_pipeline.py --repo-dir repos/example
#   → data/repos/example-repo/training.jsonl + holdout.jsonl

# Inspect a record
python3 -c "
import json
r = json.loads(open('data/repos/example-repo/training.jsonl').readline())
print('Q:    ', r['question'])
print('Unit: ', r['output']['unit'], r['output']['name'])
print('File: ', r['output']['file'])
"

# Augment (all offline except bootstrap)
python scripts/repo/gen_code_dpo.py --input data/repos/example-repo/training.jsonl \
  --output data/repos/example-repo/dpo.jsonl
python scripts/repo/evolve_code_questions.py --input-dir data/repos/example-repo --per-record 2
python scripts/repo/gen_code_router_data.py --input-dir data/repos/example-repo \
  --out-dir data/repos/example-repo/router
python scripts/repo/train_code_router.py --data-dir data/repos/example-repo/router
```

Generated `data/` is gitignored; only `repos/*/repo.yaml` configs are tracked.

---

## Verifier rubric (`scripts/eval.py`)

The code path's offline reward function — the analog of "the API accepted it":

| Function | Returns | Meaning |
|----------|---------|---------|
| `code_format_score(output)` | `1.0` / `0.0` | Structural validity: required keys `{unit, name, file, signature}` present, `unit` in the 4-set, name/file/signature non-empty strings, `class` a string if present |
| `code_signature_valid(output)` | `True`/`False`/`None` | `api_call` signatures parse as expressions; others as `def {sig}: pass`. `None` when no signature (absence ≠ malformation) |
| `code_field_accuracy(predicted, expected)` | dict | Per-field exact match + `field_accuracy`; compares `class` only when expected has a truthy class |
| `score_code_record(predicted, expected, check_signature=False)` | dict | Composes the tiers into a banded composite score |

These mirror the API path's 3-tier eval (format validity → param F1 → executability), with code
fields standing in for params and AST well-formedness standing in for live executability.
