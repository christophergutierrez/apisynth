# Troubleshooting

Quick reference for common failures in apisynth.

---

## sweep.py

### "No confirmed variants after sweep"
The sweep ran but nothing passed the confirmation step.

**Check first:**
```bash
# Was the token valid?
videoamp me

# Did the sweep actually find valid values?
grep "valid_values" apis/<vendor>/<endpoint>/config.yaml
```

**Common causes:**
- Auth token expired — run `videoamp login`
- The param range in `config.yaml` doesn't overlap real data (e.g., `range: [1, 100]` but IDs start at 10000)
- The endpoint requires a filter param to return results (e.g., `networkId` required to get episodes) — add it to `params` in config before sweeping
- Rate limited mid-sweep — the script will print `WARNING: rate limited`, wait 60s and retry; if it still fails, rerun sweep and it picks up from existing `swept_through`

### "Sweep skipped — N values already known"
Expected. Sweep is incremental. To re-sweep from scratch, delete `status.valid_values` from the relevant param in `config.yaml`.

### "API returned 500 on param Y"
The sweep treats 500 as not-found and moves on. If 500s dominate, the endpoint may require additional headers or the account lacks permissions. Check `PROGRESS.md` for known blocked endpoints.

### Rate limiting mid-sweep
The script backs off automatically (60s). For sustained rate limiting, reduce `workers` in `config.yaml` from 2 to 1, or increase `per_user_rpm` headroom by running during off-hours.

---

## run.py

### "Training file empty / no records written"
```bash
# Check confirmed variants exist
grep -A5 "variants:" apis/<vendor>/<endpoint>/config.yaml | grep confirmed

# Run sweep first if empty
python scripts/sweep.py --config apis/<vendor>/<endpoint>/config.yaml
```

`run.py` exits early if no confirmed variants exist. Sweep must complete successfully first.

### "WARNING: only generated N/M questions for variant"
The phrasing generator ran out of unique questions before hitting the target. Causes:
- Target is too high relative to available phrasing templates
- Endpoint has very few valid ID values (path-param endpoints)

Lower `training.target_per_variant` in `config.yaml` or add phrasing templates to `gen_questions()` in `run.py`.

### "RATE_LIMITED" in output
Run is paused for 60s then retried automatically. If rate limiting is persistent, reduce `workers` to 1.

### Records have wrong param values
Re-run sweep — valid values may have changed since the last sweep. Sweep results are cached in `config.yaml` under `status`.

---

## add_thinking.py

### Thinking trace looks wrong (wrong entity, wrong endpoint)
The deterministic trace is derived from `api_call.endpoint` — if the underlying training record has a wrong endpoint, the trace will reflect that. Check the source `training.jsonl` record, not the thinking trace.

### Thinking for synonym prompts doesn't mention the correct domain note
The domain note lookup uses `(synonym_word, canonical_entity)` as the key. If a synonym isn't being picked up, check `_ENTITY_SYNONYMS` — the synonym word must appear as a substring in the question and be listed there.

### Re-generating traces after changing templates
```bash
python scripts/add_thinking.py --input-dir data/<vendor> --force
```
Without `--force`, records that already have a `thinking` field are skipped.

---

## bootstrap_traces.py

### "no-cli-mapping" warning
The model generated an endpoint that has no CLI equivalent in `CLI_MAP`. The record is skipped from output. To add support, extend `CLI_MAP` in `bootstrap_traces.py` with the new pattern.

### All completions are wrong
Run with `--dry-run` first to inspect outputs before writing. Check that the vLLM server is running and the correct adapter is loaded.

### Verification returning errors on valid calls
The `videoamp` CLI must be authenticated. Run `videoamp login` then retry.

---

## train_router.py

### "Test set has labels not in training data"
A route_key appears in the test split but not in training. Usually caused by an endpoint with very few records that all land in the test split by chance.

**Fix:** Increase records for that endpoint (re-run `run.py`) or reduce `--test-frac`.

### Accuracy below 0.85
Check per-route accuracy in the output. Common causes:
- Too few training records for a route (< 20)
- Two routes have very similar description text (check `route_descriptions.json` — make descriptions more distinct)
- Embedding model `all-MiniLM-L6-v2` is the default; try `all-mpnet-base-v2` for higher accuracy at slower speed

---

## Config validation

### "config.yaml is missing required key"
The trainLLM loader checks for required keys. Run the script and read the error — it names the missing key exactly.

### "unknown training keys"
A key in the `training:` section of `config.yaml` isn't in `_KNOWN_TRAINING_KEYS`. Check for typos.

---

## General

### "Please use 'videoamp login' before running this command"
Token expired. Run `videoamp login` to refresh. Tokens typically last 8 hours.

### Script crashes with `ModuleNotFoundError: yaml`
```bash
pip install pyyaml
```

### Script crashes with `ModuleNotFoundError: sentence_transformers`
Only required for `train_router.py`:
```bash
pip install sentence-transformers scikit-learn joblib
```
