# Performance Tuning

How to balance speed, data quality, and rate limit compliance.

---

## Choosing `workers`

`workers` controls how many API calls run in parallel during sweep and run.

| API rate limit | Recommended workers |
|---------------|---------------------|
| < 30 req/min  | 1 |
| 30–120 req/min | 2 |
| 120–300 req/min | 4 |
| > 300 req/min | 8 |

The VideoAmp API allows 400 req/min per user and 4,000/hour per tenant.
With `workers: 2` and ~0.3s per call, peak usage is ~120 req/min — well within limits.

**Rule of thumb:** Start at 2. If you see `RATE_LIMITED` in output, drop to 1.

---

## Estimating sweep time

```
sweep_time ≈ (range_size / workers) × avg_call_latency
```

Example: sweeping `id` over range 1–500, `workers: 2`, 0.3s per call:
```
(500 / 2) × 0.3s = 75 seconds
```

For path params with large ranges, limit sweep scope with a narrower `range`:
```yaml
path_params:
  id:
    sweep:
      range: [1, 200]   # sweep 200 values instead of 10000
```

Sweep results are cached — if 100 valid IDs are enough for training, stop at 100.

---

## Estimating training data generation time

```
run_time ≈ (total_records_needed / workers) × avg_call_latency
```

Example: 30 target records × 5 confirmed variants = 150 records, workers: 2, 0.3s/call:
```
(150 / 2) × 0.3s = 22.5 seconds
```

Actual time is higher because some questions fail validation and the script generates
extras to compensate.

---

## `--dry-run` and `--sample` usage

Use these before a full run to catch config problems early:

```bash
# Check what sweep would do without making API calls
python scripts/sweep.py --config apis/<vendor>/<ep>/config.yaml --dry-run

# Preview 5 thinking traces before enriching all records
python scripts/add_thinking.py --input-dir data/<vendor> --dry-run --sample 5
```

`--dry-run` on `run.py` shows variant counts and deficits without calling the API
or writing records — useful for checking that sweep results are sane before investing
time in data generation.

---

## Resuming interrupted runs

**Sweep:** Safe to rerun at any time. The script reads existing `status.swept_through`
and starts from where it left off. If the config shows `swept_through: 150`, sweep
resumes from value 151.

**run.py:** Counts existing records before generating. If a run was interrupted after
writing 20 of 30 target records, rerunning generates only the remaining 10.

**add_thinking.py:** Skips records that already have a `thinking` field unless
`--force` is passed. Safe to rerun.

---

## Tuning `target_per_variant`

More records = better coverage but longer training.

| Model size | Min useful records/variant | Recommended |
|-----------|------------------------:|------------:|
| 1.5B | 10 | 20–30 |
| 7–8B | 20 | 30–50 |
| 27B+ | 30 | 50–100 |

For synonym/disambiguation variants (e.g., "campaign" → measurements), use the higher
end — these require more examples to override pretrained priors.

---

## Monitoring progress

```bash
# Watch records accumulate in real time
watch -n 5 'wc -l data/<vendor>/*/training.jsonl'

# Check how many records exist vs target
python scripts/run.py --config apis/<vendor>/<ep>/config.yaml --dry-run
```

---

## When to re-run sweep

Re-run sweep when:
- New IDs were added to the API (e.g., new programs created)
- Auth token changed and the account now has access to more data
- An endpoint previously returned 500s and now works
- You suspect `valid_values` are stale (> 30 days old)

To force a full re-sweep, delete the `status` section from `config.yaml` and rerun.
