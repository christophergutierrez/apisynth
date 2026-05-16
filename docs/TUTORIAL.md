# Tutorial: Adding a New Vendor

This walkthrough adds training data for a fictional `acme` API with two endpoints:
`GET /v1/widgets` (list) and `GET /v1/widgets/{id}` (by ID).

---

## 1. Create the vendor directory structure

```bash
mkdir -p apis/acme/widgets
mkdir -p apis/acme/widget
```

---

## 2. Write config.yaml for the list endpoint

`apis/acme/widgets/config.yaml`:

```yaml
endpoint:
  vendor: acme
  name: widgets
  method: GET
  base_url: https://api.acme.example/v1/widgets

auth:
  env_var: ACME_ACCESS_TOKEN
  cli_fallback: "acme config get --key access_token"

limits:
  workers: 2
  per_user_rpm: 60
  per_tenant_rph: 600

params:
  pageSize:
    type: integer
    sweep:
      values: [1, 5, 10, 25, 50]

training:
  target_per_variant: 30
```

**Key decisions:**
- `base_url` includes the full path to the list endpoint
- `params.pageSize.sweep.values` lists exact values to test (not a range)
- `target_per_variant: 30` means generate 30 questions per confirmed param combination

---

## 3. Write config.yaml for the by-ID endpoint

`apis/acme/widget/config.yaml`:

```yaml
endpoint:
  vendor: acme
  name: widget
  method: GET
  base_url: https://api.acme.example
  path: /v1/widgets/{id}

auth:
  env_var: ACME_ACCESS_TOKEN
  cli_fallback: "acme config get --key access_token"

limits:
  workers: 2
  per_user_rpm: 60
  per_tenant_rph: 600

path_params:
  id:
    sweep:
      range: [1, 200]

training:
  target_per_variant: 30
```

**Key differences from list endpoint:**
- `path` is separate from `base_url` (because the ID goes in the path)
- `path_params.id` instead of `params` (path params use range-based sweeping)

---

## 4. Set the auth token

```bash
export ACME_ACCESS_TOKEN=your_token_here
# or configure the CLI: acme login
```

---

## 5. Sweep the list endpoint

```bash
python scripts/sweep.py --config apis/acme/widgets/config.yaml
```

Expected output:
```
Sweeping query param: pageSize
  Confirming pageSize=1 ... OK (0.3s)
  Confirming pageSize=5 ... OK (0.3s)
  Confirming pageSize=10 ... OK (0.3s)
  ...
Variant confirmation:
  [pageSize=1]  ... confirmed
  [pageSize=5]  ... confirmed
  ...
Updated: apis/acme/widgets/config.yaml
```

The `status` section of `config.yaml` is now populated with confirmed variants.

---

## 6. Sweep the by-ID endpoint

```bash
python scripts/sweep.py --config apis/acme/widget/config.yaml
```

This sweeps integers 1–200 to find valid widget IDs. Results go into
`status.id.valid_values` in the config.

---

## 7. Generate training records

```bash
# List endpoint
python scripts/run.py --config apis/acme/widgets/config.yaml

# By-ID endpoint
python scripts/run.py --config apis/acme/widget/config.yaml
```

Each run makes real API calls to validate each (question, params) pair.
Progress is printed; records are written to `data/acme/widgets/training.jsonl`
and `data/acme/widget/training.jsonl`.

Expected output per endpoint:
```
=== widgets ===
Variant status (target: 30 each):
  ['pageSize']      0/30  need 30
  ...
Generating 150 records across 2 workers...
[  1/150] OK   0.31s  Get 1 widget
[  2/150] OK   0.29s  List 5 widgets
...
Done — 147 passed, 3 failed
```

---

## 8. Add thinking traces

```bash
python scripts/add_thinking.py --input-dir data/acme
```

This enriches every record in `data/acme/*/training.jsonl` with a `thinking` field.
Run with `--sample 3` first to preview what the traces look like:

```bash
python scripts/add_thinking.py --input-dir data/acme --dry-run --sample 3
```

---

## 9. Generate router training data (optional)

If you are using a separate intent router:

```bash
python scripts/gen_router_data.py \
  --apis-dir apis/acme \
  --out-dir data/acme/router
```

Then train the classifier:

```bash
python scripts/train_router.py --data-dir data/acme/router
```

---

## 10. Check what you have

```bash
# Record counts by endpoint
for f in data/acme/*/training.jsonl; do
  echo "$(wc -l < $f) records: $f"
done

# Preview a record with thinking
python3 -c "
import json
r = json.loads(open('data/acme/widgets/training.jsonl').readline())
print('Q:', r['question'])
print('Think:', r['thinking'][:200])
print('API:', r['api_call'])
"
```

---

## Expected file sizes and timings

| Endpoint type | Target records | Sweep time | Run time |
|--------------|---------------:|----------:|----------:|
| Simple list (1–3 params) | 60–120 | 1–3 min | 5–15 min |
| By-ID (50 valid IDs) | 50–100 | 5–10 min | 10–20 min |
| Nested list (parent + child) | 60–120 | 10–20 min | 15–30 min |

Times depend on API latency and rate limits. Use `workers: 1` to stay well under limits.

---

## Common mistakes on first run

**Sweep finds zero confirmed variants**
→ Check auth (`export ACME_ACCESS_TOKEN=...`) and try `curl` manually.

**run.py reports 0 confirmed variants**
→ Sweep must complete first. Check that `config.yaml` has a `status.variants` section.

**Records have wrong question phrasing**
→ Questions come from `gen_questions()` templates in `run.py`. The templates use
`endpoint.name` (humanized) — if the name is `widget`, questions say "Get widget 5".
To customize, add phrasings to the appropriate template section.

**Thinking traces say "Entity: resource"**
→ The endpoint pattern doesn't match any entry in `_EP_ENTITY` in `add_thinking.py`.
Add a new entry for your endpoint path pattern.
