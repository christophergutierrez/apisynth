# Example vendor (offline reference)

A **fictional** API used as a runnable template. Nothing here contacts a real
service — the `base_url`s point at `localhost` and are never called in offline
mode. Use it to see the pipeline end-to-end with no credentials, as a CI
fixture, or as a starting point for a real vendor.

## Endpoints

| Dir | Endpoint | Intents exercised |
|-----|----------|-------------------|
| `episodes/` | `GET /external/v1/content/episodes` | bare-list, paginated, filtered |
| `episode/`  | `GET /external/v1/content/episodes/{episodeId}` | by-id, chained (two-step) |

## Generate data offline

```bash
# List endpoint — bare-list / paginated / filtered records:
python scripts/run.py --config apis/example/episodes/config.yaml --offline

# By-id + chained two-step records:
python scripts/run.py --config apis/example/episode/config.yaml --offline

# Add deterministic reasoning traces (no API, no LLM):
python scripts/add_thinking.py --input-dir data/example
```

Output lands in `data/example/<endpoint>/training.jsonl`.

## What `--offline` means

`--offline` skips the live-API validation that `run.py` normally performs on every
record. Generation itself (questions, params, traces) is already offline; the flag
just bypasses the verification GET. Consequences:

- **Records are structurally valid but UNVERIFIED** — no API has confirmed the
  param combination is accepted. The live path remains the default for real vendors.
- **No `sweep.py`** — offline mode cannot discover variants, so they must be
  pre-confirmed in each config's `status.variants` (as they are here).
- **IDs come from `status`** — for path-param/chained endpoints, valid IDs are read
  from `status.<param>.valid_values` instead of being collected over the network.

To turn this into a real vendor, copy the directory, point `base_url`/`auth` at the
real service, and run `sweep.py` then `run.py` *without* `--offline` to get
execution-verified data.
