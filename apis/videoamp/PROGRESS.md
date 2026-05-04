# VideoAmp API Training Progress

## Status Summary

| Endpoint                        | Dir                  | Records | Variants  | Status          |
|---------------------------------|----------------------|---------|-----------|-----------------|
| list-episodes                   | episodes/            | 245     | 8/8       | ✅ Done         |
| list-programs                   | programs/            | 240     | 8/8       | ✅ Done         |
| list-networks                   | networks/            | 120     | 4/4       | ✅ Done         |
| list-network-mediagroups        | media-groups/        | 60      | 2/2       | ✅ Done         |
| list-metric-and-dimension-types | metric-types/        | 30      | 1/1       | ✅ Done         |
| get-episode                     | episode/             | 30      | 1/1       | ✅ Done         |
| get-program                     | program/             | 30      | 1/1       | ✅ Done         |
| get-network                     | network/             | 30      | 1/1       | ✅ Done         |
| get-metrics (content-metric)    | content-metric/      | 0       | 0/1       | ⛔ Blocked      |
| list-ad (measurements)          | measurements/        | 120     | 4/8       | ✅ Done (partial)|
| get-ad (measurement)            | measurement/         | 18      | 1/1       | ✅ Done         |
| list-audiences                  | audiences/           | 240     | 8/8       | ✅ Done         |
| get-audience                    | audience/            | 18      | 1/1       | ✅ Done         |
| list-statuses                   | audience-statuses/   | 120     | 4/4       | ✅ Done         |
| list-export                     | audience-exports/    | 20      | 1/1       | ✅ Done         |
| currency-of-record              | currency-of-record/  | 60      | 2/2       | ✅ Done         |
| me                              | me/                  | 30      | 1/1       | ✅ Done         |
| consents                        | consents/            | 60      | 2/2       | ✅ Done         |

**Total: 1,471 records across 17 endpoints**

---

## Tools

- **`apis/videoamp/sweep.py`** — generic sweep; finds valid param values and confirms variant combinations for any config.yaml
- **`apis/videoamp/run.py`** — generic training data generator; reads confirmed variants, generates varied question/param pairs, validates via real API calls, writes JSONL

---

## Key findings from sweeps

- **networkId:** 39 valid values (1–24 confirmed via episodes; extended to 39 via programs sweep)
- **currencyOfRecord:** `[23, 25, 26, 27]` — sparse, 24 is absent
- **programId:** Sequential from 1; thousands of valid values — pre-seeded 10 known-good IDs (sweep hit 429 at ~3,995 before saving)
- **episodeId:** Sequential from low numbers; hundreds of valid values — pre-seeded 10 known-good IDs
- **measurement.id:** UUID (requestId), not integer — seeded from list endpoint response
- **audience.id:** Integer IDs in ~232k range — pre-seeded `[232962]`
- **pageToken `CAU=`:** Works as valid pagination token across all paginated endpoints

---

## Partial / under-target endpoints

- **measurements (list-ad):** 4/8 variants — `pageToken=CAU=` variants unconfirmed (token is endpoint-specific). 120 records from 4 confirmed variants.
- **audience-exports:** 20 records — only 1 seeded audienceId; 2 phrasings deduplicated (resource==singular). Seed more IDs to reach 30.
- **measurement/audience (get):** 18 records each — same deduplication issue with 1 seeded ID.

---

## Blocked

- **content-metric** (`GET /external/v1/content/metrics/{id}`): Account lacks permission to `POST /external/v1/content/metrics` (403). No UUID obtainable.

---

## Bugs fixed

- **sweep.py:** HTTP 500 added to handled codes (episodeId=9 returned 500, not 404)
- **sweep.py:** 429 now saves partial progress via `RateLimitError` instead of crashing
- **run.py:** Infinite loop in path-param question generator — parallel cycling of id/phrasing cycles hits lcm(10,20)=20 unique questions, can't reach target=30. Fixed with nested loops.
- **run.py:** 429 raises `RateLimitError` and reports cleanly instead of silently dropping records
