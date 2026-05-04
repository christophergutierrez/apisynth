# apisynth

Generate validated training data mapping natural-language questions to structured API calls.

Each script reads a question, calls the real API to validate the params, then writes a JSONL record:
```json
{"question": "Get 10 episodes from network 5 with CoR 25", "api_call": {"endpoint": "GET /external/v1/content/episodes", "params": {"pageSize": 10, "networkId": 5, "currencyOfRecord": 25}}}
```

The training data teaches an LLM *intent → API parameters*, not API responses.

## Structure

```
apisynth/
├── apis/
│   └── videoamp/
│       └── episodes/
│           ├── qa.py      # Core: question → validate → write JSONL record
│           ├── run.py     # Orchestrator: runs 100 (question, params) pairs
│           └── probe.py   # Discovery: find valid networkId / programId values
└── data/                  # Generated output (gitignored)
    └── videoamp/
        └── episodes/
            ├── training.jsonl
            └── run.log.jsonl
```

## Usage

### Generate all 100 training records
```bash
cd apis/videoamp/episodes
python run.py
# Output: ../../data/videoamp/episodes/training.jsonl
```

### Generate a single record
```bash
echo "Get 10 episodes from network 5" > q.txt
python qa.py --question q.txt --network-id 5 --page-size 10
```

### Probe for valid filter values
```bash
python probe.py
# Prints valid networkId and programId values with episode counts
```

## Auth

Token resolution order:
1. `VIDEOAMP_ACCESS_TOKEN` env var
2. `videoamp config get --key access_token` CLI

## Rate limits

- Per-user: 400 req/min
- Per-tenant: 4,000 req/hour (binding constraint at ~66 req/min)
- Recommended parallelism: 2 workers (stays under 50% of tenant quota)
