# MovieLens Hybrid Recommender (Spark + FastAPI)

## Run Training Pipeline
```bash
python scripts/run_pipeline.py
```

Optional arguments:
```bash
python scripts/run_pipeline.py --top-k 15 --disable-tags
```

## Run API
```bash
python scripts/run_api.py
```

Endpoints:
- `GET /health`
- `GET /recommend?user_id=1&k=10`
- `GET /metrics`
- `POST /reload`

## Output Artifacts
- `data/processed/recommendations` (Parquet)
- `data/processed/user_profiles` (Parquet)
- `data/processed/metrics/metrics.json`
- `data/processed/metrics/metrics_parquet`
- `data/processed/models/als`
