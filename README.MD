# MovieLens Hybrid Recommender (Spark + FastAPI)

## Quick Start (Local)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_api.py
```

## Quick Start (Colab)
```python
!git clone https://github.com/younesda/recommandation_film.git
%cd recommandation_film
!apt-get -qq install openjdk-17-jdk-headless
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ["PATH"]
os.environ["SPARK_DRIVER_MEMORY"] = "6g"
os.environ["SPARK_EXECUTOR_MEMORY"] = "6g"
!pip -q install -r requirements.txt
!python scripts/run_pipeline.py
```

## API Endpoints
- `GET /health`
- `GET /recommend?user_id=1&k=10`
- `GET /metrics`
- `POST /reload`

## Smoke Test API
```bash
python scripts/smoke_test_api.py --user-id 1 --k 10
```

## Export to PostgreSQL (Grafana)
```bash
python scripts/export_artifacts_to_postgres.py \
  --jdbc-url "jdbc:postgresql://localhost:5432/recommender_system" \
  --user postgres \
  --password password
```

This exports:
- `movie_recommendations`
- `recommender_metrics`
- `user_seen_items`

## Grafana Dashboards
Import these JSON dashboards in Grafana:
- `dashboards/recommender_quality_dashboard.json`
- `dashboards/recommender_catalog_dashboard.json`

## Tests
```bash
pytest -q
```

## Output Artifacts
- `data/processed/recommendations` (Parquet)
- `data/processed/user_profiles` (Parquet)
- `data/processed/seen_interactions` (Parquet)
- `data/processed/metrics/metrics.json`
- `data/processed/metrics/metrics_parquet`
- `data/processed/models/als`
