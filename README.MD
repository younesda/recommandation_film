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

The ranking stack now uses:
- explicit ALS for `RMSE` / `MAE`
- implicit ALS for collaborative candidate generation
- XGBoost ranker for final recommendation ordering

## API Endpoints
- `GET /health`
- `GET /recommend?user_id=1&k=10`
- `GET /metrics`
- `GET /metrics/rows`
- `GET /metrics/value?metric=rmse`
- `GET /metrics/history?metric=rmse`
- `GET /metrics/history/rows`
- `GET /dashboard/summary`
- `GET /dashboard/genres?limit=10`
- `GET /dashboard/movies?limit=20`
- `GET /dashboard/final-score-distribution?bins=10`
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
- `dashboards/netflix_api_dashboard.json` (API overview)
- `dashboards/netflix_api_ranking_dashboard.json` (ranking-focused API dashboard)
- `dashboards/netflix_api_catalog_dashboard.json` (catalog + live API dashboard)

## Grafana Without PostgreSQL (Parquet + API)
Use FastAPI as datasource with Grafana Infinity plugin.

1. Run pipeline and API:
```bash
python scripts/run_pipeline.py
python scripts/run_api.py
```

2. Install Grafana plugin:
`yesoreyeram-infinity-datasource`

3. Add Infinity datasource:
- Base URL: `http://localhost:8000` (or your server URL)
- Access mode: `Server`

4. Create Stat panels using URL queries:
- `/metrics/value?metric=rmse`
- `/metrics/value?metric=mae`
- `/metrics/value?metric=precision_at_10`
- `/metrics/value?metric=recall_at_10`
- `/metrics/value?metric=ndcg_at_10`

Use field `value` for each Stat panel.

5. Create Table panel with URL:
- `/metrics/rows`

Set root selector to `rows`.

6. Create Recommendation table with URL:
- `/recommend?user_id=1&k=10`

Set root selector to `recommendations`.

Netflix-style option:
- Import:
  - `dashboards/netflix_api_dashboard.json`
  - `dashboards/netflix_api_ranking_dashboard.json`
  - `dashboards/netflix_api_catalog_dashboard.json`
- Set Infinity datasource variable `DS_INFINITY`
- Keep `api_base` as `http://localhost:8000` (or your server URL)

## Hosted Demo
The repo now includes a lightweight hosted demo under `hosting/demo_data/`:
- reference metrics from the best run
- a small recommendation sample for a few users
- history data for dashboard tables

### Run the hosted demo locally with Docker
```bash
docker build -t movielens-api .
docker run -p 8000:8000 movielens-api
```

### Deploy on Render
This repo includes `render.yaml` and a Docker image definition.

The default hosted deployment uses:
- `RECOMMENDATIONS_PATH=/app/hosting/demo_data/recommendations.parquet`
- `METRICS_PATH=/app/hosting/demo_data/metrics.json`
- `SEEN_INTERACTIONS_PATH=/app/hosting/demo_data/seen_interactions.parquet`
- `METRICS_HISTORY_PATH=/app/hosting/demo_data/metrics_history.jsonl`

This is ideal for a student demo because it deploys directly from GitHub without uploading the full `data/processed/` directory.

### Full deployment with real pipeline artifacts
If you want the hosted API to serve the complete generated artifacts instead of the lightweight demo, override:
- `RECOMMENDATIONS_PATH`
- `METRICS_PATH`
- `SEEN_INTERACTIONS_PATH`
- `METRICS_HISTORY_PATH`

pointing them to your real `data/processed/...` outputs on the target machine or mounted volume.

## Tests
```bash
pytest -q
```

## Output Artifacts
- `data/processed/recommendations` (Parquet)
- `data/processed/user_profiles` (Parquet)
- `data/processed/seen_interactions` (Parquet)
- `data/processed/metrics/metrics.json`
- `data/processed/metrics/metrics_history.jsonl`
- `data/processed/metrics/metrics_parquet`
- `data/processed/models/als`

## Dashboard Ideas
- Run quality cards: `rmse`, `mae`, `precision_at_10`, `recall_at_10`, `ndcg_at_10`
- Retrieval cards: `val_candidate_recall`, `test_candidate_recall`, `candidate_pool_size`
- Ranker cards: `ranker_val_ndcg_at_k`, `ranker_n_estimators`, `ranking_als_alpha`
- Coverage cards: `users_covered`, `movies_recommended`, `catalog_coverage_ratio`, `user_coverage_ratio`
- Score health: `avg_final_score`, `p50_final_score`, `p90_final_score`, `avg_als_score`, `avg_content_score`
- Catalog exploration: `/dashboard/genres`, `/dashboard/movies`
- Score spread histogram: `/dashboard/final-score-distribution`
- Trend over runs: `/metrics/history?metric=rmse`, `/metrics/history?metric=ndcg_at_10`
