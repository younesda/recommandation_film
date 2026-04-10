# Hosted Demo Data

This folder contains lightweight API artifacts for public hosting:

- `metrics.json`: reference metrics from the best run
- `metrics_history.jsonl`: small history for Grafana tables
- `recommendations.parquet`: demo recommendations for a few users
- `seen_interactions.parquet`: demo seen items used by `/recommend`

The hosted demo is intentionally small so the API can be deployed from GitHub without committing the full `data/processed/` directory.

For a full deployment with the real generated artifacts, override:

- `RECOMMENDATIONS_PATH`
- `METRICS_PATH`
- `SEEN_INTERACTIONS_PATH`
- `METRICS_HISTORY_PATH`
