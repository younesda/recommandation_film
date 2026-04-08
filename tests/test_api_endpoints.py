from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import APP


def test_api_metrics_and_recommend() -> None:
    tmp_path = Path(".tmp_api_test") / str(uuid4())
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        rec_path = tmp_path / "recommendations.parquet"
        metrics_path = tmp_path / "metrics.json"
        seen_path = tmp_path / "seen.parquet"
        history_path = tmp_path / "metrics_history.jsonl"

        rec_df = pd.DataFrame(
            [
                {
                    "userId": 1,
                    "movieId": 101,
                    "title": "A",
                    "genres": "Comedy",
                    "rank": 1,
                    "als_score": 4.5,
                    "content_score": 0.2,
                    "final_score": 0.7,
                    "explanation": "x",
                },
                {
                    "userId": 1,
                    "movieId": 102,
                    "title": "B",
                    "genres": "Drama",
                    "rank": 2,
                    "als_score": 4.2,
                    "content_score": 0.1,
                    "final_score": 0.6,
                    "explanation": "y",
                },
            ]
        )
        rec_df.to_parquet(rec_path)
        pd.DataFrame([{"userId": 1, "movieId": 101}]).to_parquet(seen_path)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "rmse": 0.7,
                    "mae": 0.5,
                    "precision_at_10": 0.1,
                    "catalog_coverage_ratio": 0.4,
                    "recommendation_rows": 2.0,
                },
                handle,
            )
        with open(history_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"generated_at_utc": "2026-04-07T22:00:00+00:00", "rmse": 0.9, "mae": 0.6}) + "\n")
            handle.write(json.dumps({"generated_at_utc": "2026-04-07T23:00:00+00:00", "rmse": 0.7, "mae": 0.5}) + "\n")

        os.environ["RECOMMENDATIONS_PATH"] = str(rec_path)
        os.environ["METRICS_PATH"] = str(metrics_path)
        os.environ["SEEN_INTERACTIONS_PATH"] = str(seen_path)
        os.environ["METRICS_HISTORY_PATH"] = str(history_path)

        client = TestClient(APP)
        client.post("/reload")

        health_resp = client.get("/health")
        assert health_resp.status_code == 200

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert metrics_resp.json()["rmse"] == 0.7

        rec_resp = client.get("/recommend", params={"user_id": 1, "k": 10})
        assert rec_resp.status_code == 200
        movie_ids = [item["movieId"] for item in rec_resp.json()["recommendations"]]
        assert 101 not in movie_ids
        assert 102 in movie_ids

        history_resp = client.get("/metrics/history", params={"metric": "rmse"})
        assert history_resp.status_code == 200
        assert len(history_resp.json()["points"]) == 2
        assert history_resp.json()["points"][-1]["value"] == 0.7

        history_rows_resp = client.get("/metrics/history/rows", params={"limit": 2})
        assert history_rows_resp.status_code == 200
        assert len(history_rows_resp.json()["rows"]) >= 2

        summary_resp = client.get("/dashboard/summary")
        assert summary_resp.status_code == 200
        assert summary_resp.json()["rmse"] == 0.7
        assert summary_resp.json()["latest_run_at"] == "2026-04-07T23:00:00+00:00"

        genres_resp = client.get("/dashboard/genres", params={"limit": 5})
        assert genres_resp.status_code == 200
        genres_rows = genres_resp.json()["rows"]
        assert any(row["genre"] == "Comedy" for row in genres_rows)

        movies_resp = client.get("/dashboard/movies", params={"limit": 5})
        assert movies_resp.status_code == 200
        assert movies_resp.json()["rows"][0]["movieId"] in {101, 102}

        distribution_resp = client.get("/dashboard/final-score-distribution", params={"bins": 4})
        assert distribution_resp.status_code == 200
        assert sum(row["count"] for row in distribution_resp.json()["rows"]) == len(rec_df)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
