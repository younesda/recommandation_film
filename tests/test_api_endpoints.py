from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import APP


def test_api_metrics_and_recommend(tmp_path: Path) -> None:
    rec_path = tmp_path / "recommendations"
    metrics_path = tmp_path / "metrics.json"
    seen_path = tmp_path / "seen"

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
        json.dump({"rmse": 0.7}, handle)

    os.environ["RECOMMENDATIONS_PATH"] = str(rec_path)
    os.environ["METRICS_PATH"] = str(metrics_path)
    os.environ["SEEN_INTERACTIONS_PATH"] = str(seen_path)

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
