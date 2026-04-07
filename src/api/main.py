from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)
APP = FastAPI(title="MovieLens Hybrid Recommender API", version="1.0.0")


def _default_recommendations_path() -> str:
    return os.getenv("RECOMMENDATIONS_PATH", "data/processed/recommendations")


def _default_metrics_path() -> str:
    return os.getenv("METRICS_PATH", "data/processed/metrics/metrics.json")


def _default_seen_interactions_path() -> str:
    return os.getenv("SEEN_INTERACTIONS_PATH", "data/processed/seen_interactions")


@lru_cache(maxsize=1)
def _load_recommendations() -> pd.DataFrame:
    path = _default_recommendations_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recommendations not found at path={path}. Run the training pipeline first.")
    LOGGER.info("Loading recommendations from %s", path)
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_metrics() -> Dict[str, Any]:
    path = _default_metrics_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics not found at path={path}. Run the training pipeline first.")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def _load_seen_interactions() -> pd.DataFrame:
    path = _default_seen_interactions_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=["userId", "movieId"])
    return pd.read_parquet(path)[["userId", "movieId"]]


@APP.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@APP.get("/recommend")
def recommend(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    try:
        recs = _load_recommendations()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    seen = _load_seen_interactions()
    seen_movie_ids = set(seen[seen["userId"] == user_id]["movieId"].tolist())

    user_recs = recs[recs["userId"] == user_id]
    if seen_movie_ids:
        user_recs = user_recs[~user_recs["movieId"].isin(seen_movie_ids)]
    user_recs = user_recs.sort_values("rank").head(k)
    if user_recs.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user_id={user_id}")

    records = user_recs[
        ["movieId", "title", "genres", "rank", "als_score", "content_score", "final_score", "explanation"]
    ].to_dict(orient="records")
    return {"user_id": user_id, "k": k, "recommendations": records}


@APP.get("/metrics")
def metrics() -> Dict[str, Any]:
    try:
        return _load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@APP.get("/metrics/rows")
def metrics_rows() -> Dict[str, Any]:
    try:
        raw_metrics = _load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = []
    for key, value in raw_metrics.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({"metric": key, "value": numeric_value})

    rows.sort(key=lambda item: item["metric"])
    return {"rows": rows}


@APP.get("/metrics/value")
def metric_value(metric: str = Query(..., min_length=1)) -> Dict[str, float | str]:
    try:
        raw_metrics = _load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if metric not in raw_metrics:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found")

    try:
        value = float(raw_metrics[metric])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Metric '{metric}' is not numeric") from exc

    return {"metric": metric, "value": value}


@APP.post("/reload")
def reload_cache() -> Dict[str, str]:
    _load_recommendations.cache_clear()
    _load_metrics.cache_clear()
    _load_seen_interactions.cache_clear()
    return {"status": "reloaded"}
