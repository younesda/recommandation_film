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


def _default_metrics_history_path() -> str:
    return os.getenv("METRICS_HISTORY_PATH", "data/processed/metrics/metrics_history.jsonl")


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


@lru_cache(maxsize=1)
def _load_metrics_history() -> list[Dict[str, Any]]:
    path = _default_metrics_history_path()
    if not os.path.exists(path):
        return []

    rows: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


@APP.get("/metrics/history")
def metrics_history(
    metric: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=5000),
) -> Dict[str, Any]:
    history = _load_metrics_history()

    points = []
    for row in history:
        if metric not in row:
            continue
        try:
            numeric_value = float(row[metric])
        except (TypeError, ValueError):
            continue
        points.append({"timestamp": row.get("generated_at_utc"), "value": numeric_value})

    if not points:
        raise HTTPException(status_code=404, detail=f"Metric history '{metric}' not found")

    return {"metric": metric, "points": points[-limit:]}


@APP.get("/metrics/history/rows")
def metrics_history_rows(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    history = _load_metrics_history()
    rows = []

    for record in history[-limit:]:
        timestamp = record.get("generated_at_utc")
        for key, value in record.items():
            if key == "generated_at_utc":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            rows.append({"timestamp": timestamp, "metric": key, "value": numeric_value})

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


@APP.get("/dashboard/summary")
def dashboard_summary() -> Dict[str, Any]:
    try:
        raw_metrics = _load_metrics()
        recs = _load_recommendations()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    summary: Dict[str, Any] = {}
    for key, value in raw_metrics.items():
        try:
            summary[key] = float(value)
        except (TypeError, ValueError):
            continue

    history = _load_metrics_history()
    if history:
        summary["latest_run_at"] = history[-1].get("generated_at_utc")

    summary.setdefault("recommendation_rows", float(len(recs)))
    summary.setdefault("users_covered", float(recs["userId"].nunique()))
    summary.setdefault("movies_recommended", float(recs["movieId"].nunique()))
    return summary


@APP.get("/dashboard/genres")
def dashboard_genres(limit: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    try:
        recs = _load_recommendations()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if recs.empty:
        return {"rows": []}

    exploded = recs[["genres", "final_score"]].copy()
    exploded["genre"] = exploded["genres"].fillna("").str.split("|")
    exploded = exploded.explode("genre")
    exploded["genre"] = exploded["genre"].fillna("").str.strip()
    exploded = exploded[exploded["genre"] != ""]

    if exploded.empty:
        return {"rows": []}

    grouped = (
        exploded.groupby("genre", as_index=False)
        .agg(
            recommendation_count=("genre", "size"),
            avg_final_score=("final_score", "mean"),
        )
        .sort_values(["recommendation_count", "avg_final_score", "genre"], ascending=[False, False, True])
    )
    total_recommendations = float(grouped["recommendation_count"].sum())
    grouped["share"] = grouped["recommendation_count"] / total_recommendations
    grouped = grouped.head(limit).reset_index(drop=True)

    rows = [
        {
            "genre": str(row["genre"]),
            "recommendation_count": int(row["recommendation_count"]),
            "avg_final_score": float(row["avg_final_score"]),
            "share": float(row["share"]),
        }
        for _, row in grouped.iterrows()
    ]
    return {"rows": rows}


@APP.get("/dashboard/movies")
def dashboard_movies(limit: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    try:
        recs = _load_recommendations()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if recs.empty:
        return {"rows": []}

    grouped = (
        recs.groupby(["movieId", "title", "genres"], as_index=False, dropna=False)
        .agg(
            exposure_count=("userId", "size"),
            avg_final_score=("final_score", "mean"),
            avg_rank=("rank", "mean"),
        )
        .sort_values(["exposure_count", "avg_final_score", "avg_rank"], ascending=[False, False, True])
        .head(limit)
        .reset_index(drop=True)
    )

    rows = [
        {
            "movieId": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "exposure_count": int(row["exposure_count"]),
            "avg_final_score": float(row["avg_final_score"]),
            "avg_rank": float(row["avg_rank"]),
        }
        for _, row in grouped.iterrows()
    ]
    return {"rows": rows}


@APP.get("/dashboard/final-score-distribution")
def dashboard_final_score_distribution(bins: int = Query(10, ge=3, le=30)) -> Dict[str, Any]:
    try:
        recs = _load_recommendations()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    scores = recs["final_score"].dropna()
    if scores.empty:
        return {"rows": []}

    min_score = float(scores.min())
    max_score = float(scores.max())
    if min_score == max_score:
        return {
            "rows": [
                {
                    "bin_label": f"{min_score:.4f}",
                    "bin_start": min_score,
                    "bin_end": max_score,
                    "count": int(len(scores)),
                }
            ]
        }

    buckets = pd.cut(scores, bins=bins, include_lowest=True)
    counts = buckets.value_counts().sort_index()
    rows = [
        {
            "bin_label": str(interval),
            "bin_start": float(interval.left),
            "bin_end": float(interval.right),
            "count": int(count),
        }
        for interval, count in counts.items()
    ]
    return {"rows": rows}


@APP.post("/reload")
def reload_cache() -> Dict[str, str]:
    _load_recommendations.cache_clear()
    _load_metrics.cache_clear()
    _load_seen_interactions.cache_clear()
    _load_metrics_history.cache_clear()
    return {"status": "reloaded"}
