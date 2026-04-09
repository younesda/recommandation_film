from __future__ import annotations

import math
from itertools import product
from typing import Any, Dict, Sequence

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config.settings import PipelineSettings, RankerSettings


RANKER_FEATURE_COLUMNS = [
    "cf_score",
    "explicit_als_score",
    "als_candidate_score",
    "content_score",
    "content_genre_score",
    "content_tag_score",
    "content_candidate_score",
    "tag_candidate_score",
    "popular_candidate_score",
    "recent_candidate_score",
    "item_interaction_count",
    "item_avg_rating",
    "item_positive_rate",
    "item_popularity_log",
    "item_popularity_score",
    "item_recent_score",
    "item_novelty_score",
    "num_ratings",
    "avg_rating",
    "user_positive_rate",
    "user_rating_stddev",
    "matched_genre_count",
    "candidate_source_count",
    "source_als_candidate",
    "source_content_candidate",
    "source_tag_candidate",
    "source_popular_candidate",
    "source_recent_candidate",
    "als_candidate_rank_reciprocal",
    "content_candidate_rank_reciprocal",
    "tag_candidate_rank_reciprocal",
    "popular_candidate_rank_reciprocal",
    "recent_candidate_rank_reciprocal",
    "best_candidate_rank_reciprocal",
    "source_score_sum",
    "cf_x_content",
    "cf_x_popularity",
    "content_x_popularity",
]


def build_ranking_features(
    candidate_sources_df: DataFrame,
    collaborative_scores_df: DataFrame,
    explicit_scores_df: DataFrame,
    content_scores_df: DataFrame,
    item_features_df: DataFrame,
    user_profiles_df: DataFrame,
) -> DataFrame:
    candidate_selected = candidate_sources_df.select(
        "userId",
        "movieId",
        "candidate_source_count",
        "source_als_candidate",
        "source_content_candidate",
        "source_tag_candidate",
        "source_popular_candidate",
        "source_recent_candidate",
        "als_candidate_score",
        "content_candidate_score",
        "tag_candidate_score",
        "popular_candidate_score",
        "recent_candidate_score",
        "als_candidate_rank",
        "content_candidate_rank",
        "tag_candidate_rank",
        "popular_candidate_rank",
        "recent_candidate_rank",
    )
    content_selected = content_scores_df.select(
        "userId",
        "movieId",
        "content_score",
        "content_genre_score",
        "content_tag_score",
        "matched_genres",
    )
    item_selected = item_features_df.select(
        "movieId",
        "item_interaction_count",
        "item_avg_rating",
        "item_positive_rate",
        "item_popularity_log",
        "item_popularity_score",
        "item_recent_score",
        "item_novelty_score",
    )
    user_selected = user_profiles_df.select(
        "userId",
        "avg_rating",
        "num_ratings",
        "user_positive_rate",
        "user_rating_stddev",
    )
    large_rank = F.lit(1_000_000.0)

    features = (
        candidate_selected.join(
            collaborative_scores_df.select("userId", "movieId", F.col("als_score").alias("cf_score")),
            on=["userId", "movieId"],
            how="left",
        )
        .join(
            explicit_scores_df.select("userId", "movieId", F.col("als_score").alias("explicit_als_score")),
            on=["userId", "movieId"],
            how="left",
        )
        .join(content_selected, on=["userId", "movieId"], how="left")
        .join(item_selected, on="movieId", how="left")
        .join(user_selected, on="userId", how="left")
        .fillna(
            {
                "cf_score": 0.0,
                "explicit_als_score": 0.0,
                "als_candidate_score": 0.0,
                "content_score": 0.0,
                "content_genre_score": 0.0,
                "content_tag_score": 0.0,
                "content_candidate_score": 0.0,
                "tag_candidate_score": 0.0,
                "popular_candidate_score": 0.0,
                "recent_candidate_score": 0.0,
                "item_interaction_count": 0.0,
                "item_avg_rating": 0.0,
                "item_positive_rate": 0.0,
                "item_popularity_log": 0.0,
                "item_popularity_score": 0.0,
                "item_recent_score": 0.0,
                "item_novelty_score": 0.0,
                "avg_rating": 0.0,
                "num_ratings": 0.0,
                "user_positive_rate": 0.0,
                "user_rating_stddev": 0.0,
            }
        )
        .withColumn(
            "matched_genre_count",
            F.when(F.col("matched_genres").isNull(), F.lit(0)).otherwise(F.size(F.col("matched_genres"))),
        )
        .withColumn(
            "als_candidate_rank_reciprocal",
            F.when(F.col("als_candidate_rank").isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / F.col("als_candidate_rank")),
        )
        .withColumn(
            "content_candidate_rank_reciprocal",
            F.when(F.col("content_candidate_rank").isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / F.col("content_candidate_rank")),
        )
        .withColumn(
            "tag_candidate_rank_reciprocal",
            F.when(F.col("tag_candidate_rank").isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / F.col("tag_candidate_rank")),
        )
        .withColumn(
            "popular_candidate_rank_reciprocal",
            F.when(F.col("popular_candidate_rank").isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / F.col("popular_candidate_rank")),
        )
        .withColumn(
            "recent_candidate_rank_reciprocal",
            F.when(F.col("recent_candidate_rank").isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / F.col("recent_candidate_rank")),
        )
        .withColumn(
            "best_candidate_rank_reciprocal",
            F.when(
                F.least(
                    F.coalesce(F.col("als_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("content_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("tag_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("popular_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("recent_candidate_rank").cast("double"), large_rank),
                )
                < large_rank,
                F.lit(1.0)
                / F.least(
                    F.coalesce(F.col("als_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("content_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("tag_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("popular_candidate_rank").cast("double"), large_rank),
                    F.coalesce(F.col("recent_candidate_rank").cast("double"), large_rank),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "source_score_sum",
            F.col("als_candidate_score")
            + F.col("content_candidate_score")
            + F.col("tag_candidate_score")
            + F.col("popular_candidate_score")
            + F.col("recent_candidate_score"),
        )
        .withColumn("cf_x_content", F.col("cf_score") * F.col("content_score"))
        .withColumn("cf_x_popularity", F.col("cf_score") * F.col("item_popularity_score"))
        .withColumn("content_x_popularity", F.col("content_score") * F.col("item_popularity_score"))
    )

    return features


def build_ranker_training_frame(
    feature_df: DataFrame,
    ground_truth_df: DataFrame,
    positive_threshold: float,
) -> DataFrame:
    label_df = (
        ground_truth_df.select("userId", "movieId", "rating")
        .withColumn(
            "label",
            F.when(
                F.col("rating") >= F.lit(positive_threshold),
                (F.floor(((F.col("rating") - F.lit(positive_threshold)) / F.lit(0.5))) + F.lit(1)).cast("int"),
            ).otherwise(F.lit(0)),
        )
        .select("userId", "movieId", "rating", "label")
    )

    training_df = (
        feature_df.join(label_df, on=["userId", "movieId"], how="left")
        .fillna({"rating": 0.0, "label": 0})
        .withColumn("is_positive_label", F.when(F.col("label") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
    )

    eligible_users_df = (
        training_df.groupBy("userId")
        .agg(
            F.max("label").alias("max_label"),
            F.count("*").alias("candidate_count"),
        )
        .filter((F.col("max_label") > F.lit(0)) & (F.col("candidate_count") > F.lit(1)))
        .select("userId")
    )
    return training_df.join(eligible_users_df, on="userId", how="inner")


def _frame_to_pandas(df: DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    ordered_cols = ["userId", "movieId", "label", *feature_cols]
    available_cols = [col for col in ordered_cols if col in df.columns]
    pdf = df.select(*available_cols).toPandas()
    if pdf.empty:
        return pdf

    for col in feature_cols:
        if col not in pdf.columns:
            pdf[col] = 0.0

    pdf["label"] = pdf["label"].astype(int)
    pdf = pdf.sort_values(["userId", "movieId"]).reset_index(drop=True)
    return pdf


def _group_sizes(pdf: pd.DataFrame) -> list[int]:
    return pdf.groupby("userId", sort=False).size().astype(int).tolist()


def _evaluate_ranked_frame(pdf: pd.DataFrame, score_col: str, top_k: int) -> Dict[str, float]:
    if pdf.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    for _, user_df in pdf.groupby("userId", sort=False):
        ranked = user_df.sort_values([score_col, "movieId"], ascending=[False, True]).head(top_k)
        relevant = user_df[user_df["label"] > 0]
        if relevant.empty:
            continue

        hit_count = float((ranked["label"] > 0).sum())
        precision_scores.append(hit_count / float(top_k))
        recall_scores.append(hit_count / float(len(relevant)))

        dcg = 0.0
        for rank_idx, label in enumerate(ranked["label"].tolist(), start=1):
            if label <= 0:
                continue
            dcg += float(label) / float(math.log2(rank_idx + 1))

        ideal_labels = sorted(relevant["label"].tolist(), reverse=True)[:top_k]
        idcg = 0.0
        for rank_idx, label in enumerate(ideal_labels, start=1):
            idcg += float(label) / float(math.log2(rank_idx + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    if not precision_scores:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    return {
        "precision_at_k": float(sum(precision_scores) / len(precision_scores)),
        "recall_at_k": float(sum(recall_scores) / len(recall_scores)),
        "ndcg_at_k": float(sum(ndcg_scores) / len(ndcg_scores)),
    }


def train_xgb_ranker(
    ranking_train_df: DataFrame,
    settings: PipelineSettings | None = None,
) -> tuple[Any, list[str], Dict[str, float | int | str]]:
    cfg = settings or PipelineSettings()
    ranker_cfg: RankerSettings = cfg.ranker

    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise RuntimeError("xgboost is required for the ranking model. Install requirements.txt before running.") from exc

    feature_cols = [col for col in RANKER_FEATURE_COLUMNS if col in ranking_train_df.columns]
    pdf = _frame_to_pandas(ranking_train_df, feature_cols)
    if pdf.empty:
        raise RuntimeError("Ranking training frame is empty. Candidate generation produced no trainable rows.")

    unique_users = sorted(pdf["userId"].unique().tolist())
    holdout_users = [user_id for user_id in unique_users if int(user_id) % ranker_cfg.holdout_user_modulo == 0]
    train_users = [user_id for user_id in unique_users if user_id not in holdout_users]

    if len(train_users) < ranker_cfg.min_training_groups or len(holdout_users) < max(5, ranker_cfg.min_training_groups // 5):
        holdout_users = []
        train_users = unique_users

    train_pdf = pdf[pdf["userId"].isin(train_users)].reset_index(drop=True)
    eval_pdf = pdf[pdf["userId"].isin(holdout_users)].reset_index(drop=True)

    param_grid = product(
        ranker_cfg.n_estimators_candidates,
        ranker_cfg.max_depth_candidates,
        ranker_cfg.learning_rate_candidates,
        ranker_cfg.min_child_weight_candidates,
    )

    best_model: Any | None = None
    best_info: Dict[str, float | int | str] = {
        "ranker_model": "xgboost",
        "ranker_holdout_groups": int(len(holdout_users)),
        "ranker_train_groups": int(len(train_users)),
        "ranker_val_precision_at_k": 0.0,
        "ranker_val_recall_at_k": 0.0,
        "ranker_val_ndcg_at_k": -1.0,
    }

    X_train = train_pdf[feature_cols]
    y_train = train_pdf["label"]
    group_train = _group_sizes(train_pdf)

    if eval_pdf.empty:
        default_params = {
            "n_estimators": ranker_cfg.n_estimators_candidates[0],
            "max_depth": ranker_cfg.max_depth_candidates[0],
            "learning_rate": ranker_cfg.learning_rate_candidates[0],
            "min_child_weight": ranker_cfg.min_child_weight_candidates[0],
        }
        best_model = XGBRanker(
            objective=ranker_cfg.objective,
            eval_metric=f"ndcg@{ranker_cfg.eval_at_k}",
            tree_method="hist",
            random_state=ranker_cfg.random_state,
            subsample=ranker_cfg.subsample,
            colsample_bytree=ranker_cfg.colsample_bytree,
            reg_lambda=ranker_cfg.reg_lambda,
            **default_params,
        )
        best_model.fit(X_train, y_train, group=group_train, verbose=False)
        best_info.update(
            {
                "ranker_n_estimators": int(default_params["n_estimators"]),
                "ranker_max_depth": int(default_params["max_depth"]),
                "ranker_learning_rate": float(default_params["learning_rate"]),
                "ranker_min_child_weight": float(default_params["min_child_weight"]),
                "ranker_val_ndcg_at_k": 0.0,
            }
        )
    else:
        X_eval = eval_pdf[feature_cols]
        y_eval = eval_pdf["label"]
        group_eval = _group_sizes(eval_pdf)

        for n_estimators, max_depth, learning_rate, min_child_weight in param_grid:
            model = XGBRanker(
                objective=ranker_cfg.objective,
                eval_metric=f"ndcg@{ranker_cfg.eval_at_k}",
                tree_method="hist",
                random_state=ranker_cfg.random_state,
                subsample=ranker_cfg.subsample,
                colsample_bytree=ranker_cfg.colsample_bytree,
                reg_lambda=ranker_cfg.reg_lambda,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
            )
            model.fit(X_train, y_train, group=group_train, verbose=False)

            eval_scored = eval_pdf.copy()
            eval_scored["pred_score"] = model.predict(X_eval)
            metrics = _evaluate_ranked_frame(eval_scored, "pred_score", ranker_cfg.eval_at_k)

            is_better = (metrics["ndcg_at_k"] > float(best_info["ranker_val_ndcg_at_k"])) or (
                metrics["ndcg_at_k"] == float(best_info["ranker_val_ndcg_at_k"])
                and metrics["precision_at_k"] > float(best_info["ranker_val_precision_at_k"])
            )
            if is_better:
                best_model = model
                best_info.update(
                    {
                        "ranker_n_estimators": int(n_estimators),
                        "ranker_max_depth": int(max_depth),
                        "ranker_learning_rate": float(learning_rate),
                        "ranker_min_child_weight": float(min_child_weight),
                        "ranker_val_precision_at_k": float(metrics["precision_at_k"]),
                        "ranker_val_recall_at_k": float(metrics["recall_at_k"]),
                        "ranker_val_ndcg_at_k": float(metrics["ndcg_at_k"]),
                    }
                )

        if best_model is None:
            raise RuntimeError("XGBoost ranking tuning failed to produce a model.")

        best_model = XGBRanker(
            objective=ranker_cfg.objective,
            eval_metric=f"ndcg@{ranker_cfg.eval_at_k}",
            tree_method="hist",
            random_state=ranker_cfg.random_state,
            subsample=ranker_cfg.subsample,
            colsample_bytree=ranker_cfg.colsample_bytree,
            reg_lambda=ranker_cfg.reg_lambda,
            n_estimators=int(best_info["ranker_n_estimators"]),
            max_depth=int(best_info["ranker_max_depth"]),
            learning_rate=float(best_info["ranker_learning_rate"]),
            min_child_weight=float(best_info["ranker_min_child_weight"]),
        )
        best_model.fit(pdf[feature_cols], pdf["label"], group=_group_sizes(pdf), verbose=False)

    best_info["ranker_feature_count"] = int(len(feature_cols))
    return best_model, feature_cols, best_info


def score_candidates_with_ranker(
    spark: SparkSession,
    candidate_feature_df: DataFrame,
    ranker_model: Any,
    feature_cols: Sequence[str],
) -> DataFrame:
    available_cols = [col for col in feature_cols if col in candidate_feature_df.columns]
    candidate_pdf = candidate_feature_df.select("userId", "movieId", *available_cols).toPandas()
    if candidate_pdf.empty:
        return spark.createDataFrame([], "userId int, movieId int, final_score double")

    score_input = candidate_pdf[available_cols].copy()
    candidate_pdf["final_score"] = ranker_model.predict(score_input)
    predictions_pdf = candidate_pdf[["userId", "movieId", "final_score"]]
    return spark.createDataFrame(predictions_pdf)
