from __future__ import annotations

from typing import Dict, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _build_als(rank: int, reg_param: float, max_iter: int, seed: int) -> ALS:
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        implicitPrefs=False,
        nonnegative=True,
        coldStartStrategy="drop",
        seed=seed,
    )


def train_als_with_tuning(
    train_df: DataFrame,
    val_df: DataFrame,
    settings: PipelineSettings | None = None,
) -> Tuple[ALSModel, Dict[str, float]]:
    cfg = settings or PipelineSettings()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    best_model: ALSModel | None = None
    best_params: Dict[str, float] = {}
    best_rmse = float("inf")

    LOGGER.info("Starting ALS tuning")
    for rank in cfg.als.rank_candidates:
        for reg_param in cfg.als.reg_param_candidates:
            for max_iter in cfg.als.max_iter_candidates:
                model = _build_als(rank, reg_param, max_iter, cfg.als.seed).fit(train_df)
                predictions = model.transform(val_df).dropna(subset=["prediction"])

                if len(predictions.head(1)) == 0:
                    LOGGER.warning(
                        "Skipping ALS config rank=%s regParam=%s maxIter=%s because validation predictions are empty",
                        rank,
                        reg_param,
                        max_iter,
                    )
                    continue

                rmse = evaluator.evaluate(predictions)
                LOGGER.info(
                    "ALS candidate rank=%s regParam=%s maxIter=%s rmse=%.6f",
                    rank,
                    reg_param,
                    max_iter,
                    rmse,
                )

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {
                        "rank": float(rank),
                        "regParam": float(reg_param),
                        "maxIter": float(max_iter),
                        "val_rmse": float(rmse),
                    }

    if best_model is None:
        raise RuntimeError("ALS tuning failed: no valid model produced predictions.")

    LOGGER.info("Best ALS params=%s", best_params)
    return best_model, best_params


def score_als(model: ALSModel, df: DataFrame) -> DataFrame:
    return model.transform(df).dropna(subset=["prediction"])


def score_als_candidates(model: ALSModel, candidate_df: DataFrame) -> DataFrame:
    base_candidates = candidate_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"])
    predictions = model.transform(base_candidates)
    return (
        base_candidates.join(
            predictions.select("userId", "movieId", F.col("prediction").alias("als_score")),
            on=["userId", "movieId"],
            how="left",
        )
        .fillna({"als_score": 0.0})
    )


def recommend_for_all_users_flat(model: ALSModel, k: int) -> DataFrame:
    return (
        model.recommendForAllUsers(k)
        .withColumn("recommendation", F.explode("recommendations"))
        .select(
            F.col("userId"),
            F.col("recommendation.movieId").alias("movieId"),
            F.col("recommendation.rating").alias("als_score"),
        )
    )


def recommend_for_users_flat(model: ALSModel, users_df: DataFrame, k: int) -> DataFrame:
    return (
        model.recommendForUserSubset(users_df, k)
        .withColumn("recommendation", F.explode("recommendations"))
        .select(
            F.col("userId"),
            F.col("recommendation.movieId").alias("movieId"),
            F.col("recommendation.rating").alias("als_score"),
        )
    )


def retrain_best_als(
    train_df: DataFrame,
    val_df: DataFrame,
    best_params: Dict[str, float],
    settings: PipelineSettings | None = None,
) -> ALSModel:
    cfg = settings or PipelineSettings()
    combined_train = train_df.unionByName(val_df)
    model = _build_als(
        rank=int(best_params["rank"]),
        reg_param=float(best_params["regParam"]),
        max_iter=int(best_params["maxIter"]),
        seed=cfg.als.seed,
    ).fit(combined_train)
    LOGGER.info("Retrained ALS model on train+val")
    return model
