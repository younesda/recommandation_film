from __future__ import annotations

from typing import Dict, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.settings import PipelineSettings
from src.evaluation.ndcg_at_k import compute_ndcg_at_k
from src.evaluation.precision_at_k import compute_precision_at_k
from src.evaluation.recall_at_k import compute_recall_at_k
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


def _build_ranking_als(rank: int, reg_param: float, alpha: float, max_iter: int, seed: int) -> ALS:
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="preference",
        rank=rank,
        regParam=reg_param,
        alpha=alpha,
        maxIter=max_iter,
        implicitPrefs=True,
        nonnegative=True,
        coldStartStrategy="drop",
        seed=seed,
    )


def prepare_implicit_feedback(
    ratings_df: DataFrame,
    positive_threshold: float = 4.0,
) -> DataFrame:
    implicit_df = (
        ratings_df.filter(F.col("rating") >= F.lit(positive_threshold))
        .withColumn(
            "preference",
            (F.floor(((F.col("rating") - F.lit(positive_threshold)) / F.lit(0.5))) + F.lit(1)).cast("double"),
        )
        .select("userId", "movieId", "preference")
    )
    return implicit_df.dropDuplicates(["userId", "movieId"])


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


def train_ranking_als_with_tuning(
    train_df: DataFrame,
    val_df: DataFrame,
    settings: PipelineSettings | None = None,
) -> Tuple[ALSModel, Dict[str, float]]:
    cfg = settings or PipelineSettings()
    implicit_train_df = prepare_implicit_feedback(train_df, positive_threshold=cfg.min_positive_rating).cache()
    if len(implicit_train_df.head(1)) == 0:
        raise RuntimeError("Implicit ALS tuning failed: no positive interactions found in training data.")

    val_users_df = val_df.select("userId").distinct().cache()
    seen_train_df = train_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).cache()
    ranking_window = Window.partitionBy("userId").orderBy(F.col("als_score").desc(), F.col("movieId").asc())

    best_model: ALSModel | None = None
    best_params: Dict[str, float] = {}
    best_ndcg = -1.0
    best_precision = -1.0

    LOGGER.info("Starting implicit ALS tuning for ranking")
    for rank in cfg.als.ranking_rank_candidates:
        for reg_param in cfg.als.ranking_reg_param_candidates:
            for alpha in cfg.als.ranking_alpha_candidates:
                for max_iter in cfg.als.ranking_max_iter_candidates:
                    model = _build_ranking_als(rank, reg_param, alpha, max_iter, cfg.als.seed).fit(implicit_train_df)
                    recommendations = recommend_for_users_flat(model, val_users_df, cfg.hybrid.top_k)
                    recommendations = recommendations.join(seen_train_df, on=["userId", "movieId"], how="left_anti")
                    recommendations = (
                        recommendations.withColumn("rank", F.row_number().over(ranking_window))
                        .filter(F.col("rank") <= F.lit(cfg.hybrid.top_k))
                        .cache()
                    )

                    if len(recommendations.head(1)) == 0:
                        LOGGER.warning(
                            "Skipping implicit ALS config rank=%s regParam=%s alpha=%s maxIter=%s because recommendations are empty",
                            rank,
                            reg_param,
                            alpha,
                            max_iter,
                        )
                        continue

                    precision_val = compute_precision_at_k(
                        recommendations_df=recommendations.select(
                            "userId",
                            "movieId",
                            "rank",
                            F.col("als_score").alias("final_score"),
                        ),
                        ground_truth_df=val_df,
                        k=cfg.hybrid.top_k,
                        positive_threshold=cfg.min_positive_rating,
                    )
                    recall_val = compute_recall_at_k(
                        recommendations_df=recommendations.select(
                            "userId",
                            "movieId",
                            "rank",
                            F.col("als_score").alias("final_score"),
                        ),
                        ground_truth_df=val_df,
                        k=cfg.hybrid.top_k,
                        positive_threshold=cfg.min_positive_rating,
                    )
                    ndcg_val = compute_ndcg_at_k(
                        recommendations_df=recommendations.select(
                            "userId",
                            "movieId",
                            "rank",
                            F.col("als_score").alias("final_score"),
                        ),
                        ground_truth_df=val_df,
                        k=cfg.hybrid.top_k,
                        positive_threshold=cfg.min_positive_rating,
                    )

                    LOGGER.info(
                        "Implicit ALS candidate rank=%s regParam=%s alpha=%s maxIter=%s precision=%.6f recall=%.6f ndcg=%.6f",
                        rank,
                        reg_param,
                        alpha,
                        max_iter,
                        precision_val,
                        recall_val,
                        ndcg_val,
                    )

                    is_better = (ndcg_val > best_ndcg) or (ndcg_val == best_ndcg and precision_val > best_precision)
                    if is_better:
                        best_ndcg = float(ndcg_val)
                        best_precision = float(precision_val)
                        best_model = model
                        best_params = {
                            "rank": float(rank),
                            "regParam": float(reg_param),
                            "alpha": float(alpha),
                            "maxIter": float(max_iter),
                            "val_precision": float(precision_val),
                            "val_recall": float(recall_val),
                            "val_ndcg": float(ndcg_val),
                        }

    if best_model is None:
        raise RuntimeError("Implicit ALS tuning failed: no valid ranking model produced recommendations.")

    LOGGER.info("Best implicit ALS params=%s", best_params)
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


def retrain_best_ranking_als(
    train_df: DataFrame,
    val_df: DataFrame,
    best_params: Dict[str, float],
    settings: PipelineSettings | None = None,
) -> ALSModel:
    cfg = settings or PipelineSettings()
    combined_train = train_df.unionByName(val_df)
    implicit_combined_df = prepare_implicit_feedback(combined_train, positive_threshold=cfg.min_positive_rating)
    model = _build_ranking_als(
        rank=int(best_params["rank"]),
        reg_param=float(best_params["regParam"]),
        alpha=float(best_params["alpha"]),
        max_iter=int(best_params["maxIter"]),
        seed=cfg.als.seed,
    ).fit(implicit_combined_df)
    LOGGER.info("Retrained implicit ALS model on train+val")
    return model
