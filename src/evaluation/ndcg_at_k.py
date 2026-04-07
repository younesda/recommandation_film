from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _top_k_ranked_predictions(recommendations_df: DataFrame, k: int) -> DataFrame:
    if "rank" in recommendations_df.columns:
        return (
            recommendations_df.filter(F.col("rank") <= F.lit(k))
            .select("userId", "movieId", "rank")
            .dropDuplicates(["userId", "movieId"])
        )

    ranking_window = Window.partitionBy("userId").orderBy(F.col("final_score").desc(), F.col("movieId").asc())
    ranked = recommendations_df.withColumn("rank", F.row_number().over(ranking_window))
    return ranked.filter(F.col("rank") <= F.lit(k)).select("userId", "movieId", "rank")


def compute_ndcg_at_k(
    recommendations_df: DataFrame,
    ground_truth_df: DataFrame,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> float:
    if k <= 0:
        raise ValueError("k must be > 0 for NDCG@K")

    predicted_top_k = _top_k_ranked_predictions(recommendations_df, k)
    relevant_items = (
        ground_truth_df.filter(F.col("rating") >= F.lit(positive_threshold))
        .select("userId", "movieId")
        .dropDuplicates(["userId", "movieId"])
        .withColumn("relevance", F.lit(1.0))
    )

    with_relevance = predicted_top_k.join(relevant_items, on=["userId", "movieId"], how="left").fillna({"relevance": 0.0})
    dcg_by_user = with_relevance.withColumn(
        "dcg_component",
        F.col("relevance") / F.log2(F.col("rank") + F.lit(1.0)),
    ).groupBy("userId").agg(F.sum("dcg_component").alias("dcg"))

    relevant_count = relevant_items.groupBy("userId").agg(F.count("*").alias("relevant_count"))
    idcg_by_user = (
        relevant_count.withColumn("max_rank", F.least(F.col("relevant_count"), F.lit(k)))
        .withColumn("ideal_ranks", F.sequence(F.lit(1), F.col("max_rank")))
        .withColumn(
            "idcg",
            F.expr("aggregate(transform(ideal_ranks, r -> 1D / log2(r + 1D)), 0D, (acc, x) -> acc + x)"),
        )
        .select("userId", "idcg")
    )

    ndcg_by_user = (
        idcg_by_user.join(dcg_by_user, on="userId", how="left")
        .fillna({"dcg": 0.0})
        .withColumn(
            "ndcg_at_k",
            F.when(F.col("idcg") > F.lit(0.0), F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)),
        )
    )

    result = ndcg_by_user.agg(F.avg("ndcg_at_k").alias("mean_ndcg")).collect()[0]["mean_ndcg"]
    return float(result) if result is not None else 0.0
