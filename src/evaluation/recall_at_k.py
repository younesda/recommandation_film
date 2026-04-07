from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _top_k_predictions(recommendations_df: DataFrame, k: int) -> DataFrame:
    if "rank" in recommendations_df.columns:
        return recommendations_df.filter(F.col("rank") <= F.lit(k)).select("userId", "movieId").dropDuplicates(
            ["userId", "movieId"]
        )

    ranking_window = Window.partitionBy("userId").orderBy(F.col("final_score").desc(), F.col("movieId").asc())
    ranked = recommendations_df.withColumn("rank", F.row_number().over(ranking_window))
    return ranked.filter(F.col("rank") <= F.lit(k)).select("userId", "movieId").dropDuplicates(["userId", "movieId"])


def compute_recall_at_k(
    recommendations_df: DataFrame,
    ground_truth_df: DataFrame,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> float:
    if k <= 0:
        raise ValueError("k must be > 0 for Recall@K")

    predicted_top_k = _top_k_predictions(recommendations_df, k)
    relevant_items = (
        ground_truth_df.filter(F.col("rating") >= F.lit(positive_threshold))
        .select("userId", "movieId")
        .dropDuplicates(["userId", "movieId"])
    )

    relevant_count = relevant_items.groupBy("userId").agg(F.count("*").alias("relevant_count"))
    hits = predicted_top_k.join(relevant_items, on=["userId", "movieId"], how="inner")
    hit_count = hits.groupBy("userId").agg(F.count("*").alias("hit_count"))

    recall_by_user = (
        relevant_count.join(hit_count, on="userId", how="left")
        .fillna({"hit_count": 0})
        .withColumn(
            "recall_at_k",
            F.when(F.col("relevant_count") > F.lit(0), F.col("hit_count") / F.col("relevant_count")).otherwise(F.lit(0.0)),
        )
    )

    result = recall_by_user.agg(F.avg("recall_at_k").alias("mean_recall")).collect()[0]["mean_recall"]
    return float(result) if result is not None else 0.0
