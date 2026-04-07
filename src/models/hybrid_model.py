from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.settings import PipelineSettings


def combine_hybrid_scores(
    als_scores_df: DataFrame,
    content_scores_df: DataFrame,
    settings: PipelineSettings | None = None,
    als_weight: float | None = None,
    content_weight: float | None = None,
) -> DataFrame:
    cfg = settings or PipelineSettings()
    resolved_als_weight = cfg.hybrid.als_weight if als_weight is None else als_weight
    resolved_content_weight = cfg.hybrid.content_weight if content_weight is None else content_weight

    als_normalized = als_scores_df.withColumn("als_score", F.coalesce(F.col("als_score"), F.lit(0.0)))
    als_normalized = als_normalized.withColumn(
        "als_score_norm",
        F.when(F.col("als_score") > F.lit(0.0), F.col("als_score") / F.lit(5.0)).otherwise(F.lit(0.0)),
    )

    joined = als_normalized.join(
        content_scores_df.select("userId", "movieId", "content_score", "matched_genres"),
        on=["userId", "movieId"],
        how="left",
    ).fillna({"content_score": 0.0})

    scored = joined.withColumn(
        "final_score",
        (F.lit(resolved_als_weight) * F.col("als_score_norm"))
        + (F.lit(resolved_content_weight) * F.col("content_score")),
    )
    return scored


def select_top_k_recommendations(
    hybrid_scores_df: DataFrame,
    k: int,
) -> DataFrame:
    ranking_window = Window.partitionBy("userId").orderBy(F.col("final_score").desc(), F.col("movieId").asc())
    ranked = hybrid_scores_df.withColumn("rank", F.row_number().over(ranking_window)).filter(F.col("rank") <= F.lit(k))

    with_reason = ranked.withColumn(
        "explanation",
        F.when(
            F.col("matched_genres").isNotNull() & (F.size(F.col("matched_genres")) > F.lit(0)),
            F.concat(
                F.lit("Matched your preferred genres: "),
                F.array_join(F.expr("slice(matched_genres, 1, 3)"), F.lit(", ")),
            ),
        ).otherwise(F.lit("Recommended from collaborative behavior similar to users with close tastes.")),
    )
    return with_reason
