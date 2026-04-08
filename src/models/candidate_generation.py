from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def build_item_ranking_features(
    interactions_df: DataFrame,
    positive_threshold: float = 4.0,
) -> DataFrame:
    latest_ts = interactions_df.agg(F.max("timestamp").alias("latest_ts")).collect()[0]["latest_ts"]
    latest_ts = float(latest_ts or 0.0)

    item_stats = interactions_df.groupBy("movieId").agg(
        F.count("*").alias("item_interaction_count"),
        F.avg("rating").alias("item_avg_rating"),
        F.avg(F.when(F.col("rating") >= F.lit(positive_threshold), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
            "item_positive_rate"
        ),
        F.max("timestamp").alias("item_last_interaction_ts"),
        F.min("timestamp").alias("item_first_interaction_ts"),
    )

    return (
        item_stats.withColumn("item_popularity_log", F.log1p(F.col("item_interaction_count")))
        .withColumn(
            "item_recency_days",
            F.greatest(F.lit(latest_ts) - F.col("item_last_interaction_ts"), F.lit(0.0)) / F.lit(86400.0),
        )
        .withColumn("item_recent_score", F.lit(1.0) / (F.lit(1.0) + F.col("item_recency_days")))
        .withColumn(
            "item_novelty_score",
            F.lit(1.0) / F.log1p(F.col("item_interaction_count") + F.lit(1.0)),
        )
        .withColumn(
            "item_popularity_score",
            F.col("item_popularity_log") * (F.lit(0.5) + F.col("item_positive_rate")),
        )
    )


def generate_popular_candidates(
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    item_features_df: DataFrame,
    seen_interactions_df: DataFrame,
    k: int = 100,
) -> DataFrame:
    positive_user_genres = user_genre_profiles_df.filter(F.col("genre_weight") > F.lit(0.0)).select(
        "userId",
        "genre",
        "genre_weight",
    )
    movie_popularity_df = item_features_df.select("movieId", "item_popularity_score")

    scored = (
        positive_user_genres.join(
            movie_genre_weights_df.select("movieId", "genre", "movie_genre_weight"),
            on="genre",
            how="inner",
        )
        .join(movie_popularity_df, on="movieId", how="inner")
        .withColumn(
            "popular_candidate_score",
            F.col("genre_weight") * F.col("movie_genre_weight") * F.col("item_popularity_score"),
        )
    )

    aggregated = scored.groupBy("userId", "movieId").agg(
        F.sum("popular_candidate_score").alias("popular_candidate_score")
    )
    unseen = aggregated.join(
        seen_interactions_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]),
        on=["userId", "movieId"],
        how="left_anti",
    )

    ranking_window = Window.partitionBy("userId").orderBy(
        F.col("popular_candidate_score").desc(),
        F.col("movieId").asc(),
    )
    return (
        unseen.withColumn("popular_rank", F.row_number().over(ranking_window))
        .filter(F.col("popular_rank") <= F.lit(k))
        .select("userId", "movieId", "popular_candidate_score")
    )


def generate_recent_candidates(
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    item_features_df: DataFrame,
    seen_interactions_df: DataFrame,
    k: int = 100,
) -> DataFrame:
    positive_user_genres = user_genre_profiles_df.filter(F.col("genre_weight") > F.lit(0.0)).select(
        "userId",
        "genre",
        "genre_weight",
    )
    movie_recent_df = item_features_df.select("movieId", "item_recent_score", "item_popularity_log")

    scored = (
        positive_user_genres.join(
            movie_genre_weights_df.select("movieId", "genre", "movie_genre_weight"),
            on="genre",
            how="inner",
        )
        .join(movie_recent_df, on="movieId", how="inner")
        .withColumn(
            "recent_candidate_score",
            F.col("genre_weight")
            * F.col("movie_genre_weight")
            * F.col("item_recent_score")
            * (F.lit(1.0) + F.col("item_popularity_log")),
        )
    )

    aggregated = scored.groupBy("userId", "movieId").agg(
        F.sum("recent_candidate_score").alias("recent_candidate_score")
    )
    unseen = aggregated.join(
        seen_interactions_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]),
        on=["userId", "movieId"],
        how="left_anti",
    )

    ranking_window = Window.partitionBy("userId").orderBy(
        F.col("recent_candidate_score").desc(),
        F.col("movieId").asc(),
    )
    return (
        unseen.withColumn("recent_rank", F.row_number().over(ranking_window))
        .filter(F.col("recent_rank") <= F.lit(k))
        .select("userId", "movieId", "recent_candidate_score")
    )


def merge_candidate_sources(
    als_candidates_df: DataFrame,
    content_candidates_df: DataFrame,
    tag_candidates_df: DataFrame,
    popular_candidates_df: DataFrame,
    recent_candidates_df: DataFrame,
) -> DataFrame:
    def _flag(df: DataFrame, score_col: str, flag_col: str) -> DataFrame:
        return df.select("userId", "movieId", F.col(score_col), F.lit(1).alias(flag_col))

    flagged_sources = [
        _flag(als_candidates_df, "als_score", "source_als_candidate"),
        _flag(content_candidates_df, "content_score", "source_content_candidate"),
        _flag(tag_candidates_df, "content_tag_score", "source_tag_candidate"),
        _flag(popular_candidates_df, "popular_candidate_score", "source_popular_candidate"),
        _flag(recent_candidates_df, "recent_candidate_score", "source_recent_candidate"),
    ]

    base = flagged_sources[0]
    for source_df in flagged_sources[1:]:
        base = base.unionByName(source_df, allowMissingColumns=True)

    return (
        base.groupBy("userId", "movieId")
        .agg(
            F.max(F.coalesce(F.col("source_als_candidate"), F.lit(0))).alias("source_als_candidate"),
            F.max(F.coalesce(F.col("source_content_candidate"), F.lit(0))).alias("source_content_candidate"),
            F.max(F.coalesce(F.col("source_tag_candidate"), F.lit(0))).alias("source_tag_candidate"),
            F.max(F.coalesce(F.col("source_popular_candidate"), F.lit(0))).alias("source_popular_candidate"),
            F.max(F.coalesce(F.col("source_recent_candidate"), F.lit(0))).alias("source_recent_candidate"),
        )
        .withColumn(
            "candidate_source_count",
            F.col("source_als_candidate")
            + F.col("source_content_candidate")
            + F.col("source_tag_candidate")
            + F.col("source_popular_candidate")
            + F.col("source_recent_candidate"),
        )
    )
