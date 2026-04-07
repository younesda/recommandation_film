from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def create_user_profiles(ratings_df: DataFrame) -> DataFrame:
    profiles = ratings_df.groupBy("userId").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("*").alias("num_ratings"),
        F.min("timestamp").alias("first_rating_ts"),
        F.max("timestamp").alias("last_rating_ts"),
    )
    LOGGER.info("Built user profiles")
    return profiles


def build_user_genre_profiles(
    ratings_df: DataFrame,
    movie_genres_df: DataFrame,
) -> DataFrame:
    joined = ratings_df.join(movie_genres_df.select("movieId", "genre"), on="movieId", how="inner")
    scored = joined.withColumn("genre_signal", F.greatest(F.col("rating") - F.lit(2.5), F.lit(0.0)))

    aggregated = scored.groupBy("userId", "genre").agg(
        F.avg("genre_signal").alias("avg_genre_signal"),
        F.count("*").alias("genre_events"),
    )

    normalization_window = Window.partitionBy("userId")
    normalized = (
        aggregated.withColumn("signal_sum", F.sum("avg_genre_signal").over(normalization_window))
        .withColumn(
            "genre_weight",
            F.when(F.col("signal_sum") > F.lit(0.0), F.col("avg_genre_signal") / F.col("signal_sum")).otherwise(F.lit(0.0)),
        )
        .drop("signal_sum")
    )
    LOGGER.info("Built user genre profiles")
    return normalized


def build_user_tag_profiles(tags_df: DataFrame, min_tag_count: int = 2) -> DataFrame:
    tag_counts = tags_df.groupBy("userId", "tag").agg(F.count("*").alias("tag_count"))
    filtered = tag_counts.filter(F.col("tag_count") >= F.lit(min_tag_count))

    normalization_window = Window.partitionBy("userId")
    weighted = (
        filtered.withColumn("user_tag_total", F.sum("tag_count").over(normalization_window))
        .withColumn("user_tag_weight", F.col("tag_count") / F.col("user_tag_total"))
        .drop("user_tag_total")
    )
    LOGGER.info("Built user tag profiles")
    return weighted
