from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _is_empty(df: DataFrame) -> bool:
    return len(df.head(1)) == 0


def clean_movies(movies_df: DataFrame) -> DataFrame:
    cleaned = (
        movies_df.select("movieId", "title", "genres")
        .dropna(subset=["movieId", "title"])
        .withColumn(
            "genres",
            F.when(F.col("genres").isNull() | (F.col("genres") == ""), F.lit("(no genres listed)")).otherwise(
                F.col("genres")
            ),
        )
        .withColumn("movieId", F.col("movieId").cast("int"))
        .dropDuplicates(["movieId"])
    )
    LOGGER.info("Movies cleaned")
    return cleaned


def clean_tags(tags_df: DataFrame) -> DataFrame:
    cleaned = (
        tags_df.select("userId", "movieId", "tag", "timestamp")
        .dropna(subset=["userId", "movieId", "tag", "timestamp"])
        .withColumn("userId", F.col("userId").cast("int"))
        .withColumn("movieId", F.col("movieId").cast("int"))
        .withColumn("timestamp", F.col("timestamp").cast("long"))
        .withColumn("tag", F.trim(F.lower(F.col("tag"))))
        .filter(F.col("tag") != "")
    )
    LOGGER.info("Tags cleaned")
    return cleaned


def clean_ratings(
    ratings_df: DataFrame,
    settings: PipelineSettings | None = None,
) -> DataFrame:
    cfg = settings or PipelineSettings()
    LOGGER.info("Cleaning ratings data")

    base = (
        ratings_df.select("userId", "movieId", "rating", "timestamp")
        .dropna(subset=["userId", "movieId", "rating", "timestamp"])
        .withColumn("userId", F.col("userId").cast("int"))
        .withColumn("movieId", F.col("movieId").cast("int"))
        .withColumn("rating", F.col("rating").cast("double"))
        .withColumn("timestamp", F.col("timestamp").cast("long"))
        .filter((F.col("rating") >= F.lit(0.5)) & (F.col("rating") <= F.lit(5.0)))
    )

    if _is_empty(base):
        raise ValueError("No valid ratings available after null/range filtering.")

    q1, q3 = base.approxQuantile("rating", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower_bound = max(0.5, q1 - 1.5 * iqr)
    upper_bound = min(5.0, q3 + 1.5 * iqr)
    bounded = base.filter((F.col("rating") >= F.lit(lower_bound)) & (F.col("rating") <= F.lit(upper_bound)))

    user_activity = bounded.groupBy("userId").agg(F.count("*").alias("user_interactions"))
    active_users = user_activity.filter(F.col("user_interactions") >= F.lit(cfg.min_user_interactions)).select("userId")

    item_activity = bounded.groupBy("movieId").agg(F.count("*").alias("item_interactions"))
    active_items = item_activity.filter(F.col("item_interactions") >= F.lit(cfg.min_item_interactions)).select("movieId")

    cleaned = (
        bounded.join(active_users, on="userId", how="inner")
        .join(active_items, on="movieId", how="inner")
        .dropDuplicates(["userId", "movieId", "timestamp"])
    )

    if _is_empty(cleaned):
        raise ValueError("Ratings became empty after activity filters. Lower min interaction thresholds.")

    LOGGER.info("Ratings cleaned bounds=(%.2f, %.2f)", lower_bound, upper_bound)
    return cleaned


def time_based_split(
    ratings_df: DataFrame,
    settings: PipelineSettings | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    cfg = settings or PipelineSettings()
    train_cutoff = cfg.train_ratio
    val_cutoff = cfg.train_ratio + cfg.val_ratio

    if abs((cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    LOGGER.info(
        "Performing time split train=%.2f val=%.2f test=%.2f",
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
    )

    window = Window.partitionBy("userId").orderBy(F.col("timestamp").asc())
    ranked = ratings_df.withColumn("user_event_percentile", F.percent_rank().over(window))

    train_df = ranked.filter(F.col("user_event_percentile") < F.lit(train_cutoff)).drop("user_event_percentile")
    val_df = ranked.filter(
        (F.col("user_event_percentile") >= F.lit(train_cutoff)) & (F.col("user_event_percentile") < F.lit(val_cutoff))
    ).drop("user_event_percentile")
    test_df = ranked.filter(F.col("user_event_percentile") >= F.lit(val_cutoff)).drop("user_event_percentile")

    if _is_empty(train_df) or _is_empty(val_df) or _is_empty(test_df):
        raise ValueError("Strict time split failed: one split is empty. Adjust split ratios or filtering thresholds.")

    train_last = train_df.groupBy("userId").agg(F.max("timestamp").alias("train_last_ts"))
    val_first = val_df.groupBy("userId").agg(F.min("timestamp").alias("val_first_ts"))
    val_last = val_df.groupBy("userId").agg(F.max("timestamp").alias("val_last_ts"))
    test_first = test_df.groupBy("userId").agg(F.min("timestamp").alias("test_first_ts"))

    leakage_train_val = (
        train_last.join(val_first, on="userId", how="inner")
        .filter(F.col("train_last_ts") > F.col("val_first_ts"))
        .limit(1)
        .count()
    )
    leakage_val_test = (
        val_last.join(test_first, on="userId", how="inner")
        .filter(F.col("val_last_ts") > F.col("test_first_ts"))
        .limit(1)
        .count()
    )
    if leakage_train_val > 0 or leakage_val_test > 0:
        raise ValueError("Time split validation failed: temporal leakage detected between splits.")

    LOGGER.info("Time split completed with strict temporal ordering")
    return train_df, val_df, test_df
