from __future__ import annotations

from typing import Tuple

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def add_time_features(ratings_df: DataFrame) -> DataFrame:
    enriched = (
        ratings_df.withColumn("event_ts", F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
        .withColumn("event_year", F.year(F.col("event_ts")))
        .withColumn("event_month", F.month(F.col("event_ts")))
        .withColumn("event_day_of_week", F.dayofweek(F.col("event_ts")))
        .withColumn("event_hour", F.hour(F.col("event_ts")))
    )
    LOGGER.info("Added time features")
    return enriched


def encode_genres(movies_df: DataFrame) -> DataFrame:
    exploded = (
        movies_df.withColumn("genre_array", F.split(F.col("genres"), "\\|"))
        .withColumn("genre", F.explode(F.col("genre_array")))
        .drop("genre_array")
    )
    LOGGER.info("Encoded genres")
    return exploded


def build_movie_genre_weights(movie_genres_df: DataFrame) -> DataFrame:
    window = Window.partitionBy("movieId")
    weighted = (
        movie_genres_df.select("movieId", "genre")
        .dropDuplicates(["movieId", "genre"])
        .withColumn("genre_count", F.count("*").over(window))
        .withColumn("movie_genre_weight", F.lit(1.0) / F.col("genre_count"))
        .drop("genre_count")
    )
    return weighted


def build_tag_features(tags_df: DataFrame, min_tag_count: int = 5) -> DataFrame:
    movie_tag_counts = (
        tags_df.groupBy("movieId", "tag")
        .agg(F.count("*").alias("tag_count"))
        .filter(F.col("tag_count") >= F.lit(min_tag_count))
    )

    tag_total_window = Window.partitionBy("movieId")
    normalized = (
        movie_tag_counts.withColumn("movie_tag_total", F.sum("tag_count").over(tag_total_window))
        .withColumn("movie_tag_weight", F.col("tag_count") / F.col("movie_tag_total"))
        .drop("movie_tag_total")
    )
    LOGGER.info("Built tag features")
    return normalized


def filter_tags_to_training_window(tags_df: DataFrame, train_ratings_df: DataFrame) -> DataFrame:
    user_cutoff_df = train_ratings_df.groupBy("userId").agg(F.max("timestamp").alias("train_cutoff_ts"))
    filtered = (
        tags_df.join(user_cutoff_df, on="userId", how="inner")
        .filter(F.col("timestamp") <= F.col("train_cutoff_ts"))
        .drop("train_cutoff_ts")
    )
    LOGGER.info("Filtered tags to training time window")
    return filtered


def build_tag_tfidf_features(
    tags_df: DataFrame,
    num_features: int = 1 << 16,
    min_doc_freq: int = 2,
) -> Tuple[DataFrame, DataFrame]:
    movie_docs = tags_df.groupBy("movieId").agg(F.concat_ws(" ", F.collect_list("tag")).alias("tags_text"))
    user_docs = tags_df.groupBy("userId").agg(F.concat_ws(" ", F.collect_list("tag")).alias("tags_text"))

    tokenizer = RegexTokenizer(inputCol="tags_text", outputCol="tokens", pattern="\\W+", minTokenLength=2)
    hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=num_features)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features", minDocFreq=min_doc_freq)

    movie_tokens = tokenizer.transform(movie_docs)
    user_tokens = tokenizer.transform(user_docs)

    movie_tf = hashing_tf.transform(movie_tokens)
    user_tf = hashing_tf.transform(user_tokens)

    idf_model = idf.fit(movie_tf)
    movie_tfidf = idf_model.transform(movie_tf).select("movieId", F.col("tfidf_features").alias("movie_tag_tfidf"))
    user_tfidf = idf_model.transform(user_tf).select("userId", F.col("tfidf_features").alias("user_tag_tfidf"))

    LOGGER.info("Built TF-IDF tag features")
    return movie_tfidf, user_tfidf
