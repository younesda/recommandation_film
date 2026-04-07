from __future__ import annotations

from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def score_content_candidates(
    candidate_items_df: DataFrame,
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
) -> DataFrame:
    expanded_candidates = candidate_items_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"])
    candidate_genres = expanded_candidates.join(movie_genre_weights_df, on="movieId", how="left")

    scored = candidate_genres.join(
        user_genre_profiles_df.select("userId", "genre", "genre_weight"),
        on=["userId", "genre"],
        how="left",
    ).fillna({"genre_weight": 0.0, "movie_genre_weight": 0.0})

    content_scores = scored.groupBy("userId", "movieId").agg(
        F.sum(F.col("genre_weight") * F.col("movie_genre_weight")).alias("content_genre_score"),
        F.array_sort(F.collect_set(F.when(F.col("genre_weight") > F.lit(0.0), F.col("genre")))).alias("matched_genres"),
    )
    content_scores = content_scores.withColumn("matched_genres", F.expr("filter(matched_genres, x -> x is not null)"))

    LOGGER.info("Computed genre content scores")
    return content_scores


def score_tag_candidates(
    candidate_items_df: DataFrame,
    user_tag_tfidf_df: DataFrame,
    movie_tag_tfidf_df: DataFrame,
) -> DataFrame:
    if user_tag_tfidf_df is None or movie_tag_tfidf_df is None:
        return candidate_items_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).withColumn(
            "content_tag_score",
            F.lit(0.0),
        )

    candidates = candidate_items_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"])
    tagged_candidates = (
        candidates.join(movie_tag_tfidf_df.select("movieId", "movie_tag_tfidf"), on="movieId", how="left")
        .join(user_tag_tfidf_df.select("userId", "user_tag_tfidf"), on="userId", how="left")
    )

    with_arrays = (
        tagged_candidates.withColumn("user_arr", vector_to_array("user_tag_tfidf"))
        .withColumn("movie_arr", vector_to_array("movie_tag_tfidf"))
        .withColumn(
            "dot_product",
            F.expr(
                "aggregate(zip_with(user_arr, movie_arr, (x, y) -> coalesce(x, 0D) * coalesce(y, 0D)), 0D, (acc, x) -> acc + x)"
            ),
        )
        .withColumn(
            "user_norm",
            F.sqrt(F.expr("aggregate(transform(user_arr, x -> x * x), 0D, (acc, x) -> acc + x)")),
        )
        .withColumn(
            "movie_norm",
            F.sqrt(F.expr("aggregate(transform(movie_arr, x -> x * x), 0D, (acc, x) -> acc + x)")),
        )
        .withColumn(
            "content_tag_score",
            F.when(
                (F.col("user_norm") > F.lit(0.0)) & (F.col("movie_norm") > F.lit(0.0)),
                F.col("dot_product") / (F.col("user_norm") * F.col("movie_norm")),
            ).otherwise(F.lit(0.0)),
        )
    )

    return with_arrays.select("userId", "movieId", "content_tag_score")


def _score_tag_candidates_legacy(
    candidate_items_df: DataFrame,
    user_tag_profiles_df: DataFrame,
    movie_tag_features_df: DataFrame,
) -> DataFrame:
    candidates = candidate_items_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"])
    movie_tags = candidates.join(movie_tag_features_df.select("movieId", "tag", "movie_tag_weight"), on="movieId", how="left")

    joined = movie_tags.join(
        user_tag_profiles_df.select("userId", "tag", "user_tag_weight"),
        on=["userId", "tag"],
        how="left",
    ).fillna({"user_tag_weight": 0.0, "movie_tag_weight": 0.0})

    return joined.groupBy("userId", "movieId").agg(
        F.sum(F.col("user_tag_weight") * F.col("movie_tag_weight")).alias("content_tag_score")
    )


def build_content_scores(
    candidate_items_df: DataFrame,
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    user_tag_tfidf_df: DataFrame | None = None,
    movie_tag_tfidf_df: DataFrame | None = None,
    user_tag_profiles_df: DataFrame | None = None,
    movie_tag_features_df: DataFrame | None = None,
    tag_weight: float = 0.2,
) -> DataFrame:
    genre_scores = score_content_candidates(
        candidate_items_df=candidate_items_df,
        user_genre_profiles_df=user_genre_profiles_df,
        movie_genre_weights_df=movie_genre_weights_df,
    )
    if user_tag_tfidf_df is not None and movie_tag_tfidf_df is not None:
        tag_scores = score_tag_candidates(
            candidate_items_df=candidate_items_df,
            user_tag_tfidf_df=user_tag_tfidf_df,
            movie_tag_tfidf_df=movie_tag_tfidf_df,
        )
    elif user_tag_profiles_df is not None and movie_tag_features_df is not None:
        tag_scores = _score_tag_candidates_legacy(
            candidate_items_df=candidate_items_df,
            user_tag_profiles_df=user_tag_profiles_df,
            movie_tag_features_df=movie_tag_features_df,
        )
    else:
        tag_scores = candidate_items_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).withColumn(
            "content_tag_score",
            F.lit(0.0),
        )

    combined = genre_scores.join(tag_scores, on=["userId", "movieId"], how="left").fillna({"content_tag_score": 0.0})
    combined = combined.withColumn(
        "content_score",
        (F.lit(1.0 - tag_weight) * F.col("content_genre_score")) + (F.lit(tag_weight) * F.col("content_tag_score")),
    )
    return combined


def generate_content_candidates(
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    seen_interactions_df: DataFrame,
    k: int = 50,
) -> DataFrame:
    positive_user_genres = user_genre_profiles_df.filter(F.col("genre_weight") > F.lit(0.0)).select(
        "userId",
        "genre",
        "genre_weight",
    )

    scored = positive_user_genres.join(movie_genre_weights_df.select("movieId", "genre", "movie_genre_weight"), on="genre", how="inner")
    aggregated = scored.groupBy("userId", "movieId").agg(
        F.sum(F.col("genre_weight") * F.col("movie_genre_weight")).alias("content_score"),
        F.array_sort(F.collect_set("genre")).alias("matched_genres"),
    )

    unseen = aggregated.join(
        seen_interactions_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]),
        on=["userId", "movieId"],
        how="left_anti",
    )

    ranking_window = Window.partitionBy("userId").orderBy(F.col("content_score").desc(), F.col("movieId").asc())
    ranked = unseen.withColumn("content_rank", F.row_number().over(ranking_window)).filter(F.col("content_rank") <= F.lit(k))
    return ranked.select("userId", "movieId", "content_score", "matched_genres")
