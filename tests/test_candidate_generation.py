from __future__ import annotations

from pyspark.sql import SparkSession

from src.models.candidate_generation import merge_candidate_sources
from src.models.ranking_model import build_ranking_features


def test_candidate_sources_preserve_scores_and_rank_features() -> None:
    spark = SparkSession.builder.master("local[1]").appName("test-candidate-sources").getOrCreate()
    try:
        als_candidates = spark.createDataFrame(
            [
                (1, 10, 0.92, 1),
                (1, 20, 0.61, 3),
            ],
            ["userId", "movieId", "als_score", "als_candidate_rank"],
        )
        content_candidates = spark.createDataFrame(
            [
                (1, 10, 0.30, 2, ["Comedy"]),
                (1, 30, 0.55, 1, ["Drama"]),
            ],
            ["userId", "movieId", "content_score", "content_rank", "matched_genres"],
        )
        tag_candidates = spark.createDataFrame(
            [
                (1, 10, 0.10, 4),
            ],
            ["userId", "movieId", "content_tag_score", "tag_rank"],
        )
        popular_candidates = spark.createDataFrame(
            [
                (1, 30, 0.80, 2),
            ],
            ["userId", "movieId", "popular_candidate_score", "popular_rank"],
        )
        recent_candidates = spark.createDataFrame(
            [
                (1, 20, 0.50, 1),
            ],
            ["userId", "movieId", "recent_candidate_score", "recent_rank"],
        )

        merged = merge_candidate_sources(
            als_candidates_df=als_candidates,
            content_candidates_df=content_candidates,
            tag_candidates_df=tag_candidates,
            popular_candidates_df=popular_candidates,
            recent_candidates_df=recent_candidates,
        )

        collaborative_scores = spark.createDataFrame(
            [
                (1, 10, 0.95),
                (1, 20, 0.66),
                (1, 30, 0.20),
            ],
            ["userId", "movieId", "als_score"],
        )
        explicit_scores = spark.createDataFrame(
            [
                (1, 10, 4.5),
                (1, 20, 4.0),
                (1, 30, 3.8),
            ],
            ["userId", "movieId", "als_score"],
        )
        content_scores = spark.createDataFrame(
            [
                (1, 10, 0.35, 0.30, 0.10, ["Comedy"]),
                (1, 20, 0.12, 0.12, 0.00, []),
                (1, 30, 0.60, 0.55, 0.05, ["Drama"]),
            ],
            ["userId", "movieId", "content_score", "content_genre_score", "content_tag_score", "matched_genres"],
        )
        item_features = spark.createDataFrame(
            [
                (10, 50.0, 4.2, 0.6, 3.9, 2.2, 0.2, 0.1),
                (20, 20.0, 3.9, 0.4, 3.0, 1.2, 0.5, 0.2),
                (30, 70.0, 4.5, 0.7, 4.3, 2.8, 0.3, 0.08),
            ],
            [
                "movieId",
                "item_interaction_count",
                "item_avg_rating",
                "item_positive_rate",
                "item_popularity_log",
                "item_popularity_score",
                "item_recent_score",
                "item_novelty_score",
            ],
        )
        user_profiles = spark.createDataFrame(
            [
                (1, 4.1, 120.0, 0.55, 0.8),
            ],
            ["userId", "avg_rating", "num_ratings", "user_positive_rate", "user_rating_stddev"],
        )

        features = build_ranking_features(
            candidate_sources_df=merged,
            collaborative_scores_df=collaborative_scores,
            explicit_scores_df=explicit_scores,
            content_scores_df=content_scores,
            item_features_df=item_features,
            user_profiles_df=user_profiles,
        )

        by_movie = {row["movieId"]: row.asDict() for row in features.collect()}

        assert by_movie[10]["als_candidate_score"] == 0.92
        assert by_movie[10]["content_candidate_score"] == 0.30
        assert by_movie[10]["tag_candidate_score"] == 0.10
        assert by_movie[10]["source_als_candidate"] == 1
        assert by_movie[10]["source_content_candidate"] == 1
        assert by_movie[10]["source_tag_candidate"] == 1
        assert abs(by_movie[10]["als_candidate_rank_reciprocal"] - 1.0) < 1e-9
        assert abs(by_movie[10]["content_candidate_rank_reciprocal"] - 0.5) < 1e-9
        assert by_movie[20]["source_recent_candidate"] == 1
        assert abs(by_movie[20]["recent_candidate_rank_reciprocal"] - 1.0) < 1e-9
        assert by_movie[30]["source_popular_candidate"] == 1
        assert by_movie[30]["popular_candidate_score"] == 0.80
        assert by_movie[30]["best_candidate_rank_reciprocal"] == 1.0
    finally:
        spark.stop()
