from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.settings import PipelineSettings
from src.evaluation.mae import compute_mae
from src.evaluation.ndcg_at_k import compute_ndcg_at_k
from src.evaluation.precision_at_k import compute_precision_at_k
from src.evaluation.recall_at_k import compute_recall_at_k
from src.evaluation.rmse import compute_rmse
from src.ingestion.load_data import load_all_data
from src.models.als_model import (
    recommend_for_users_flat,
    retrain_best_als,
    retrain_best_ranking_als,
    score_als,
    score_als_candidates,
    train_als_with_tuning,
    train_ranking_als_with_tuning,
)
from src.models.candidate_generation import (
    build_item_ranking_features,
    generate_popular_candidates,
    generate_recent_candidates,
    merge_candidate_sources,
)
from src.models.content_model import (
    build_content_scores,
    generate_content_candidates,
    generate_tag_candidates,
)
from src.models.hybrid_model import select_top_k_recommendations
from src.models.ranking_model import (
    build_ranker_training_frame,
    build_ranking_features,
    score_candidates_with_ranker,
    train_xgb_ranker,
)
from src.preprocessing.clean_data import clean_movies, clean_ratings, clean_tags, time_based_split
from src.preprocessing.feature_engineering import (
    add_time_features,
    build_movie_genre_weights,
    build_tag_features,
    build_tag_tfidf_features,
    encode_genres,
    filter_tags_to_training_window,
)
from src.preprocessing.user_profiles import build_user_genre_profiles, build_user_tag_profiles, create_user_profiles
from src.storage.database import save_to_postgres
from src.storage.save_parquet import append_metrics_history, save_metrics, save_parquet
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _prepare_outputs(base_dir: str) -> Dict[str, str]:
    return {
        "recommendations_parquet": os.path.join(base_dir, "recommendations"),
        "user_profiles_parquet": os.path.join(base_dir, "user_profiles"),
        "seen_interactions_parquet": os.path.join(base_dir, "seen_interactions"),
        "metrics_json": os.path.join(base_dir, "metrics", "metrics.json"),
        "metrics_history_jsonl": os.path.join(base_dir, "metrics", "metrics_history.jsonl"),
        "metrics_parquet": os.path.join(base_dir, "metrics", "metrics_parquet"),
        "als_model": os.path.join(base_dir, "models", "als"),
        "ranking_als_model": os.path.join(base_dir, "models", "ranking_als"),
    }


def _attach_movie_metadata(recommendations_df: DataFrame, movies_df: DataFrame) -> DataFrame:
    return recommendations_df.join(
        movies_df.select("movieId", "title", "genres"),
        on="movieId",
        how="left",
    )


def _is_empty(df: DataFrame) -> bool:
    return len(df.head(1)) == 0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _is_windows_without_hadoop() -> bool:
    return os.name == "nt" and not (os.getenv("HADOOP_HOME") or os.getenv("hadoop.home.dir"))


def _exclude_seen_interactions(df: DataFrame, seen_interactions_df: DataFrame) -> DataFrame:
    seen = seen_interactions_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"])
    return df.join(seen, on=["userId", "movieId"], how="left_anti")


def _build_tag_tfidf_if_available(
    use_tags: bool,
    tags_df: DataFrame,
) -> tuple[DataFrame | None, DataFrame | None]:
    if not use_tags or _is_empty(tags_df):
        return None, None
    movie_tag_tfidf_df, user_tag_tfidf_df = build_tag_tfidf_features(tags_df)
    return movie_tag_tfidf_df.cache(), user_tag_tfidf_df.cache()


def _build_tag_profiles_if_available(
    use_tags: bool,
    tags_df: DataFrame,
) -> tuple[DataFrame | None, DataFrame | None]:
    if not use_tags or _is_empty(tags_df):
        return None, None
    movie_tag_features_df = build_tag_features(tags_df).cache()
    user_tag_profiles_df = build_user_tag_profiles(tags_df).cache()
    return movie_tag_features_df, user_tag_profiles_df


def _resolve_candidate_k(cfg: PipelineSettings) -> int:
    return max(cfg.hybrid.top_k * cfg.hybrid.candidate_multiplier, cfg.hybrid.top_k)


def _resolve_als_request_k(cfg: PipelineSettings, candidate_k: int) -> int:
    return max(candidate_k * cfg.hybrid.als_candidate_overfetch_multiplier, candidate_k)


def _zero_source_candidates(candidate_df: DataFrame) -> DataFrame:
    return candidate_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).select(
        "userId",
        "movieId",
        F.lit(0.0).alias("als_candidate_score"),
        F.lit(0.0).alias("content_candidate_score"),
        F.lit(0.0).alias("tag_candidate_score"),
        F.lit(0.0).alias("popular_candidate_score"),
        F.lit(0.0).alias("recent_candidate_score"),
        F.lit(None).cast("int").alias("als_candidate_rank"),
        F.lit(None).cast("int").alias("content_candidate_rank"),
        F.lit(None).cast("int").alias("tag_candidate_rank"),
        F.lit(None).cast("int").alias("popular_candidate_rank"),
        F.lit(None).cast("int").alias("recent_candidate_rank"),
        F.lit(0).alias("source_als_candidate"),
        F.lit(0).alias("source_content_candidate"),
        F.lit(0).alias("source_tag_candidate"),
        F.lit(0).alias("source_popular_candidate"),
        F.lit(0).alias("source_recent_candidate"),
        F.lit(0).alias("candidate_source_count"),
    )


def _add_oracle_positives_to_candidates(
    candidate_sources_df: DataFrame,
    ground_truth_df: DataFrame,
    positive_threshold: float,
) -> DataFrame:
    oracle_positive_df = (
        ground_truth_df.filter(F.col("rating") >= F.lit(positive_threshold))
        .select("userId", "movieId")
        .dropDuplicates(["userId", "movieId"])
    )
    zero_flag_df = _zero_source_candidates(oracle_positive_df)

    merged = candidate_sources_df.unionByName(zero_flag_df, allowMissingColumns=True)
    return (
        merged.groupBy("userId", "movieId")
        .agg(
            F.max("als_candidate_score").alias("als_candidate_score"),
            F.max("content_candidate_score").alias("content_candidate_score"),
            F.max("tag_candidate_score").alias("tag_candidate_score"),
            F.max("popular_candidate_score").alias("popular_candidate_score"),
            F.max("recent_candidate_score").alias("recent_candidate_score"),
            F.min("als_candidate_rank").alias("als_candidate_rank"),
            F.min("content_candidate_rank").alias("content_candidate_rank"),
            F.min("tag_candidate_rank").alias("tag_candidate_rank"),
            F.min("popular_candidate_rank").alias("popular_candidate_rank"),
            F.min("recent_candidate_rank").alias("recent_candidate_rank"),
            F.max("source_als_candidate").alias("source_als_candidate"),
            F.max("source_content_candidate").alias("source_content_candidate"),
            F.max("source_tag_candidate").alias("source_tag_candidate"),
            F.max("source_popular_candidate").alias("source_popular_candidate"),
            F.max("source_recent_candidate").alias("source_recent_candidate"),
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


def _compute_candidate_recall(
    candidates_df: DataFrame,
    ground_truth_df: DataFrame,
    positive_threshold: float,
) -> float:
    relevant_items = (
        ground_truth_df.filter(F.col("rating") >= F.lit(positive_threshold))
        .select("userId", "movieId")
        .dropDuplicates(["userId", "movieId"])
    )
    relevant_count = relevant_items.groupBy("userId").agg(F.count("*").alias("relevant_count"))
    hit_count = (
        candidates_df.select("userId", "movieId")
        .dropDuplicates(["userId", "movieId"])
        .join(relevant_items, on=["userId", "movieId"], how="inner")
        .groupBy("userId")
        .agg(F.count("*").alias("hit_count"))
    )
    recall_df = (
        relevant_count.join(hit_count, on="userId", how="left")
        .fillna({"hit_count": 0})
        .withColumn(
            "candidate_recall",
            F.when(F.col("relevant_count") > F.lit(0), F.col("hit_count") / F.col("relevant_count")).otherwise(F.lit(0.0)),
        )
    )
    result = recall_df.agg(F.avg("candidate_recall").alias("mean_candidate_recall")).collect()[0]["mean_candidate_recall"]
    return float(result) if result is not None else 0.0


def _build_multi_source_candidates(
    users_df: DataFrame,
    ranking_als_model,
    seen_interactions_df: DataFrame,
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    user_tag_profiles_df: DataFrame | None,
    movie_tag_features_df: DataFrame | None,
    item_features_df: DataFrame,
    candidate_k: int,
    settings: PipelineSettings,
) -> Dict[str, DataFrame]:
    user_genre_subset_df = user_genre_profiles_df.join(users_df, on="userId", how="inner").cache()
    user_tag_subset_df = None
    if user_tag_profiles_df is not None:
        user_tag_subset_df = user_tag_profiles_df.join(users_df, on="userId", how="inner").cache()

    als_request_k = _resolve_als_request_k(settings, candidate_k)
    als_ranking_window = Window.partitionBy("userId").orderBy(F.col("als_score").desc(), F.col("movieId").asc())
    als_candidates_df = (
        recommend_for_users_flat(ranking_als_model, users_df, als_request_k)
        .join(
            seen_interactions_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]),
            on=["userId", "movieId"],
            how="left_anti",
        )
        .withColumn("als_candidate_rank", F.row_number().over(als_ranking_window))
        .filter(F.col("als_candidate_rank") <= F.lit(candidate_k))
        .select("userId", "movieId", "als_score", "als_candidate_rank")
        .cache()
    )
    genre_candidates_df = generate_content_candidates(
        user_genre_profiles_df=user_genre_subset_df,
        movie_genre_weights_df=movie_genre_weights_df,
        seen_interactions_df=seen_interactions_df,
        k=candidate_k,
    ).cache()
    tag_candidates_df = generate_tag_candidates(
        user_tag_profiles_df=user_tag_subset_df,
        movie_tag_features_df=movie_tag_features_df,
        seen_interactions_df=seen_interactions_df,
        k=candidate_k,
    ).cache()
    popular_candidates_df = generate_popular_candidates(
        user_genre_profiles_df=user_genre_subset_df,
        movie_genre_weights_df=movie_genre_weights_df,
        item_features_df=item_features_df,
        seen_interactions_df=seen_interactions_df,
        k=candidate_k,
    ).cache()
    recent_candidates_df = generate_recent_candidates(
        user_genre_profiles_df=user_genre_subset_df,
        movie_genre_weights_df=movie_genre_weights_df,
        item_features_df=item_features_df,
        seen_interactions_df=seen_interactions_df,
        k=candidate_k,
    ).cache()

    candidate_sources_df = merge_candidate_sources(
        als_candidates_df=als_candidates_df,
        content_candidates_df=genre_candidates_df,
        tag_candidates_df=tag_candidates_df,
        popular_candidates_df=popular_candidates_df,
        recent_candidates_df=recent_candidates_df,
    )
    candidate_sources_df = _exclude_seen_interactions(candidate_sources_df, seen_interactions_df).cache()

    return {
        "als_candidates": als_candidates_df,
        "genre_candidates": genre_candidates_df,
        "tag_candidates": tag_candidates_df,
        "popular_candidates": popular_candidates_df,
        "recent_candidates": recent_candidates_df,
        "candidate_sources": candidate_sources_df,
    }


def _build_content_feature_scores(
    candidate_sources_df: DataFrame,
    user_genre_profiles_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    user_tag_tfidf_df: DataFrame | None,
    movie_tag_tfidf_df: DataFrame | None,
    user_tag_profiles_df: DataFrame | None,
    movie_tag_features_df: DataFrame | None,
    tag_weight: float,
) -> DataFrame:
    return build_content_scores(
        candidate_items_df=candidate_sources_df.select("userId", "movieId"),
        user_genre_profiles_df=user_genre_profiles_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_tfidf_df=user_tag_tfidf_df,
        movie_tag_tfidf_df=movie_tag_tfidf_df,
        user_tag_profiles_df=user_tag_profiles_df,
        movie_tag_features_df=movie_tag_features_df,
        tag_weight=tag_weight,
    ).cache()


def _compute_dashboard_metrics(
    recommendation_export_df: DataFrame,
    movies_clean_df: DataFrame,
    ratings_clean_df: DataFrame,
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    train_val_df: DataFrame,
) -> Dict[str, float]:
    total_recommendations = float(recommendation_export_df.count())
    active_users = float(ratings_clean_df.select("userId").distinct().count())
    active_movies = float(movies_clean_df.select("movieId").distinct().count())
    users_covered = float(recommendation_export_df.select("userId").distinct().count())
    movies_recommended = float(recommendation_export_df.select("movieId").distinct().count())

    score_stats = recommendation_export_df.agg(
        F.avg("final_score").alias("avg_final_score"),
        F.avg("als_score").alias("avg_als_score"),
        F.avg("content_score").alias("avg_content_score"),
    ).collect()[0]
    final_score_quantiles = recommendation_export_df.approxQuantile("final_score", [0.5, 0.9], 0.01)

    recommended_genres_df = (
        recommendation_export_df.select(F.explode(F.split(F.col("genres"), "\\|")).alias("genre"))
        .filter(F.col("genre").isNotNull() & (F.trim(F.col("genre")) != F.lit("")))
    )
    genres_covered = float(recommended_genres_df.select("genre").distinct().count())

    item_popularity_df = train_val_df.groupBy("movieId").agg(F.count("*").alias("interaction_count")).cache()
    popularity_quantiles = item_popularity_df.approxQuantile("interaction_count", [0.5], 0.01)
    popularity_median = float(popularity_quantiles[0]) if popularity_quantiles else 0.0
    rec_with_popularity_df = recommendation_export_df.join(item_popularity_df, on="movieId", how="left").fillna(
        {"interaction_count": 0}
    )
    long_tail_recommendations = float(
        rec_with_popularity_df.filter(F.col("interaction_count") <= F.lit(popularity_median)).count()
    )

    return {
        "active_users": active_users,
        "active_movies": active_movies,
        "train_rows": float(train_df.count()),
        "val_rows": float(val_df.count()),
        "test_rows": float(test_df.count()),
        "recommendation_rows": total_recommendations,
        "users_covered": users_covered,
        "movies_recommended": movies_recommended,
        "user_coverage_ratio": _safe_ratio(users_covered, active_users),
        "catalog_coverage_ratio": _safe_ratio(movies_recommended, active_movies),
        "avg_recommendations_per_user": _safe_ratio(total_recommendations, users_covered),
        "avg_final_score": float(score_stats["avg_final_score"] or 0.0),
        "p50_final_score": float(final_score_quantiles[0]) if len(final_score_quantiles) > 0 else 0.0,
        "p90_final_score": float(final_score_quantiles[1]) if len(final_score_quantiles) > 1 else 0.0,
        "avg_als_score": float(score_stats["avg_als_score"] or 0.0),
        "avg_content_score": float(score_stats["avg_content_score"] or 0.0),
        "genres_covered": genres_covered,
        "popularity_median_interactions": popularity_median,
        "long_tail_recommendation_share": _safe_ratio(long_tail_recommendations, total_recommendations),
    }


def run_pipeline(
    spark: SparkSession,
    settings: PipelineSettings | None = None,
    use_tags: bool = True,
    save_recommendations_to_postgres: bool = False,
    postgres_jdbc_url: str | None = None,
    postgres_user: str | None = None,
    postgres_password: str | None = None,
) -> Dict[str, float | str]:
    cfg = settings or PipelineSettings()
    output_paths = _prepare_outputs(cfg.data_paths.output_base)
    candidate_k = _resolve_candidate_k(cfg)
    content_tag_weight = 0.3

    LOGGER.info("Pipeline started")
    ratings_df, movies_df, tags_df = load_all_data(spark=spark, settings=cfg, prefer_parquet=True)

    movies_clean = clean_movies(movies_df).cache()
    tags_clean = clean_tags(tags_df).cache()
    ratings_clean = clean_ratings(ratings_df, settings=cfg).cache()
    ratings_features = add_time_features(ratings_clean).cache()

    movie_genres_df = encode_genres(movies_clean).cache()
    movie_genre_weights_df = build_movie_genre_weights(movie_genres_df).cache()

    train_df, val_df, test_df = time_based_split(ratings_features, settings=cfg)
    train_df = train_df.cache()
    val_df = val_df.cache()
    test_df = test_df.cache()

    explicit_als_train_model, best_als_params = train_als_with_tuning(train_df, val_df, settings=cfg)
    ranking_als_train_model, best_ranking_als_params = train_ranking_als_with_tuning(train_df, val_df, settings=cfg)

    train_user_profiles_df = create_user_profiles(train_df).cache()
    user_genre_profiles_train_df = build_user_genre_profiles(train_df, movie_genres_df).cache()
    seen_train_df = train_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).cache()
    item_features_train_df = build_item_ranking_features(train_df, positive_threshold=cfg.min_positive_rating).cache()

    tags_train_df = filter_tags_to_training_window(tags_clean, train_df).cache()
    movie_tag_tfidf_train_df, user_tag_tfidf_train_df = _build_tag_tfidf_if_available(use_tags, tags_train_df)
    movie_tag_features_train_df, user_tag_profiles_train_df = _build_tag_profiles_if_available(use_tags, tags_train_df)

    val_users_df = val_df.select("userId").distinct().cache()
    ranking_train_candidates = _build_multi_source_candidates(
        users_df=val_users_df,
        ranking_als_model=ranking_als_train_model,
        seen_interactions_df=seen_train_df,
        user_genre_profiles_df=user_genre_profiles_train_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_profiles_df=user_tag_profiles_train_df,
        movie_tag_features_df=movie_tag_features_train_df,
        item_features_df=item_features_train_df,
        candidate_k=candidate_k,
        settings=cfg,
    )

    val_candidate_recall = _compute_candidate_recall(
        ranking_train_candidates["candidate_sources"],
        val_df,
        positive_threshold=cfg.min_positive_rating,
    )

    ranking_train_sources_df = _add_oracle_positives_to_candidates(
        ranking_train_candidates["candidate_sources"],
        val_df,
        positive_threshold=cfg.min_positive_rating,
    ).cache()
    ranking_train_content_scores_df = _build_content_feature_scores(
        candidate_sources_df=ranking_train_sources_df,
        user_genre_profiles_df=user_genre_profiles_train_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_tfidf_df=user_tag_tfidf_train_df,
        movie_tag_tfidf_df=movie_tag_tfidf_train_df,
        user_tag_profiles_df=user_tag_profiles_train_df,
        movie_tag_features_df=movie_tag_features_train_df,
        tag_weight=content_tag_weight,
    )
    ranking_train_collab_scores_df = score_als_candidates(
        ranking_als_train_model,
        ranking_train_sources_df.select("userId", "movieId"),
    ).cache()
    ranking_train_explicit_scores_df = score_als_candidates(
        explicit_als_train_model,
        ranking_train_sources_df.select("userId", "movieId"),
    ).cache()
    ranking_train_feature_df = build_ranking_features(
        candidate_sources_df=ranking_train_sources_df,
        collaborative_scores_df=ranking_train_collab_scores_df,
        explicit_scores_df=ranking_train_explicit_scores_df,
        content_scores_df=ranking_train_content_scores_df,
        item_features_df=item_features_train_df,
        user_profiles_df=train_user_profiles_df,
    ).cache()
    ranker_training_df = build_ranker_training_frame(
        feature_df=ranking_train_feature_df,
        ground_truth_df=val_df,
        positive_threshold=cfg.min_positive_rating,
    ).cache()
    ranker_model, ranker_feature_cols, ranker_info = train_xgb_ranker(ranker_training_df, settings=cfg)

    explicit_als_model = retrain_best_als(train_df, val_df, best_als_params, settings=cfg)
    ranking_als_model = retrain_best_ranking_als(train_df, val_df, best_ranking_als_params, settings=cfg)

    if _is_windows_without_hadoop():
        LOGGER.warning(
            "Skipping ALS model persistence at path=%s because Windows local Spark writes need HADOOP_HOME/winutils.",
            output_paths["als_model"],
        )
    else:
        explicit_als_model.write().overwrite().save(output_paths["als_model"])
        ranking_als_model.write().overwrite().save(output_paths["ranking_als_model"])

    train_val_df = train_df.unionByName(val_df).cache()
    seen_interactions_df = train_val_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).cache()
    user_profiles_final_df = create_user_profiles(train_val_df).cache()
    user_genre_profiles_final_df = build_user_genre_profiles(train_val_df, movie_genres_df).cache()
    item_features_final_df = build_item_ranking_features(train_val_df, positive_threshold=cfg.min_positive_rating).cache()

    tags_train_val_df = filter_tags_to_training_window(tags_clean, train_val_df).cache()
    movie_tag_tfidf_final_df, user_tag_tfidf_final_df = _build_tag_tfidf_if_available(use_tags, tags_train_val_df)
    movie_tag_features_final_df, user_tag_profiles_final_df = _build_tag_profiles_if_available(use_tags, tags_train_val_df)

    als_predictions_test = score_als(explicit_als_model, test_df)
    rmse = compute_rmse(als_predictions_test)
    mae = compute_mae(als_predictions_test)

    all_users_df = train_val_df.select("userId").distinct().cache()
    final_candidates = _build_multi_source_candidates(
        users_df=all_users_df,
        ranking_als_model=ranking_als_model,
        seen_interactions_df=seen_interactions_df,
        user_genre_profiles_df=user_genre_profiles_final_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_profiles_df=user_tag_profiles_final_df,
        movie_tag_features_df=movie_tag_features_final_df,
        item_features_df=item_features_final_df,
        candidate_k=candidate_k,
        settings=cfg,
    )

    test_candidate_recall = _compute_candidate_recall(
        final_candidates["candidate_sources"],
        test_df,
        positive_threshold=cfg.min_positive_rating,
    )

    final_content_scores_df = _build_content_feature_scores(
        candidate_sources_df=final_candidates["candidate_sources"],
        user_genre_profiles_df=user_genre_profiles_final_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_tfidf_df=user_tag_tfidf_final_df,
        movie_tag_tfidf_df=movie_tag_tfidf_final_df,
        user_tag_profiles_df=user_tag_profiles_final_df,
        movie_tag_features_df=movie_tag_features_final_df,
        tag_weight=content_tag_weight,
    )
    final_collab_scores_df = score_als_candidates(
        ranking_als_model,
        final_candidates["candidate_sources"].select("userId", "movieId"),
    ).cache()
    final_explicit_scores_df = score_als_candidates(
        explicit_als_model,
        final_candidates["candidate_sources"].select("userId", "movieId"),
    ).cache()
    final_feature_df = build_ranking_features(
        candidate_sources_df=final_candidates["candidate_sources"],
        collaborative_scores_df=final_collab_scores_df,
        explicit_scores_df=final_explicit_scores_df,
        content_scores_df=final_content_scores_df,
        item_features_df=item_features_final_df,
        user_profiles_df=user_profiles_final_df,
    ).cache()

    ranker_scores_df = score_candidates_with_ranker(
        spark=spark,
        candidate_feature_df=final_feature_df,
        ranker_model=ranker_model,
        feature_cols=ranker_feature_cols,
    )
    ranked_candidates_df = (
        final_feature_df.join(ranker_scores_df, on=["userId", "movieId"], how="inner")
        .withColumn("als_score", F.col("cf_score"))
        .cache()
    )
    ranked_candidates_df = _exclude_seen_interactions(ranked_candidates_df, seen_interactions_df).cache()

    top_recommendations_df = select_top_k_recommendations(ranked_candidates_df, cfg.hybrid.top_k)
    top_recommendations_df = _attach_movie_metadata(top_recommendations_df, movies_clean).cache()

    precision_k = compute_precision_at_k(
        recommendations_df=top_recommendations_df.select("userId", "movieId", "rank", "final_score"),
        ground_truth_df=test_df,
        k=cfg.hybrid.top_k,
        positive_threshold=cfg.min_positive_rating,
    )
    recall_k = compute_recall_at_k(
        recommendations_df=top_recommendations_df.select("userId", "movieId", "rank", "final_score"),
        ground_truth_df=test_df,
        k=cfg.hybrid.top_k,
        positive_threshold=cfg.min_positive_rating,
    )
    ndcg_k = compute_ndcg_at_k(
        recommendations_df=top_recommendations_df.select("userId", "movieId", "rank", "final_score"),
        ground_truth_df=test_df,
        k=cfg.hybrid.top_k,
        positive_threshold=cfg.min_positive_rating,
    )

    recommendation_export_df = top_recommendations_df.select(
        "userId",
        "movieId",
        "title",
        "genres",
        "rank",
        "als_score",
        "explicit_als_score",
        "content_score",
        "content_genre_score",
        "content_tag_score",
        "final_score",
        "candidate_source_count",
        "als_candidate_score",
        "content_candidate_score",
        "tag_candidate_score",
        "popular_candidate_score",
        "recent_candidate_score",
        "source_als_candidate",
        "source_content_candidate",
        "source_tag_candidate",
        "source_popular_candidate",
        "source_recent_candidate",
        "explanation",
    )

    dashboard_metrics = _compute_dashboard_metrics(
        recommendation_export_df=recommendation_export_df,
        movies_clean_df=movies_clean,
        ratings_clean_df=ratings_clean,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_val_df=train_val_df,
    )

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        f"precision_at_{cfg.hybrid.top_k}": float(precision_k),
        f"recall_at_{cfg.hybrid.top_k}": float(recall_k),
        f"ndcg_at_{cfg.hybrid.top_k}": float(ndcg_k),
        "content_tag_weight": float(content_tag_weight),
        "candidate_pool_size": float(candidate_k),
        "val_candidate_recall": float(val_candidate_recall),
        "test_candidate_recall": float(test_candidate_recall),
        "ranker_val_precision_at_k": float(ranker_info["ranker_val_precision_at_k"]),
        "ranker_val_recall_at_k": float(ranker_info["ranker_val_recall_at_k"]),
        "ranker_val_ndcg_at_k": float(ranker_info["ranker_val_ndcg_at_k"]),
        "ranker_n_estimators": float(ranker_info["ranker_n_estimators"]),
        "ranker_max_depth": float(ranker_info["ranker_max_depth"]),
        "ranker_learning_rate": float(ranker_info["ranker_learning_rate"]),
        "ranker_min_child_weight": float(ranker_info["ranker_min_child_weight"]),
        "ranker_feature_count": float(ranker_info["ranker_feature_count"]),
        "ranking_als_rank": float(best_ranking_als_params["rank"]),
        "ranking_als_reg_param": float(best_ranking_als_params["regParam"]),
        "ranking_als_alpha": float(best_ranking_als_params["alpha"]),
        "ranking_als_max_iter": float(best_ranking_als_params["maxIter"]),
        "ranking_als_val_precision_at_k": float(best_ranking_als_params["val_precision"]),
        "ranking_als_val_recall_at_k": float(best_ranking_als_params["val_recall"]),
        "ranking_als_val_ndcg_at_k": float(best_ranking_als_params["val_ndcg"]),
        "ranking_als_val_candidate_recall": float(best_ranking_als_params["val_candidate_recall"]),
        "als_rank": float(best_als_params["rank"]),
        "als_reg_param": float(best_als_params["regParam"]),
        "als_max_iter": float(best_als_params["maxIter"]),
        "als_val_rmse": float(best_als_params["val_rmse"]),
        **dashboard_metrics,
    }
    run_record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    save_parquet(recommendation_export_df, output_paths["recommendations_parquet"])
    save_parquet(user_profiles_final_df, output_paths["user_profiles_parquet"])
    save_parquet(seen_interactions_df, output_paths["seen_interactions_parquet"])
    save_metrics(metrics, output_paths["metrics_json"])
    append_metrics_history(run_record, output_paths["metrics_history_jsonl"])
    metrics_df = spark.createDataFrame([(k, float(v)) for k, v in metrics.items()], ["metric", "value"])
    save_parquet(metrics_df, output_paths["metrics_parquet"])

    if save_recommendations_to_postgres:
        missing_postgres_config = not (postgres_jdbc_url and postgres_user and postgres_password)
        if missing_postgres_config:
            raise ValueError("PostgreSQL export requested but JDBC credentials are incomplete.")
        save_to_postgres(
            df=recommendation_export_df,
            table_name="movie_recommendations",
            jdbc_url=postgres_jdbc_url,
            user=postgres_user,
            password=postgres_password,
        )

    LOGGER.info("Pipeline finished metrics=%s", metrics)
    return {
        "recommendations_path": output_paths["recommendations_parquet"],
        "metrics_path": output_paths["metrics_json"],
        **metrics,
    }
