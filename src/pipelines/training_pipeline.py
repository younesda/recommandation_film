from __future__ import annotations

import os
from typing import Dict

from pyspark.sql import DataFrame, SparkSession

from src.config.settings import PipelineSettings
from src.evaluation.mae import compute_mae
from src.evaluation.ndcg_at_k import compute_ndcg_at_k
from src.evaluation.precision_at_k import compute_precision_at_k
from src.evaluation.recall_at_k import compute_recall_at_k
from src.evaluation.rmse import compute_rmse
from src.ingestion.load_data import load_all_data
from src.models.als_model import (
    recommend_for_all_users_flat,
    recommend_for_users_flat,
    retrain_best_als,
    score_als,
    score_als_candidates,
    train_als_with_tuning,
)
from src.models.content_model import build_content_scores, generate_content_candidates
from src.models.hybrid_model import combine_hybrid_scores, select_top_k_recommendations
from src.preprocessing.clean_data import clean_movies, clean_ratings, clean_tags, time_based_split
from src.preprocessing.feature_engineering import (
    add_time_features,
    build_movie_genre_weights,
    build_tag_tfidf_features,
    encode_genres,
    filter_tags_to_training_window,
)
from src.preprocessing.user_profiles import build_user_genre_profiles, create_user_profiles
from src.storage.database import save_to_postgres
from src.storage.save_parquet import save_metrics, save_parquet
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _prepare_outputs(base_dir: str) -> Dict[str, str]:
    return {
        "recommendations_parquet": os.path.join(base_dir, "recommendations"),
        "user_profiles_parquet": os.path.join(base_dir, "user_profiles"),
        "seen_interactions_parquet": os.path.join(base_dir, "seen_interactions"),
        "metrics_json": os.path.join(base_dir, "metrics", "metrics.json"),
        "metrics_parquet": os.path.join(base_dir, "metrics", "metrics_parquet"),
        "als_model": os.path.join(base_dir, "models", "als"),
    }


def _attach_movie_metadata(recommendations_df: DataFrame, movies_df: DataFrame) -> DataFrame:
    return recommendations_df.join(
        movies_df.select("movieId", "title", "genres"),
        on="movieId",
        how="left",
    )


def _is_empty(df: DataFrame) -> bool:
    return len(df.head(1)) == 0


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


def _tune_hybrid_weights(
    cfg: PipelineSettings,
    als_model_train,
    val_df: DataFrame,
    seen_train_df: DataFrame,
    user_genre_profiles_train_df: DataFrame,
    movie_genre_weights_df: DataFrame,
    movie_tag_tfidf_train_df: DataFrame | None,
    user_tag_tfidf_train_df: DataFrame | None,
) -> Dict[str, float]:
    weight_candidates = [0.5, 0.6, 0.7, 0.8, 0.9]
    val_users_df = val_df.select("userId").distinct().cache()

    als_val_candidates_df = recommend_for_users_flat(als_model_train, val_users_df, 50).select("userId", "movieId").cache()
    content_val_candidates_df = (
        generate_content_candidates(
            user_genre_profiles_df=user_genre_profiles_train_df,
            movie_genre_weights_df=movie_genre_weights_df,
            seen_interactions_df=seen_train_df,
            k=50,
        )
        .select("userId", "movieId")
        .join(val_users_df, on="userId", how="inner")
        .cache()
    )

    merged_val_candidates_df = (
        als_val_candidates_df.unionByName(content_val_candidates_df)
        .dropDuplicates(["userId", "movieId"])
    )
    merged_val_candidates_df = _exclude_seen_interactions(merged_val_candidates_df, seen_train_df).cache()

    if _is_empty(merged_val_candidates_df):
        return {
            "als_weight": float(cfg.hybrid.als_weight),
            "content_weight": float(cfg.hybrid.content_weight),
            "val_precision": 0.0,
            "val_ndcg": 0.0,
        }

    als_val_scores_df = score_als_candidates(als_model_train, merged_val_candidates_df).cache()
    content_val_scores_df = build_content_scores(
        candidate_items_df=merged_val_candidates_df,
        user_genre_profiles_df=user_genre_profiles_train_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_tfidf_df=user_tag_tfidf_train_df,
        movie_tag_tfidf_df=movie_tag_tfidf_train_df,
    ).cache()

    best_choice = {
        "als_weight": float(cfg.hybrid.als_weight),
        "content_weight": float(cfg.hybrid.content_weight),
        "val_precision": -1.0,
        "val_ndcg": -1.0,
    }

    for als_weight in weight_candidates:
        content_weight = 1.0 - als_weight
        hybrid_val_df = combine_hybrid_scores(
            als_scores_df=als_val_scores_df,
            content_scores_df=content_val_scores_df,
            settings=cfg,
            als_weight=als_weight,
            content_weight=content_weight,
        )
        top_val_df = select_top_k_recommendations(hybrid_val_df, cfg.hybrid.top_k)

        precision_val = compute_precision_at_k(
            recommendations_df=top_val_df.select("userId", "movieId", "rank", "final_score"),
            ground_truth_df=val_df,
            k=cfg.hybrid.top_k,
            positive_threshold=cfg.min_positive_rating,
        )
        ndcg_val = compute_ndcg_at_k(
            recommendations_df=top_val_df.select("userId", "movieId", "rank", "final_score"),
            ground_truth_df=val_df,
            k=cfg.hybrid.top_k,
            positive_threshold=cfg.min_positive_rating,
        )

        is_better = (ndcg_val > best_choice["val_ndcg"]) or (
            ndcg_val == best_choice["val_ndcg"] and precision_val > best_choice["val_precision"]
        )
        if is_better:
            best_choice = {
                "als_weight": float(als_weight),
                "content_weight": float(content_weight),
                "val_precision": float(precision_val),
                "val_ndcg": float(ndcg_val),
            }

    return best_choice


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

    LOGGER.info("Pipeline started")
    ratings_df, movies_df, tags_df = load_all_data(spark=spark, settings=cfg, prefer_parquet=True)

    movies_clean = clean_movies(movies_df).cache()
    tags_clean = clean_tags(tags_df).cache()
    ratings_clean = clean_ratings(ratings_df, settings=cfg).cache()
    ratings_features = add_time_features(ratings_clean).cache()

    user_profiles_df = create_user_profiles(ratings_features).cache()
    movie_genres_df = encode_genres(movies_clean).cache()
    movie_genre_weights_df = build_movie_genre_weights(movie_genres_df).cache()

    train_df, val_df, test_df = time_based_split(ratings_features, settings=cfg)
    train_df = train_df.cache()
    val_df = val_df.cache()
    test_df = test_df.cache()

    als_model_train, best_als_params = train_als_with_tuning(train_df, val_df, settings=cfg)

    user_genre_profiles_train_df = build_user_genre_profiles(train_df, movie_genres_df).cache()
    seen_train_df = train_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).cache()
    tags_train_df = filter_tags_to_training_window(tags_clean, train_df).cache()
    movie_tag_tfidf_train_df, user_tag_tfidf_train_df = _build_tag_tfidf_if_available(use_tags, tags_train_df)

    best_hybrid = _tune_hybrid_weights(
        cfg=cfg,
        als_model_train=als_model_train,
        val_df=val_df,
        seen_train_df=seen_train_df,
        user_genre_profiles_train_df=user_genre_profiles_train_df,
        movie_genre_weights_df=movie_genre_weights_df,
        movie_tag_tfidf_train_df=movie_tag_tfidf_train_df,
        user_tag_tfidf_train_df=user_tag_tfidf_train_df,
    )

    als_model = retrain_best_als(train_df, val_df, best_als_params, settings=cfg)
    als_model.write().overwrite().save(output_paths["als_model"])

    train_val_df = train_df.unionByName(val_df).cache()
    seen_interactions_df = train_val_df.select("userId", "movieId").dropDuplicates(["userId", "movieId"]).cache()

    als_predictions_test = score_als(als_model, test_df)
    rmse = compute_rmse(als_predictions_test)
    mae = compute_mae(als_predictions_test)

    user_genre_profiles_final_df = build_user_genre_profiles(train_val_df, movie_genres_df).cache()
    tags_train_val_df = filter_tags_to_training_window(tags_clean, train_val_df).cache()
    movie_tag_tfidf_final_df, user_tag_tfidf_final_df = _build_tag_tfidf_if_available(use_tags, tags_train_val_df)

    als_candidates_df = recommend_for_all_users_flat(als_model, 50).select("userId", "movieId").cache()
    content_candidates_df = (
        generate_content_candidates(
            user_genre_profiles_df=user_genre_profiles_final_df,
            movie_genre_weights_df=movie_genre_weights_df,
            seen_interactions_df=seen_interactions_df,
            k=50,
        )
        .select("userId", "movieId")
        .cache()
    )

    merged_candidates_df = (
        als_candidates_df.unionByName(content_candidates_df)
        .dropDuplicates(["userId", "movieId"])
    )
    merged_candidates_df = _exclude_seen_interactions(merged_candidates_df, seen_interactions_df).cache()

    content_scores_df = build_content_scores(
        candidate_items_df=merged_candidates_df,
        user_genre_profiles_df=user_genre_profiles_final_df,
        movie_genre_weights_df=movie_genre_weights_df,
        user_tag_tfidf_df=user_tag_tfidf_final_df,
        movie_tag_tfidf_df=movie_tag_tfidf_final_df,
    ).cache()

    als_scores_df = score_als_candidates(als_model, merged_candidates_df).cache()
    hybrid_scores_df = combine_hybrid_scores(
        als_scores_df=als_scores_df,
        content_scores_df=content_scores_df,
        settings=cfg,
        als_weight=best_hybrid["als_weight"],
        content_weight=best_hybrid["content_weight"],
    )
    hybrid_scores_df = _exclude_seen_interactions(hybrid_scores_df, seen_interactions_df).cache()

    top_recommendations_df = select_top_k_recommendations(hybrid_scores_df, cfg.hybrid.top_k)
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
        "content_score",
        "final_score",
        "explanation",
    )

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        f"precision_at_{cfg.hybrid.top_k}": float(precision_k),
        f"recall_at_{cfg.hybrid.top_k}": float(recall_k),
        f"ndcg_at_{cfg.hybrid.top_k}": float(ndcg_k),
        "hybrid_als_weight": float(best_hybrid["als_weight"]),
        "hybrid_content_weight": float(best_hybrid["content_weight"]),
        "val_precision_at_k": float(best_hybrid["val_precision"]),
        "val_ndcg_at_k": float(best_hybrid["val_ndcg"]),
        "als_rank": float(best_als_params["rank"]),
        "als_reg_param": float(best_als_params["regParam"]),
        "als_max_iter": float(best_als_params["maxIter"]),
        "als_val_rmse": float(best_als_params["val_rmse"]),
    }

    save_parquet(recommendation_export_df, output_paths["recommendations_parquet"])
    save_parquet(user_profiles_df, output_paths["user_profiles_parquet"])
    save_parquet(seen_interactions_df, output_paths["seen_interactions_parquet"])
    save_metrics(metrics, output_paths["metrics_json"])
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
