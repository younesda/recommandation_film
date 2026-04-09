from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "eda_movielens.ipynb"


def _source(text: str) -> list[str]:
    normalized = dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def _markdown(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "id": cell_id,
        "source": _source(text),
    }


def _code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id},
        "id": cell_id,
        "execution_count": None,
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        _markdown(
            "md-title",
            """
            # MovieLens EDA + pipeline + API

            Ce notebook est proprement regenerable et executable de bout en bout sur Colab.
            Il couvre:
            - le setup Colab
            - le choix d'un profil d'execution `fast`, `balanced` ou `full`
            - une EDA rapide sur MovieLens
            - l'execution complete du pipeline
            - la verification des artefacts
            - un test API directement dans le notebook
            - un export zip pour recuperer facilement les resultats
            """,
        )
    )

    cells.append(
        _markdown(
            "md-setup",
            """
            ## 0) Setup Colab

            Cette cellule detecte Colab, clone le repo si besoin, installe Java 17 et les dependances Python.
            En local, elle se replace a la racine du projet.
            """,
        )
    )
    cells.append(
        _code(
            "code-setup",
            """
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            IN_COLAB = "google.colab" in sys.modules
            REPO_URL = "https://github.com/younesda/recommandation_film.git"

            def find_project_root(start: Path) -> Path:
                for candidate in [start, *start.parents]:
                    if (candidate / "src").exists() and (candidate / "scripts").exists() and (candidate / "data").exists():
                        return candidate
                return start

            REPO_DIR = Path("/content/recommandation_film") if IN_COLAB else find_project_root(Path.cwd())

            if IN_COLAB:
                if not REPO_DIR.exists():
                    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
                os.chdir(REPO_DIR)

                java_home = "/usr/lib/jvm/java-17-openjdk-amd64"
                java_bin = shutil.which("java")
                if (java_bin is None) or ("java-17" not in java_bin and "jdk-17" not in java_bin):
                    subprocess.run(["apt-get", "update", "-y"], check=True)
                    subprocess.run(["apt-get", "install", "-y", "openjdk-17-jdk-headless"], check=True)
                os.environ["JAVA_HOME"] = java_home
                os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]

                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
                os.environ.setdefault("SPARK_DRIVER_MEMORY", "6g")
                os.environ.setdefault("SPARK_EXECUTOR_MEMORY", "6g")
            else:
                os.chdir(REPO_DIR)
                os.environ.setdefault("SPARK_DRIVER_MEMORY", "4g")
                os.environ.setdefault("SPARK_EXECUTOR_MEMORY", "4g")

            os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
            os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
            os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
            os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

            print(f"IN_COLAB={IN_COLAB}")
            print(f"REPO_DIR={REPO_DIR}")
            print(f"PYTHON={sys.executable}")
            print(f"JAVA_HOME={os.getenv('JAVA_HOME', 'not-set')}")
            print(f"SPARK_DRIVER_MEMORY={os.environ['SPARK_DRIVER_MEMORY']}")
            print(f"SPARK_EXECUTOR_MEMORY={os.environ['SPARK_EXECUTOR_MEMORY']}")
            """,
        )
    )

    cells.append(
        _markdown(
            "md-imports",
            """
            ## 1) Imports et config projet

            On charge les modules du projet et quelques outils d'affichage pour piloter le run.
            """,
        )
    )
    cells.append(
        _code(
            "code-imports",
            """
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import display
            from pyspark.sql import functions as F

            from src.api.main import APP
            from src.config.settings import ALSSettings, DataPaths, HybridSettings, PipelineSettings, RankerSettings
            from src.ingestion.load_data import load_all_data
            from src.pipelines.training_pipeline import run_pipeline
            from src.preprocessing.clean_data import clean_movies, clean_ratings, clean_tags, time_based_split
            from src.preprocessing.feature_engineering import add_time_features
            from src.preprocessing.spark_session import create_spark
            from src.utils.logging_utils import configure_logging

            configure_logging("INFO")
            sns.set_theme(style="whitegrid")

            PROJECT_ROOT = Path.cwd()
            RAW_DIR = PROJECT_ROOT / "data" / "raw"
            PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

            try:
                GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
            except Exception:
                GIT_COMMIT = "unknown"

            print(f"PROJECT_ROOT={PROJECT_ROOT}")
            print(f"GIT_COMMIT={GIT_COMMIT}")
            """,
        )
    )

    cells.append(
        _markdown(
            "md-profile",
            """
            ## 2) Choix du profil d'execution

            Utilise `fast` pour iterer vite, `balanced` pour Colab au quotidien, et `full` seulement si tu acceptes un run long.
            """,
        )
    )
    cells.append(
        _code(
            "code-profile",
            """
            RUN_PROFILE = "full"

            PROFILE_CONFIGS = {
                "fast": {
                    "description": "Iteration rapide pour valider l'environnement et obtenir un premier score ranking",
                    "expected_runtime": "15-30 min",
                    "candidate_multiplier": 15,
                    "als_rank_candidates": [48],
                    "als_reg_param_candidates": [0.08],
                    "als_max_iter_candidates": [10],
                    "ranking_rank_candidates": [64],
                    "ranking_reg_param_candidates": [0.05],
                    "ranking_alpha_candidates": [10.0],
                    "ranking_max_iter_candidates": [10],
                    "ranker_n_estimators_candidates": [120],
                    "ranker_max_depth_candidates": [4],
                    "ranker_learning_rate_candidates": [0.08],
                    "ranker_min_child_weight_candidates": [1.0],
                },
                "balanced": {
                    "description": "Meilleur compromis qualite / temps pour Colab",
                    "expected_runtime": "35-70 min",
                    "candidate_multiplier": 20,
                    "als_rank_candidates": [48, 64],
                    "als_reg_param_candidates": [0.05, 0.1],
                    "als_max_iter_candidates": [10],
                    "ranking_rank_candidates": [64],
                    "ranking_reg_param_candidates": [0.03, 0.05],
                    "ranking_alpha_candidates": [10.0, 20.0],
                    "ranking_max_iter_candidates": [10],
                    "ranker_n_estimators_candidates": [150],
                    "ranker_max_depth_candidates": [4, 6],
                    "ranker_learning_rate_candidates": [0.05],
                    "ranker_min_child_weight_candidates": [1.0],
                },
                "full": {
                    "description": "Recherche plus exhaustive, utile seulement pour un run final long",
                    "expected_runtime": "90-150+ min",
                    "candidate_multiplier": 40,
                    "als_rank_candidates": [32, 48, 64, 96],
                    "als_reg_param_candidates": [0.03, 0.05, 0.08, 0.1, 0.12],
                    "als_max_iter_candidates": [10, 15],
                    "ranking_rank_candidates": [64, 96],
                    "ranking_reg_param_candidates": [0.01, 0.05, 0.1],
                    "ranking_alpha_candidates": [5.0, 10.0, 20.0],
                    "ranking_max_iter_candidates": [10, 15],
                    "ranker_n_estimators_candidates": [150, 250],
                    "ranker_max_depth_candidates": [4, 6],
                    "ranker_learning_rate_candidates": [0.05, 0.1],
                    "ranker_min_child_weight_candidates": [1.0, 5.0],
                },
            }

            if RUN_PROFILE not in PROFILE_CONFIGS:
                raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE}. Expected one of {list(PROFILE_CONFIGS)}")

            profile = PROFILE_CONFIGS[RUN_PROFILE]
            profile_df = pd.DataFrame(
                [
                    {"profile": name, **cfg}
                    for name, cfg in PROFILE_CONFIGS.items()
                ]
            )
            display(profile_df[["profile", "description", "expected_runtime", "candidate_multiplier"]])
            print(f"Selected profile: {RUN_PROFILE}")
            """,
        )
    )

    cells.append(
        _markdown(
            "md-params",
            """
            ## 3) Parametres du run

            Les hyperparametres sont derives du profil choisi ci-dessus. `full` est maintenant le defaut pour maximiser le score, au prix d'un run plus long.
            """,
        )
    )
    cells.append(
        _code(
            "code-params",
            """
            TOP_K = 10
            USE_TAGS = True
            MIN_USER_INTERACTIONS = 20
            MIN_ITEM_INTERACTIONS = 5
            SHUFFLE_PARTITIONS = 16

            settings = PipelineSettings(
                shuffle_partitions=SHUFFLE_PARTITIONS,
                min_user_interactions=MIN_USER_INTERACTIONS,
                min_item_interactions=MIN_ITEM_INTERACTIONS,
                data_paths=DataPaths(
                    ratings=str(RAW_DIR / "ratings.csv"),
                    movies=str(RAW_DIR / "movies.csv"),
                    tags=str(RAW_DIR / "tags.csv"),
                    output_base=str(PROCESSED_DIR),
                ),
                hybrid=HybridSettings(
                    top_k=TOP_K,
                    candidate_multiplier=profile["candidate_multiplier"],
                    als_candidate_overfetch_multiplier=3,
                ),
                als=ALSSettings(
                    rank_candidates=profile["als_rank_candidates"],
                    reg_param_candidates=profile["als_reg_param_candidates"],
                    max_iter_candidates=profile["als_max_iter_candidates"],
                    ranking_rank_candidates=profile["ranking_rank_candidates"],
                    ranking_reg_param_candidates=profile["ranking_reg_param_candidates"],
                    ranking_alpha_candidates=profile["ranking_alpha_candidates"],
                    ranking_max_iter_candidates=profile["ranking_max_iter_candidates"],
                ),
                ranker=RankerSettings(
                    n_estimators_candidates=profile["ranker_n_estimators_candidates"],
                    max_depth_candidates=profile["ranker_max_depth_candidates"],
                    learning_rate_candidates=profile["ranker_learning_rate_candidates"],
                    min_child_weight_candidates=profile["ranker_min_child_weight_candidates"],
                ),
            )

            params_df = pd.DataFrame(
                [
                    {"parameter": "run_profile", "value": RUN_PROFILE},
                    {"parameter": "expected_runtime", "value": profile["expected_runtime"]},
                    {"parameter": "ratings_path", "value": settings.data_paths.ratings},
                    {"parameter": "movies_path", "value": settings.data_paths.movies},
                    {"parameter": "tags_path", "value": settings.data_paths.tags},
                    {"parameter": "output_base", "value": settings.data_paths.output_base},
                    {"parameter": "top_k", "value": settings.hybrid.top_k},
                    {"parameter": "candidate_multiplier", "value": settings.hybrid.candidate_multiplier},
                    {"parameter": "als_candidate_overfetch_multiplier", "value": settings.hybrid.als_candidate_overfetch_multiplier},
                    {"parameter": "hybrid_weight_candidates", "value": settings.hybrid.hybrid_weight_candidates},
                    {"parameter": "tag_weight_candidates", "value": settings.hybrid.tag_weight_candidates},
                    {"parameter": "als_rank_candidates", "value": settings.als.rank_candidates},
                    {"parameter": "als_reg_param_candidates", "value": settings.als.reg_param_candidates},
                    {"parameter": "als_max_iter_candidates", "value": settings.als.max_iter_candidates},
                    {"parameter": "ranking_rank_candidates", "value": settings.als.ranking_rank_candidates},
                    {"parameter": "ranking_reg_param_candidates", "value": settings.als.ranking_reg_param_candidates},
                    {"parameter": "ranking_alpha_candidates", "value": settings.als.ranking_alpha_candidates},
                    {"parameter": "ranking_max_iter_candidates", "value": settings.als.ranking_max_iter_candidates},
                    {"parameter": "ranker_n_estimators_candidates", "value": settings.ranker.n_estimators_candidates},
                    {"parameter": "ranker_max_depth_candidates", "value": settings.ranker.max_depth_candidates},
                    {"parameter": "ranker_learning_rate_candidates", "value": settings.ranker.learning_rate_candidates},
                    {"parameter": "ranker_min_child_weight_candidates", "value": settings.ranker.min_child_weight_candidates},
                ]
            )
            display(params_df)
            """,
        )
    )

    cells.append(
        _markdown(
            "md-spark",
            """
            ## 4) Creation de la session Spark

            Cette session est celle utilisee ensuite pour l'EDA et le pipeline.
            """,
        )
    )
    cells.append(
        _code(
            "code-spark",
            """
            spark = create_spark(settings=settings)
            print(f"Spark version={spark.version}")
            print(f"Spark app name={spark.sparkContext.appName}")
            """,
        )
    )

    cells.append(
        _markdown(
            "md-load",
            """
            ## 5) Chargement des donnees brutes

            On charge les CSV MovieLens, avec mise en cache en parquet si besoin.
            """,
        )
    )
    cells.append(
        _code(
            "code-load",
            """
            ratings_df, movies_df, tags_df = load_all_data(spark=spark, settings=settings, prefer_parquet=True)

            raw_summary_df = pd.DataFrame(
                [
                    {"dataset": "ratings", "rows": ratings_df.count(), "columns": len(ratings_df.columns)},
                    {"dataset": "movies", "rows": movies_df.count(), "columns": len(movies_df.columns)},
                    {"dataset": "tags", "rows": tags_df.count(), "columns": len(tags_df.columns)},
                ]
            )

            display(raw_summary_df)
            display(ratings_df.limit(5).toPandas())
            display(movies_df.limit(5).toPandas())
            display(tags_df.limit(5).toPandas())
            """,
        )
    )

    cells.append(
        _markdown(
            "md-prep",
            """
            ## 6) Pretraitement et vue globale

            Ici on applique les nettoyages de base et on regarde la taille des splits train/val/test.
            """,
        )
    )
    cells.append(
        _code(
            "code-prep",
            """
            movies_clean_df = clean_movies(movies_df).cache()
            tags_clean_df = clean_tags(tags_df).cache()
            ratings_clean_df = clean_ratings(ratings_df, settings=settings).cache()
            ratings_features_df = add_time_features(ratings_clean_df).cache()

            train_df, val_df, test_df = time_based_split(ratings_features_df, settings=settings)
            train_df = train_df.cache()
            val_df = val_df.cache()
            test_df = test_df.cache()

            prep_summary_df = pd.DataFrame(
                [
                    {"metric": "active_users", "value": ratings_clean_df.select("userId").distinct().count()},
                    {"metric": "active_movies", "value": ratings_clean_df.select("movieId").distinct().count()},
                    {"metric": "clean_ratings_rows", "value": ratings_clean_df.count()},
                    {"metric": "clean_movies_rows", "value": movies_clean_df.count()},
                    {"metric": "clean_tags_rows", "value": tags_clean_df.count()},
                    {"metric": "train_rows", "value": train_df.count()},
                    {"metric": "val_rows", "value": val_df.count()},
                    {"metric": "test_rows", "value": test_df.count()},
                ]
            )

            display(prep_summary_df)
            display(ratings_features_df.select("userId", "movieId", "rating", "timestamp").limit(10).toPandas())
            """,
        )
    )

    cells.append(
        _markdown(
            "md-eda",
            """
            ## 7) EDA rapide

            Quelques graphiques simples pour voir la distribution des notes et les genres dominants.
            """,
        )
    )
    cells.append(
        _code(
            "code-eda",
            """
            ratings_pdf = ratings_clean_df.select("rating").toPandas()
            movies_pdf = movies_clean_df.select("movieId", "genres").toPandas()

            genre_counts = (
                movies_pdf["genres"]
                .fillna("")
                .str.split("|")
                .explode()
                .loc[lambda series: series.str.strip() != ""]
                .value_counts()
                .head(10)
                .reset_index()
            )
            genre_counts.columns = ["genre", "movie_count"]

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(ratings_pdf["rating"], bins=10, ax=axes[0])
            axes[0].set_title("Distribution des notes")
            axes[0].set_xlabel("rating")

            sns.barplot(data=genre_counts, x="movie_count", y="genre", ax=axes[1], palette="Blues_r")
            axes[1].set_title("Top 10 genres")
            axes[1].set_xlabel("movie_count")
            axes[1].set_ylabel("genre")

            plt.tight_layout()
            plt.show()

            display(genre_counts)
            """,
        )
    )

    cells.append(
        _markdown(
            "md-pipeline",
            """
            ## 8) Pipeline complet

            Cette cellule lance l'equivalent de `python scripts/run_pipeline.py`.
            Si le run est trop long, repasse temporairement sur `RUN_PROFILE = "balanced"` ou `RUN_PROFILE = "fast"`.
            """,
        )
    )
    cells.append(
        _code(
            "code-pipeline",
            """
            print(f"Running profile: {RUN_PROFILE} ({profile['expected_runtime']})")

            pipeline_result = run_pipeline(
                spark=spark,
                settings=settings,
                use_tags=USE_TAGS,
                save_recommendations_to_postgres=False,
            )

            metrics_df = (
                pd.DataFrame(
                    [{"metric": key, "value": value} for key, value in pipeline_result.items()]
                )
                .sort_values("metric")
                .reset_index(drop=True)
            )
            display(metrics_df)

            priority_metrics = [
                "rmse",
                "mae",
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "content_tag_weight",
                "val_candidate_recall",
                "test_candidate_recall",
                "ranking_als_val_candidate_recall",
                "ranker_val_ndcg_at_k",
                "ranking_als_alpha",
            ]
            display(metrics_df[metrics_df["metric"].isin(priority_metrics)].reset_index(drop=True))
            """,
        )
    )

    cells.append(
        _markdown(
            "md-artifacts",
            """
            ## 9) Verification des artefacts generes

            On relit les fichiers produits dans `data/processed`.
            """,
        )
    )
    cells.append(
        _code(
            "code-artifacts",
            """
            recommendations_path = PROCESSED_DIR / "recommendations"
            metrics_path = PROCESSED_DIR / "metrics" / "metrics.json"
            metrics_history_path = PROCESSED_DIR / "metrics" / "metrics_history.jsonl"
            seen_interactions_path = PROCESSED_DIR / "seen_interactions"

            artifact_df = pd.DataFrame(
                [
                    {"artifact": "recommendations_path", "path": str(recommendations_path), "exists": recommendations_path.exists()},
                    {"artifact": "metrics_path", "path": str(metrics_path), "exists": metrics_path.exists()},
                    {"artifact": "metrics_history_path", "path": str(metrics_history_path), "exists": metrics_history_path.exists()},
                    {"artifact": "seen_interactions_path", "path": str(seen_interactions_path), "exists": seen_interactions_path.exists()},
                ]
            )
            display(artifact_df)

            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics_payload = json.load(handle)

            metrics_file_df = (
                pd.DataFrame([{"metric": key, "value": value} for key, value in metrics_payload.items()])
                .sort_values("metric")
                .reset_index(drop=True)
            )
            display(metrics_file_df)

            recs_df = pd.read_parquet(recommendations_path)
            display(recs_df.head(20))

            if metrics_history_path.exists():
                history_df = pd.read_json(metrics_history_path, lines=True)
                display(history_df.tail(10))
            """,
        )
    )

    cells.append(
        _markdown(
            "md-api",
            """
            ## 10) Test API dans le notebook

            Ici on recharge l'API locale sur les artefacts generes puis on interroge quelques endpoints.
            """,
        )
    )
    cells.append(
        _code(
            "code-api",
            """
            from fastapi.testclient import TestClient

            os.environ["RECOMMENDATIONS_PATH"] = str(recommendations_path)
            os.environ["METRICS_PATH"] = str(metrics_path)
            os.environ["SEEN_INTERACTIONS_PATH"] = str(seen_interactions_path)
            os.environ["METRICS_HISTORY_PATH"] = str(metrics_history_path)

            client = TestClient(APP)
            client.post("/reload")

            health_payload = client.get("/health").json()
            summary_payload = client.get("/dashboard/summary").json()
            genres_payload = client.get("/dashboard/genres", params={"limit": 10}).json()
            movies_payload = client.get("/dashboard/movies", params={"limit": 10}).json()
            distribution_payload = client.get("/dashboard/final-score-distribution", params={"bins": 10}).json()

            sample_user_id = int(recs_df["userId"].iloc[0])
            recommendation_payload = client.get("/recommend", params={"user_id": sample_user_id, "k": TOP_K}).json()

            print(health_payload)
            display(pd.DataFrame(summary_payload.items(), columns=["metric", "value"]))
            display(pd.DataFrame(genres_payload["rows"]))
            display(pd.DataFrame(movies_payload["rows"]))
            display(pd.DataFrame(distribution_payload["rows"]))
            display(pd.DataFrame(recommendation_payload["recommendations"]))
            """,
        )
    )

    cells.append(
        _markdown(
            "md-export",
            """
            ## 11) Export zip des resultats

            Cette cellule prepare un bundle facile a telecharger depuis Colab avec le notebook, le code, les dashboards et les artefacts.
            """,
        )
    )
    cells.append(
        _code(
            "code-export",
            """
            import shutil

            export_dir = PROJECT_ROOT / "recommandation_film_colab_export"
            archive_base = PROJECT_ROOT / "recommandation_film_colab_export"

            if export_dir.exists():
                shutil.rmtree(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(PROJECT_ROOT / "notebooks" / "eda_movielens.ipynb", export_dir / "eda_movielens.ipynb")
            shutil.copy2(PROJECT_ROOT / "requirements.txt", export_dir / "requirements.txt")
            shutil.copy2(PROJECT_ROOT / "README.MD", export_dir / "README.MD")
            shutil.copytree(PROJECT_ROOT / "data" / "processed", export_dir / "data_processed")
            shutil.copytree(PROJECT_ROOT / "dashboards", export_dir / "dashboards")
            shutil.copytree(PROJECT_ROOT / "src", export_dir / "src")
            shutil.copytree(PROJECT_ROOT / "scripts", export_dir / "scripts")

            archive_path = shutil.make_archive(str(archive_base), "zip", export_dir)
            print(f"Export ready: {archive_path}")

            if IN_COLAB:
                from google.colab import files

                files.download(archive_path)
            """,
        )
    )

    cells.append(
        _markdown(
            "md-cleanup",
            """
            ## 12) Nettoyage

            Quand tu as fini, tu peux fermer proprement Spark avec la cellule suivante.
            """,
        )
    )
    cells.append(
        _code(
            "code-cleanup",
            """
            # spark.stop()
            """,
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "eda_movielens.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook rebuilt at {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()


