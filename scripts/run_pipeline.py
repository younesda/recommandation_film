from __future__ import annotations

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.settings import (  # noqa: E402
    ALSSettings,
    DataPaths,
    HybridSettings,
    PipelineSettings,
)
from src.pipelines.training_pipeline import run_pipeline  # noqa: E402
from src.preprocessing.spark_session import create_spark  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end MovieLens hybrid recommendation pipeline.")
    parser.add_argument("--ratings-path", default="data/raw/ratings.csv")
    parser.add_argument("--movies-path", default="data/raw/movies.csv")
    parser.add_argument("--tags-path", default="data/raw/tags.csv")
    parser.add_argument("--output-base", default="data/processed")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-user-interactions", type=int, default=20)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    parser.add_argument("--disable-tags", action="store_true")
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--postgres-jdbc-url", default="")
    parser.add_argument("--postgres-user", default="")
    parser.add_argument("--postgres-password", default="")
    return parser.parse_args()


def main() -> None:
    configure_logging("INFO")
    args = parse_args()

    settings = PipelineSettings(
        shuffle_partitions=args.shuffle_partitions,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        data_paths=DataPaths(
            ratings=args.ratings_path,
            movies=args.movies_path,
            tags=args.tags_path,
            output_base=args.output_base,
        ),
        hybrid=HybridSettings(top_k=args.top_k),
        als=ALSSettings(),
    )

    spark = create_spark(settings=settings)
    try:
        result = run_pipeline(
            spark=spark,
            settings=settings,
            use_tags=not args.disable_tags,
            save_recommendations_to_postgres=args.postgres,
            postgres_jdbc_url=args.postgres_jdbc_url,
            postgres_user=args.postgres_user,
            postgres_password=args.postgres_password,
        )
    finally:
        spark.stop()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
