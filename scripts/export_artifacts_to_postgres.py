from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.settings import PipelineSettings  # noqa: E402
from src.preprocessing.spark_session import create_spark  # noqa: E402
from src.storage.database import save_to_postgres  # noqa: E402
from src.utils.logging_utils import configure_logging, get_logger  # noqa: E402


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed recommendation artifacts to PostgreSQL for Grafana.")
    parser.add_argument("--jdbc-url", required=True, help="JDBC URL like jdbc:postgresql://host:5432/db")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--recommendations-path", default="data/processed/recommendations")
    parser.add_argument("--metrics-path", default="data/processed/metrics/metrics_parquet")
    parser.add_argument("--seen-path", default="data/processed/seen_interactions")
    parser.add_argument("--rec-table", default="movie_recommendations")
    parser.add_argument("--metrics-table", default="recommender_metrics")
    parser.add_argument("--seen-table", default="user_seen_items")
    return parser.parse_args()


def main() -> None:
    configure_logging("INFO")
    args = parse_args()

    spark = create_spark(PipelineSettings())
    try:
        rec_df = spark.read.parquet(args.recommendations_path)
        metrics_df = spark.read.parquet(args.metrics_path)
        seen_df = spark.read.parquet(args.seen_path)

        save_to_postgres(
            df=rec_df,
            table_name=args.rec_table,
            jdbc_url=args.jdbc_url,
            user=args.user,
            password=args.password,
            mode="overwrite",
        )
        save_to_postgres(
            df=metrics_df,
            table_name=args.metrics_table,
            jdbc_url=args.jdbc_url,
            user=args.user,
            password=args.password,
            mode="overwrite",
        )
        save_to_postgres(
            df=seen_df,
            table_name=args.seen_table,
            jdbc_url=args.jdbc_url,
            user=args.user,
            password=args.password,
            mode="overwrite",
        )
        LOGGER.info("PostgreSQL export completed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
