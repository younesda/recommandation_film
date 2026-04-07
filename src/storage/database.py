from __future__ import annotations

from typing import Dict

from pyspark.sql import DataFrame, SparkSession

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def save_to_postgres(
    df: DataFrame,
    table_name: str,
    jdbc_url: str,
    user: str,
    password: str,
    mode: str = "overwrite",
) -> None:
    properties: Dict[str, str] = {
        "user": user,
        "password": password,
        "driver": "org.postgresql.Driver",
    }
    LOGGER.info("Saving DataFrame to PostgreSQL table=%s mode=%s", table_name, mode)
    (
        df.write.mode(mode)
        .option("batchsize", "10000")
        .jdbc(url=jdbc_url, table=table_name, properties=properties)
    )


def read_from_postgres(
    spark: SparkSession,
    table_name: str,
    jdbc_url: str,
    user: str,
    password: str,
) -> DataFrame:
    properties: Dict[str, str] = {
        "user": user,
        "password": password,
        "driver": "org.postgresql.Driver",
    }
    LOGGER.info("Reading DataFrame from PostgreSQL table=%s", table_name)
    return spark.read.jdbc(url=jdbc_url, table=table_name, properties=properties)
