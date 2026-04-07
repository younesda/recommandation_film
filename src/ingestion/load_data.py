from __future__ import annotations

import os
from typing import Dict, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.utils import AnalysisException

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_hdfs_path(path: str) -> bool:
    return path.startswith("hdfs://")


def _path_exists_local(path: str) -> bool:
    if _is_hdfs_path(path):
        return True
    return os.path.exists(path)


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    normalized = _normalize_path(path)
    if not _path_exists_local(normalized):
        raise FileNotFoundError(f"Input CSV not found: {normalized}")

    LOGGER.info("Loading CSV path=%s", normalized)
    try:
        return spark.read.csv(normalized, header=True, inferSchema=True)
    except AnalysisException as exc:
        LOGGER.exception("Unable to read CSV path=%s", normalized)
        raise RuntimeError(f"Failed to read CSV file: {normalized}") from exc


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    normalized = _normalize_path(path)
    if not _path_exists_local(normalized):
        raise FileNotFoundError(f"Input Parquet not found: {normalized}")

    LOGGER.info("Loading Parquet path=%s", normalized)
    try:
        return spark.read.parquet(normalized)
    except AnalysisException as exc:
        LOGGER.exception("Unable to read Parquet path=%s", normalized)
        raise RuntimeError(f"Failed to read parquet file: {normalized}") from exc


def save_as_parquet(df: DataFrame, output_path: str, repartition: int | None = None) -> None:
    normalized = _normalize_path(output_path)
    writer_df = df.repartition(repartition) if repartition else df
    LOGGER.info("Writing Parquet path=%s repartition=%s", normalized, repartition)
    writer_df.write.mode("overwrite").parquet(normalized)


def _dataset_paths(settings: PipelineSettings, paths: Dict[str, str] | None) -> Dict[str, str]:
    if paths:
        return {name: _normalize_path(value) for name, value in paths.items()}
    return {name: _normalize_path(value) for name, value in settings.to_paths_dict().items()}


def load_all_data(
    spark: SparkSession,
    settings: PipelineSettings | None = None,
    paths: Dict[str, str] | None = None,
    prefer_parquet: bool = False,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    cfg = settings or PipelineSettings()
    resolved_paths = _dataset_paths(cfg, paths)
    parquet_base = _normalize_path(cfg.data_paths.raw_parquet_base)

    LOGGER.info("Starting data ingestion prefer_parquet=%s", prefer_parquet)

    def load_or_fallback(name: str) -> DataFrame:
        csv_path = resolved_paths[name]
        parquet_path = f"{parquet_base}/{name}"

        if prefer_parquet and _path_exists_local(parquet_path):
            return read_parquet(spark, parquet_path)

        df = read_csv(spark, csv_path)
        if prefer_parquet:
            save_as_parquet(df, parquet_path)
        return df

    ratings_df = load_or_fallback("ratings")
    movies_df = load_or_fallback("movies")
    tags_df = load_or_fallback("tags")

    LOGGER.info("Ingestion completed")
    return ratings_df, movies_df, tags_df
