from __future__ import annotations

import json
import os
import shutil
from typing import Dict

from pyspark.sql import DataFrame

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _is_windows_without_hadoop() -> bool:
    return os.name == "nt" and not (os.getenv("HADOOP_HOME") or os.getenv("hadoop.home.dir"))


def _remove_existing_path(path: str) -> None:
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def save_parquet(
    df: DataFrame,
    path: str,
    mode: str = "overwrite",
    partition_cols: list[str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _is_windows_without_hadoop():
        if partition_cols:
            LOGGER.warning(
                "Partitioned parquet write requested at path=%s but Windows fallback does not support partitioning; writing a single parquet file instead.",
                path,
            )
        _remove_existing_path(path)
        df.toPandas().to_parquet(path, index=False)
        LOGGER.info("Saved parquet with pandas fallback path=%s", path)
        return

    writer = df.write.mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(path)
    LOGGER.info("Saved parquet path=%s", path)


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    LOGGER.info("Saved metrics JSON path=%s", path)


def append_metrics_history(record: Dict[str, float | str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Appended metrics history path=%s", path)
