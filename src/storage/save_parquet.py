from __future__ import annotations

import json
import os
from typing import Dict

from pyspark.sql import DataFrame

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def save_parquet(
    df: DataFrame,
    path: str,
    mode: str = "overwrite",
    partition_cols: list[str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
