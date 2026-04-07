from pyspark.sql import SparkSession
import os

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def create_spark(settings: PipelineSettings | None = None) -> SparkSession:
    cfg = settings or PipelineSettings()
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    LOGGER.info(
        "Creating Spark session app_name=%s shuffle_partitions=%s driver_memory=%s executor_memory=%s",
        cfg.app_name,
        cfg.shuffle_partitions,
        driver_memory,
        executor_memory,
    )

    spark = (
        SparkSession.builder.appName(cfg.app_name)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.logLevel", "error")
        .config("spark.sql.maxPlanStringLength", "10000")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
