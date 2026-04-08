import os
import sys

from pyspark.sql import SparkSession

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def create_spark(settings: PipelineSettings | None = None) -> SparkSession:
    cfg = settings or PipelineSettings()
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    python_executable = os.getenv("PYSPARK_PYTHON", sys.executable)

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = os.getenv("PYSPARK_DRIVER_PYTHON", python_executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    LOGGER.info(
        "Creating Spark session app_name=%s shuffle_partitions=%s driver_memory=%s executor_memory=%s python=%s",
        cfg.app_name,
        cfg.shuffle_partitions,
        driver_memory,
        executor_memory,
        python_executable,
    )

    spark = (
        SparkSession.builder.appName(cfg.app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", os.environ["PYSPARK_DRIVER_PYTHON"])
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.logLevel", "error")
        .config("spark.sql.maxPlanStringLength", "10000")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
