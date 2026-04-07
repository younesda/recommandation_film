from pyspark.sql import SparkSession

from src.config.settings import PipelineSettings
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def create_spark(settings: PipelineSettings | None = None) -> SparkSession:
    cfg = settings or PipelineSettings()
    LOGGER.info(
        "Creating Spark session app_name=%s shuffle_partitions=%s",
        cfg.app_name,
        cfg.shuffle_partitions,
    )

    spark = (
        SparkSession.builder.appName(cfg.app_name)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
