from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame


def compute_mae(predictions_df: DataFrame) -> float:
    evaluator = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction",
    )
    return float(evaluator.evaluate(predictions_df.dropna(subset=["prediction"])))
