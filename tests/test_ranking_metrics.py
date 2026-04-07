from __future__ import annotations

from pyspark.sql import SparkSession

from src.evaluation.ndcg_at_k import compute_ndcg_at_k
from src.evaluation.precision_at_k import compute_precision_at_k
from src.evaluation.recall_at_k import compute_recall_at_k


def test_precision_recall_ndcg_at_k_small_case() -> None:
    spark = SparkSession.builder.master("local[1]").appName("test-ranking-metrics").getOrCreate()
    try:
        recs = spark.createDataFrame(
            [
                (1, 10, 1, 0.9),
                (1, 20, 2, 0.8),
                (2, 30, 1, 0.7),
                (2, 40, 2, 0.6),
            ],
            ["userId", "movieId", "rank", "final_score"],
        )
        ground_truth = spark.createDataFrame(
            [
                (1, 10, 4.0),
                (1, 99, 5.0),
                (2, 30, 4.5),
                (2, 88, 4.0),
            ],
            ["userId", "movieId", "rating"],
        )

        precision = compute_precision_at_k(recs, ground_truth, k=2, positive_threshold=4.0)
        recall = compute_recall_at_k(recs, ground_truth, k=2, positive_threshold=4.0)
        ndcg = compute_ndcg_at_k(recs, ground_truth, k=2, positive_threshold=4.0)

        assert abs(precision - 0.5) < 1e-9
        assert abs(recall - 0.5) < 1e-9
        assert 0.0 < ndcg <= 1.0
    finally:
        spark.stop()
