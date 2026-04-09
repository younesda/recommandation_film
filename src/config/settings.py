from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DataPaths:
    ratings: str = "data/raw/ratings.csv"
    movies: str = "data/raw/movies.csv"
    tags: str = "data/raw/tags.csv"
    raw_parquet_base: str = "data/processed/parquet/raw"
    output_base: str = "data/processed"


@dataclass(frozen=True)
class ALSSettings:
    rank_candidates: List[int] = field(default_factory=lambda: [32, 48, 64, 96])
    reg_param_candidates: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.08, 0.1, 0.12])
    max_iter_candidates: List[int] = field(default_factory=lambda: [10, 15])
    ranking_rank_candidates: List[int] = field(default_factory=lambda: [64, 96])
    ranking_reg_param_candidates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    ranking_alpha_candidates: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])
    ranking_max_iter_candidates: List[int] = field(default_factory=lambda: [10, 15])
    seed: int = 42


@dataclass(frozen=True)
class HybridSettings:
    als_weight: float = 0.7
    content_weight: float = 0.3
    top_k: int = 10
    candidate_multiplier: int = 40
    als_candidate_overfetch_multiplier: int = 3
    hybrid_weight_candidates: List[float] = field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    tag_weight_candidates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])


@dataclass(frozen=True)
class RankerSettings:
    holdout_user_modulo: int = 5
    min_training_groups: int = 25
    n_estimators_candidates: List[int] = field(default_factory=lambda: [150, 250])
    max_depth_candidates: List[int] = field(default_factory=lambda: [4, 6])
    learning_rate_candidates: List[float] = field(default_factory=lambda: [0.05, 0.1])
    min_child_weight_candidates: List[float] = field(default_factory=lambda: [1.0, 5.0])
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    objective: str = "rank:ndcg"
    eval_at_k: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class PipelineSettings:
    app_name: str = "MovieLensHybridRecommender"
    shuffle_partitions: int = 16
    min_user_interactions: int = 20
    min_item_interactions: int = 5
    min_positive_rating: float = 4.0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    data_paths: DataPaths = field(default_factory=DataPaths)
    als: ALSSettings = field(default_factory=ALSSettings)
    hybrid: HybridSettings = field(default_factory=HybridSettings)
    ranker: RankerSettings = field(default_factory=RankerSettings)

    def to_paths_dict(self) -> Dict[str, str]:
        return {
            "ratings": self.data_paths.ratings,
            "movies": self.data_paths.movies,
            "tags": self.data_paths.tags,
        }
