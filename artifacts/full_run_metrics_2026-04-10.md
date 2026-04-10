# Full Run Metrics - 2026-04-10

Best reference run obtained with the `full` Colab profile.

## Primary Metrics
- `RMSE`: `0.7620`
- `MAE`: `0.6051`
- `Precision@10`: `0.0386`
- `Recall@10`: `0.0730`
- `NDCG@10`: `0.0682`

## Ranking Highlights
- `candidate_pool_size`: `400`
- `val_candidate_recall`: `0.7782`
- `test_candidate_recall`: `0.7628`
- `ranker_val_ndcg_at_k`: `0.3591`

## Ranking Configuration
- `content_tag_weight`: `0.3`
- `ranking_als_alpha`: `10.0`
- `ranking_als_rank`: `96`
- `ranker_n_estimators`: `250`
- `ranker_max_depth`: `6`
- `ranker_learning_rate`: `0.1`
- `ranker_min_child_weight`: `5.0`

## Coverage
- `movies_recommended`: `692`
- `catalog_coverage_ratio`: `0.0710`
- `genres_covered`: `18`

## Notes
- This run clearly improves ranking over the older hybrid baseline.
- `NDCG@10` exceeded the `0.05` target.
- `Precision@10` and `Recall@10` improved strongly, but remain below the stretch targets `0.05` and `0.10`.
