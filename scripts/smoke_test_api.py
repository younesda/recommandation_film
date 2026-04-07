from __future__ import annotations

import argparse
import os
import sys

from fastapi.testclient import TestClient


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.main import APP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test FastAPI endpoints using generated artifacts.")
    parser.add_argument("--recommendations-path", default="data/processed/recommendations")
    parser.add_argument("--metrics-path", default="data/processed/metrics/metrics.json")
    parser.add_argument("--seen-path", default="data/processed/seen_interactions")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RECOMMENDATIONS_PATH"] = args.recommendations_path
    os.environ["METRICS_PATH"] = args.metrics_path
    os.environ["SEEN_INTERACTIONS_PATH"] = args.seen_path

    client = TestClient(APP)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"/health failed: {health_resp.text}"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200, f"/metrics failed: {metrics_resp.text}"

    rec_resp = client.get("/recommend", params={"user_id": args.user_id, "k": args.k})
    assert rec_resp.status_code == 200, f"/recommend failed: {rec_resp.text}"

    payload = rec_resp.json()
    assert payload["user_id"] == args.user_id
    assert isinstance(payload["recommendations"], list)
    assert len(payload["recommendations"]) > 0
    print("API smoke test passed.")


if __name__ == "__main__":
    main()
