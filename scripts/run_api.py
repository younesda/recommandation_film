from __future__ import annotations

import os
import sys

import uvicorn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    uvicorn.run("src.api.main:APP", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
