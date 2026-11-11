"""Build all processed data, reports and models. Run from the project root:

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import rebuild  # noqa: E402

if __name__ == "__main__":
    rebuild()
