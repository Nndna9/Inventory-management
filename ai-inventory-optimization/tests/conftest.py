import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.clean import clean  # noqa: E402
from src.data_processing.loader import load_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean(raw)[0]


@pytest.fixture(scope="session")
def small(cleaned):
    """Two series - enough for feature tests, fast."""
    return cleaned[cleaned.store_id.eq("ST001") & cleaned.product_id.isin(["P001", "P002"])].copy()
