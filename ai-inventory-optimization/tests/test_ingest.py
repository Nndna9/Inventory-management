import pandas as pd
import pytest

from src.data_processing import ingest
from src.data_processing.ingest import append_new_data, validate_new_data
from src.data_processing.loader import DataValidationError
from src.data_processing.schema import SCHEMA


@pytest.fixture(scope="module")
def parts(raw):
    """Pretend the last 7 days have not arrived yet."""
    cut = raw.date.max() - pd.Timedelta(days=7)
    return raw[raw.date <= cut][list(SCHEMA)], raw[raw.date > cut][list(SCHEMA)]


def test_valid_new_data_is_accepted(parts):
    existing, new = parts
    ok = validate_new_data(new, existing)
    assert len(ok) == 700 and ok.date.min() == existing.date.max() + pd.Timedelta(days=1)


def test_overlapping_dates_rejected(parts):
    existing, _ = parts
    with pytest.raises(DataValidationError, match="already in the data"):
        validate_new_data(existing.tail(100), existing)


def test_gap_or_missing_series_rejected(parts):
    existing, new = parts
    with pytest.raises(DataValidationError, match="no gaps"):
        validate_new_data(new[new.date > new.date.min()], existing)          # skips a day
    with pytest.raises(DataValidationError, match="no gaps"):
        validate_new_data(new[new.product_id != "P001"], existing)           # a product is missing


def test_unknown_product_and_missing_column_rejected(parts):
    existing, new = parts
    with pytest.raises(DataValidationError, match="Unknown ids"):
        validate_new_data(new.assign(product_id="P999"), existing)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_new_data(new.drop(columns=["demand_units"]), existing)


def test_inconsistent_stock_rejected(parts):
    existing, new = parts
    bad = new.copy()
    bad.loc[bad.index[0], "opening_inventory_units"] += 10                   # no longer equals previous closing
    with pytest.raises(DataValidationError, match="violations"):
        validate_new_data(bad, existing)


def test_append_writes_separate_file_and_leaves_original_alone(parts, tmp_path, monkeypatch):
    existing, new = parts
    monkeypatch.setattr(ingest, "updates_dir", lambda: tmp_path / "updates")
    target = append_new_data(new, existing)
    assert target.parent == tmp_path / "updates" and target.name == "update_20251225_20251231.csv"
    assert len(pd.read_csv(target)) == 700
