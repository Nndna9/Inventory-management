import numpy as np
import pandas as pd
import pytest

from src.data_processing.clean import clean
from src.data_processing.loader import DataValidationError, load_raw, validate_schema
from src.data_processing.quality import run_quality_checks
from src.data_processing.schema import SCHEMA, data_dictionary
from src.services.data_service import baseline_kpis, current_state


def test_loads_expected_shape(raw):
    assert raw.shape == (73100, 32)
    assert raw.store_id.nunique() == 5 and raw.product_id.nunique() == 20
    assert pd.api.types.is_datetime64_any_dtype(raw.date)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError):
        load_raw(tmp_path / "nope.csv")


def test_missing_column_raises(raw):
    with pytest.raises(DataValidationError, match="demand_units"):
        validate_schema(raw.drop(columns=["demand_units"]))


def test_bad_date_raises(raw, tmp_path):
    bad = raw.head(5).copy()
    bad["date"] = bad["date"].astype(str)
    bad.loc[bad.index[0], "date"] = "not-a-date"
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_raw(p)


def test_no_error_level_quality_failures(raw):
    checks = run_quality_checks(raw)
    failed = checks[(~checks.passed) & (checks.severity == "error")]
    assert failed.empty, failed.to_string()


def test_quality_checks_detect_corruption(raw):
    bad = raw.copy()
    bad.loc[bad.index[10], "units_sold"] += 1  # breaks sold + lost = demand and the stock balance
    checks = run_quality_checks(bad).set_index("check")
    assert not checks.loc["units_sold + lost_sales = demand", "passed"]
    assert not checks.loc["closing = opening + received - sold", "passed"]


def test_clean_preserves_valid_data_and_raw_is_untouched(raw):
    before = raw.copy()
    out, log = clean(raw)
    pd.testing.assert_frame_equal(raw, before)  # original not mutated
    assert log["rows_in"] == log["rows_out"] == 73100
    assert {"stockout_flag", "inventory_position_units", "list_price_aed", "unit_margin_aed"} <= set(out.columns)


def test_clean_handles_missing_duplicates_and_negatives(raw):
    dirty = pd.concat([raw.head(50), raw.head(3)], ignore_index=True)  # 3 duplicates
    dirty["demand_units"] = dirty["demand_units"].astype(float)
    dirty.loc[10, "demand_units"] = np.nan
    dirty.loc[11, "units_sold"] = -5
    out, log = clean(dirty)
    assert log["exact_duplicates_dropped"] == 3
    assert log["missing_key_or_demand_dropped"] == 1
    assert log["negative_values_clipped"] == 1
    assert len(out) == 49 and (out.units_sold >= 0).all()


def test_list_price_recovers_pre_discount_price(cleaned):
    assert cleaned.groupby("product_id").list_price_aed.nunique().max() == 1


def test_data_dictionary_covers_every_column(cleaned):
    assert set(cleaned.columns) == set(data_dictionary().column)
    assert len(SCHEMA) == 32


def test_current_state_and_kpis_match_raw_numbers(cleaned):
    state = current_state(cleaned)
    assert len(state) == 100
    last = cleaned[cleaned.date == cleaned.date.max()]
    k = baseline_kpis(cleaned)
    assert k["inventory_units"] == int(last.closing_inventory_units.sum())
    assert k["total_lost"] == int(cleaned.lost_sales_units.sum()) == 26541
    assert k["fill_rate"] == pytest.approx(cleaned.units_sold.sum() / cleaned.demand_units.sum())
    assert (state.inventory_position_units >= state.closing_inventory_units).all()
