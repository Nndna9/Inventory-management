import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import SCHEMA
from src.feature_engineering.features import build_features, chronological_split
from src.forecasting.dataset import model_frame
from src.forecasting.future import apply_intervals, calendar_reference, future_known, recursive_forecast
from src.forecasting.metrics import evaluate
from src.forecasting.models import fit_model, holt_winters_one_step, predict_model
from src.forecasting.train import run_training
from src.services import forecast_service as fs


def test_metrics_match_hand_calculation():
    m = evaluate([10, 20, 0, 30], [12, 18, 1, 27])
    assert m["mae"] == pytest.approx(2.0)            # (2 + 2 + 1 + 3) / 4
    assert m["rmse"] == pytest.approx(np.sqrt(4.5))  # sqrt((4 + 4 + 1 + 9) / 4)
    assert m["mape"] == pytest.approx((0.2 + 0.1 + 0.1) / 3)  # zero-demand row excluded
    assert m["wape"] == pytest.approx(8 / 60)
    assert m["bias"] == pytest.approx(-0.5)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        evaluate([1, 2], [1])
    with pytest.raises(ValueError):
        evaluate([], [])


def test_model_inputs_exclude_target_and_outcome_columns(small):
    X = model_frame(build_features(small))
    leaky = {c for c, (_, role, _) in SCHEMA.items() if role == "excluded"} | {"demand_units", "competitor_price_aed"}
    assert not set(X.columns) & leaky
    assert not X.isna().any().any()


@pytest.fixture(scope="module")
def small_model(small):
    train, val, _ = chronological_split(build_features(small))
    return fit_model("Linear Regression", train, val)


def test_predictions_are_non_negative_units(small, small_model):
    _, _, test = chronological_split(build_features(small))
    p = predict_model(small_model, test)
    assert len(p) == len(test) and (p >= 0).all()
    assert evaluate(test.demand_units, p)["wape"] < 0.5


def test_recursive_forecast_shape_and_dates(small, small_model):
    ref = calendar_reference(small)
    hist = small[small.date <= "2025-09-30"]
    fc = recursive_forecast(small_model, hist, 14, ref)
    assert len(fc) == 2 * 14 and sorted(fc.step.unique()) == list(range(1, 15))
    assert fc.date.min() == pd.Timestamp("2025-10-01") and fc.date.max() == pd.Timestamp("2025-10-14")
    assert (fc.forecast >= 0).all() and fc.forecast.notna().all()


def test_recursive_forecast_never_reads_future_demand(small, small_model):
    """Back-test mode receives the real future rows; changing their demand must not change the forecast."""
    ref = calendar_reference(small)
    hist = small[small.date <= "2025-09-30"]
    truth = small[(small.date > "2025-09-30") & (small.date <= "2025-10-10")]
    tampered = truth.assign(demand_units=9999, competitor_price_aed=9999.0)
    a = recursive_forecast(small_model, hist, 10, ref, known=truth)
    b = recursive_forecast(small_model, hist, 10, ref, known=tampered)
    pd.testing.assert_frame_equal(a, b)


def test_promotion_scenario_raises_forecast(small, small_model):
    ref = calendar_reference(small)
    base = recursive_forecast(small_model, small, 7, ref)
    promo = recursive_forecast(small_model, small, 7, ref, promotion=True, discount_pct=10)
    assert promo.forecast.sum() > base.forecast.sum()


def test_future_calendar_matches_dataset_conventions(small):
    ref = calendar_reference(small)
    f = future_known(small, pd.date_range("2026-01-01", periods=7), ref)
    day = f[f.product_id == "P001"].set_index("date")
    assert day.loc["2026-01-01", "holiday_flag"] == 1            # 1 January is a holiday in the data
    assert day.loc["2026-01-02", "weekend_flag"] == 1            # Friday
    assert day.loc["2026-01-04", "weekend_flag"] == 0            # Sunday
    jan1_2025 = small[small.date == "2025-01-01"].annual_seasonality_index.iloc[0]
    assert day.loc["2026-01-01", "annual_seasonality_index"] == pytest.approx(jan1_2025, abs=1e-6)
    assert f.demand_units.isna().all()


def test_intervals_bracket_forecast():
    fc = pd.DataFrame({"step": [1, 8, 40], "forecast": [10.0, 20.0, 30.0]})
    factors = pd.DataFrame({"week": [1, 2], "lower_factor": [0.7, 0.6], "upper_factor": [1.4, 1.5]})
    out = apply_intervals(fc, factors)
    assert out.lower.tolist() == pytest.approx([7.0, 12.0, 18.0])   # step 40 uses the last available week
    assert out.upper.tolist() == pytest.approx([14.0, 30.0, 45.0])


def test_holt_winters_is_one_step_ahead(small):
    one = small[small.product_id == "P001"]
    fit_end = pd.Timestamp("2025-06-30")
    a = holt_winters_one_step(one, fit_end)
    tampered = one.copy()
    tampered.loc[tampered.date >= "2025-08-01", "demand_units"] = 500
    b = holt_winters_one_step(tampered, fit_end)
    upto = a.date <= "2025-08-01"                                  # prediction FOR 1 Aug uses data to 31 Jul
    np.testing.assert_allclose(a.prediction[upto], b.prediction[upto])


def test_training_pipeline_on_small_data(small):
    out = run_training(small, save=False, verbose=False)
    m = out["metrics"]
    assert set(m.split) == {"validation", "test"} and m.model.nunique() == 8
    val = m[m.split == "validation"].sort_values("wape")
    assert out["meta"]["selected_model"] == val.iloc[0].model      # chosen on validation, not on test
    p = out["predictions"]
    assert p[p.split == "test"].date.min() > pd.Timestamp(out["meta"]["split"]["val_end"])
    assert not p.drop(columns=["lower", "upper"]).isna().any().any()
    assert len(out["future"]) == 2 * 90 and out["future"].date.min() == small.date.max() + pd.Timedelta(days=1)


# ---- saved artefacts from the full pipeline -------------------------------------------------

needs_artefacts = pytest.mark.skipif(not fs.artefacts_ready(), reason="run scripts/run_pipeline.py first")


@needs_artefacts
def test_saved_results_support_the_before_after_claim():
    meta = fs.load_meta()
    ba = fs.before_after(meta)
    assert ba["after"]["wape"] < ba["before"]["wape"]              # claim shown on the dashboard is true on test data
    assert ba["after"]["wape"] < meta["test_metrics_naive"]["wape"]
    assert 0.70 <= meta["interval"]["coverage_30_day_test"] <= 0.90  # 80% interval is roughly calibrated


@needs_artefacts
def test_saved_predictions_and_future_are_complete(cleaned):
    meta, preds, future = fs.load_meta(), fs.load_predictions(), fs.load_future()
    test = preds[preds.split == "test"]
    assert len(test) == 100 * 92 and test.date.min() == pd.Timestamp("2025-10-01")
    recomputed = evaluate(test.actual, test[meta["best_ml_model"]])["wape"]
    assert recomputed == pytest.approx(meta["test_metrics_after"]["wape"])   # report number = stored predictions
    assert len(future) == 100 * 90 and (future.lower <= future.forecast).all() and (future.forecast <= future.upper).all()
    assert fs.predicted_demand(future, cleaned, 30) == pytest.approx(future[future.step <= 30].forecast.sum())


@needs_artefacts
def test_live_forecast_matches_saved_forecast(cleaned):
    meta, future = fs.load_meta(), fs.load_future()
    live = fs.forecast_series(cleaned, fs.load_model(), meta, fs.load_factors(), "ST001", "P001", 14)
    saved = future[(future.store_id == "ST001") & (future.product_id == "P001") & (future.step <= 14)]
    np.testing.assert_allclose(live.forecast.to_numpy(), saved.forecast.to_numpy(), rtol=1e-6)


def test_forecast_unknown_series_raises(cleaned):
    if not fs.artefacts_ready():
        pytest.skip("no artefacts")
    with pytest.raises(ValueError):
        fs.forecast_series(cleaned, fs.load_model(), fs.load_meta(), fs.load_factors(), "ST999", "P001", 7)
