import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import SCHEMA
from src.feature_engineering.features import TARGET, build_features, chronological_split, feature_columns


def test_lag_values_match_manual_lookup(small):
    f = build_features(small, horizon=1)
    s = small[small.product_id == "P001"].set_index("date").demand_units
    row = f[(f.product_id == "P001") & (f.date == pd.Timestamp("2024-06-15"))].iloc[0]
    assert row.demand_lag_1 == s[pd.Timestamp("2024-06-14")]
    assert row.demand_lag_7 == s[pd.Timestamp("2024-06-08")]
    assert row.demand_mean_7 == pytest.approx(s["2024-06-08":"2024-06-14"].mean())
    assert row.demand_same_dow_mean_4 == pytest.approx(
        np.mean([s[pd.Timestamp("2024-06-15") - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)]))


@pytest.mark.parametrize("horizon", [1, 7, 14])
def test_no_future_information_in_features(small, horizon):
    """Changing demand from day t - horizon + 1 onward must not change any feature of day t."""
    t = pd.Timestamp("2025-03-10")
    cutoff = t - pd.Timedelta(days=horizon)
    tampered = small.copy()
    tampered.loc[tampered.date > cutoff, TARGET] = 9999
    tampered.loc[tampered.date > cutoff, "competitor_price_aed"] = 9999.0
    a = build_features(small, horizon=horizon)
    b = build_features(tampered, horizon=horizon)
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a.loc[a.date == t, cols].reset_index(drop=True),
                                  b.loc[b.date == t, cols].reset_index(drop=True))


def test_features_exclude_leaky_columns(small):
    cols = set(feature_columns(build_features(small)))
    leaky = {c for c, (_, role, _) in SCHEMA.items() if role == "excluded"} | {TARGET, "competitor_price_aed"}
    assert not cols & leaky
    assert not any(c.startswith("weather") for c in cols)


def test_features_do_not_cross_series(small):
    f = build_features(small, horizon=1)
    first = f.groupby("product_id").date.min()
    # the longest lag is 28 days, so each series loses its own first 28 days
    assert (first == small.date.min() + pd.Timedelta(days=28)).all()
    assert not f[feature_columns(f)].isna().any().any()


def test_chronological_split_is_ordered_and_complete(small):
    f = build_features(small)
    train, val, test = chronological_split(f)
    assert train.date.max() < val.date.min() <= val.date.max() < test.date.min()
    assert len(train) + len(val) + len(test) == len(f)
    assert test.date.min() == pd.Timestamp("2025-10-01")


def test_invalid_inputs_raise(small):
    with pytest.raises(ValueError):
        build_features(small, horizon=0)
    with pytest.raises(ValueError):
        chronological_split(build_features(small), train_end="2025-10-01", val_end="2025-09-01")
