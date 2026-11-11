"""Leakage-safe feature engineering for demand forecasting.

Rule: a feature for day t may only use demand observed up to day t - horizon.
With horizon=1 this is a next-day forecast; with horizon=h the same row can be
predicted h days ahead without knowing anything in between.

Columns known in advance (calendar, planned promotions, seasonal index) are used
for the same day. Outcome columns (units sold, lost sales, inventory, orders) and
same-day competitor price are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_processing.schema import SERIES_KEYS

TARGET = "demand_units"
KNOWN_AHEAD = ["promotion_flag", "discount_pct", "holiday_flag", "day_of_week", "weekend_flag",
               "annual_seasonality_index", "list_price_aed"]


def build_features(df: pd.DataFrame, horizon: int = 1, lags: list[int] | None = None,
                   rolling_windows: list[int] | None = None, level_window: int | None = None) -> pd.DataFrame:
    """Return one row per store-product-day with features, target and keys.

    Rows without enough history for the longest lag are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    cfg = load_config()["features"]
    lags = lags or cfg["lags"]
    rolling_windows = rolling_windows or cfg["rolling_windows"]
    level_window = level_window or cfg["level_window"]

    d = df.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    g = d.groupby(SERIES_KEYS, sort=False)[TARGET]
    out = d[SERIES_KEYS + ["date", "category", TARGET] + KNOWN_AHEAD].copy()
    out["horizon"] = horizon

    # demand history, shifted so nothing newer than t - horizon is visible
    for lag in lags:
        out[f"demand_lag_{lag}"] = g.shift(horizon + lag - 1)
    base = g.shift(horizon)
    by = [d[k] for k in SERIES_KEYS]
    for w in rolling_windows:
        out[f"demand_mean_{w}"] = base.groupby(by, sort=False).transform(lambda s, w=w: s.rolling(w).mean())
    out[f"demand_std_{level_window}"] = base.groupby(by, sort=False).transform(
        lambda s: s.rolling(level_window).std())
    # same weekday, average of the last 4 visible occurrences
    k0 = int(np.ceil(horizon / 7))
    out["demand_same_dow_mean_4"] = sum(g.shift(7 * k) for k in range(k0, k0 + 4)) / 4

    # competitor price is only observed after the fact -> lagged ratio
    ratio = d["competitor_price_aed"] / d["list_price_aed"]
    out["competitor_price_ratio_lag"] = ratio.groupby(by, sort=False).shift(horizon)

    # calendar
    out["month"] = d["date"].dt.month
    doy = d["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # stable numeric ids taken from the id text (ST003 -> 3), so codes do not depend on which rows are present
    out["store_code"] = d["store_id"].str.extract(r"(\d+)", expand=False).astype(int)
    out["product_code"] = d["product_id"].str.extract(r"(\d+)", expand=False).astype(int)

    need = [f"demand_lag_{max(lags)}", f"demand_mean_{max(rolling_windows)}", f"demand_std_{level_window}",
            "demand_same_dow_mean_4", "competitor_price_ratio_lag"]
    return out.dropna(subset=need).reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model input columns (everything except keys, target and bookkeeping)."""
    drop = set(SERIES_KEYS + ["date", "category", TARGET, "horizon"])
    return [c for c in features.columns if c not in drop]


def chronological_split(features: pd.DataFrame, train_end: str | None = None, val_end: str | None = None):
    """Split by date: train <= train_end < validation <= val_end < test. No shuffling."""
    cfg = load_config()["split"]
    train_end = pd.Timestamp(train_end or cfg["train_end"])
    val_end = pd.Timestamp(val_end or cfg["val_end"])
    if not train_end < val_end:
        raise ValueError("train_end must be before val_end")
    train = features[features["date"] <= train_end]
    val = features[(features["date"] > train_end) & (features["date"] <= val_end)]
    test = features[features["date"] > val_end]
    return train, val, test
