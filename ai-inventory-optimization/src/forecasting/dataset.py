"""Turns the feature table into model inputs.

Demand differs 7x between series and grows year on year. Tree models cannot extrapolate a
level they have not seen, so models predict demand RELATIVE to the series' own trailing
28-day average ("level"), and the prediction is multiplied back by the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL = "demand_mean_28"
RELATIVE = ["demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28", "demand_mean_7",
            "demand_same_dow_mean_4"]
DIRECT = ["promotion_flag", "discount_pct", "holiday_flag", "day_of_week", "weekend_flag",
          "annual_seasonality_index", "competitor_price_ratio_lag", "month", "doy_sin", "doy_cos",
          "store_code", "product_code"]
ONE_HOT = ["day_of_week", "month", "store_code", "product_code"]  # for the linear model


def level(features: pd.DataFrame) -> pd.Series:
    return features[LEVEL].clip(lower=0.5)


def model_frame(features: pd.DataFrame) -> pd.DataFrame:
    lvl = level(features)
    X = pd.DataFrame(index=features.index)
    for c in RELATIVE:
        X[f"{c}_rel"] = features[c] / lvl
    X["demand_cv_28"] = features["demand_std_28"] / lvl
    X["log_level"] = np.log1p(lvl)
    X["log_list_price"] = np.log(features["list_price_aed"])
    for c in DIRECT:
        X[c] = features[c]
    return X


def target_ratio(features: pd.DataFrame) -> pd.Series:
    return features["demand_units"] / level(features)
