"""Forecasting models: simple baselines ('before AI') and machine-learning models ('after AI')."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.data_processing.schema import SERIES_KEYS
from src.forecasting.dataset import ONE_HOT, level, model_frame, target_ratio

# name -> feature column that already is the forecast
BASELINES = {
    "Naive (yesterday)": "demand_lag_1",
    "Moving average (7 days)": "demand_mean_7",
    "Moving average (28 days)": "demand_mean_28",
    "Same weekday average (4 weeks)": "demand_same_dow_mean_4",
}
ML_MODELS = ["Linear Regression", "Random Forest", "XGBoost"]
STAT_MODEL = "Holt-Winters (weekly)"


def predict_baseline(name: str, features: pd.DataFrame) -> np.ndarray:
    return features[BASELINES[name]].to_numpy(dtype=float)


def make_model(name: str, seed: int = 42, n_estimators: int | None = None):
    if name == "Linear Regression":
        pre = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT)], remainder="passthrough")
        return Pipeline([("prep", pre), ("model", LinearRegression())])
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=n_estimators or 150, max_depth=12, min_samples_leaf=25,
                                     max_features=0.5, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=n_estimators or 800, learning_rate=0.03, max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=10, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, train: pd.DataFrame, val: pd.DataFrame | None = None, seed: int = 42,
              n_estimators: int | None = None):
    """Fit on `train`. For XGBoost, `val` (later in time than train) is used only for early stopping."""
    X, y, w = model_frame(train), target_ratio(train), level(train)
    model = make_model(name, seed, n_estimators)
    if name == "XGBoost" and val is not None and n_estimators is None:
        model.set_params(early_stopping_rounds=50)
        model.fit(X, y, sample_weight=w, eval_set=[(model_frame(val), target_ratio(val))], verbose=False)
    elif name == "Linear Regression":
        model.fit(X, y, model__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def predict_model(model, features: pd.DataFrame) -> np.ndarray:
    """Predicted demand in units (ratio x level), never negative."""
    return np.clip(model.predict(model_frame(features)) * level(features).to_numpy(), 0, None)


def holt_winters_one_step(clean: pd.DataFrame, fit_end: pd.Timestamp) -> pd.DataFrame:
    """Per-series additive Holt-Winters with weekly seasonality, evaluated one day ahead.

    Smoothing parameters are estimated on data up to `fit_end` only. They are then held fixed
    and the filter is run over the whole series; each fitted value uses only earlier days,
    so predictions after `fit_end` are genuine one-step-ahead forecasts.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    out = []
    for keys, g in clean.groupby(SERIES_KEYS, sort=False):
        s = g.set_index("date")["demand_units"].astype(float).asfreq("D")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(s[:fit_end], trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit()
            keep = {k: fit.params[k] for k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal",
                                               "damping_trend", "initial_level", "initial_trend")}
            full = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                        initialization_method="known", initial_level=keep["initial_level"],
                                        initial_trend=keep["initial_trend"],
                                        initial_seasonal=fit.params["initial_seasons"]).fit(
                smoothing_level=keep["smoothing_level"], smoothing_trend=keep["smoothing_trend"],
                smoothing_seasonal=keep["smoothing_seasonal"], damping_trend=keep["damping_trend"], optimized=False)
        out.append(pd.DataFrame({SERIES_KEYS[0]: keys[0], SERIES_KEYS[1]: keys[1], "date": s.index,
                                 "prediction": np.clip(full.fittedvalues.to_numpy(), 0, None)}))
    return pd.concat(out, ignore_index=True)
