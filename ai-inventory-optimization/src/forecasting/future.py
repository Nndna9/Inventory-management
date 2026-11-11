"""Multi-day forecasting beyond the last observed day.

Method: recursive. Forecast day t+1, treat it as observed, rebuild features, forecast t+2, ...
The same `build_features` used in training is used here, so there is no train/serve mismatch.

Future inputs:
  calendar, weekend, holiday, seasonal index - derived from the date (known in advance)
  promotion / discount                     - scenario chosen by the user (default: none)
  competitor price                         - unknown; last observed ratio is carried forward
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.schema import SERIES_KEYS
from src.feature_engineering.features import build_features
from src.forecasting.models import predict_model

HISTORY_DAYS = 63  # enough for the 28-day lags and windows
NEEDED = SERIES_KEYS + ["date", "category", "demand_units", "promotion_flag", "discount_pct", "holiday_flag",
                        "day_of_week", "weekend_flag", "annual_seasonality_index", "list_price_aed",
                        "competitor_price_aed"]


def calendar_reference(clean: pd.DataFrame) -> dict:
    """Date-driven lookups learned from history: holiday month-days and seasonal index by day of year."""
    by_date = clean.groupby("date").agg(h=("holiday_flag", "first"), s=("annual_seasonality_index", "first")).reset_index()
    return {
        "holidays": set(by_date.loc[by_date.h == 1, "date"].dt.strftime("%m-%d")),
        "season": by_date.groupby(by_date["date"].dt.dayofyear)["s"].mean().to_dict(),
    }


def future_known(history: pd.DataFrame, dates: pd.DatetimeIndex, ref: dict, promotion: bool = False,
                 discount_pct: float = 0.0) -> pd.DataFrame:
    """Rows for future dates with every known-in-advance column filled; demand is left empty."""
    last = history.sort_values("date").groupby(SERIES_KEYS, sort=False).tail(1)
    base = last[SERIES_KEYS + ["category", "list_price_aed", "competitor_price_aed"]]
    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["weekend_flag"] = cal["day_of_week"].isin([4, 5]).astype(int)  # Friday, Saturday
    cal["holiday_flag"] = cal["date"].dt.strftime("%m-%d").isin(ref["holidays"]).astype(int)
    season = ref["season"]
    cal["annual_seasonality_index"] = cal["date"].dt.dayofyear.map(lambda d: season.get(d, season[min(d, 365)]))
    cal["promotion_flag"] = int(promotion)
    cal["discount_pct"] = float(discount_pct) if promotion else 0.0
    out = base.merge(cal, how="cross")
    out["demand_units"] = np.nan
    return out[NEEDED]


def recursive_forecast(model, history: pd.DataFrame, horizon: int, ref: dict, promotion: bool = False,
                       discount_pct: float = 0.0, known: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast `horizon` days after the last date in `history` for every series in it.

    `known` (optional) supplies the actual known-in-advance columns for the future dates;
    used for back-testing. Actual future demand and competitor prices are never read.
    Returns: store_id, product_id, date, step (1..horizon), forecast.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    origin = history["date"].max()
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    work = history.loc[history["date"] > origin - pd.Timedelta(days=HISTORY_DAYS), NEEDED].copy()
    work["demand_units"] = work["demand_units"].astype(float)
    fut = future_known(history, dates, ref, promotion, discount_pct)
    if known is not None:
        cols = ["promotion_flag", "discount_pct", "holiday_flag", "weekend_flag", "day_of_week", "annual_seasonality_index"]
        fut = fut.drop(columns=cols).merge(known[SERIES_KEYS + ["date"] + cols], on=SERIES_KEYS + ["date"], how="left")[NEEDED]
        if fut[cols].isna().any().any():
            raise ValueError("`known` does not cover every future date and series")

    results = []
    for step, day in enumerate(dates, start=1):
        work = pd.concat([work, fut[fut["date"] == day]], ignore_index=True)
        feats = build_features(work, horizon=1)
        today = feats[feats["date"] == day]
        pred = predict_model(model, today)
        idx = work.index[work["date"] == day]
        key = today.set_index(SERIES_KEYS).index
        lookup = pd.Series(pred, index=key)
        work.loc[idx, "demand_units"] = lookup.reindex(pd.MultiIndex.from_frame(work.loc[idx, SERIES_KEYS])).to_numpy()
        results.append(today[SERIES_KEYS + ["date"]].assign(step=step, forecast=pred))
    return pd.concat(results, ignore_index=True)


def backtest_recursive(model, clean: pd.DataFrame, origins: list[str], horizon: int, ref: dict) -> pd.DataFrame:
    """Recursive forecasts from several past origins, joined to what actually happened."""
    frames = []
    for origin in pd.to_datetime(origins):
        hist = clean[clean["date"] <= origin]
        truth = clean[(clean["date"] > origin) & (clean["date"] <= origin + pd.Timedelta(days=horizon))]
        if truth["date"].nunique() < horizon:
            continue
        fc = recursive_forecast(model, hist, horizon, ref, known=truth)
        fc = fc.merge(truth[SERIES_KEYS + ["date", "demand_units"]], on=SERIES_KEYS + ["date"])
        frames.append(fc.assign(origin=origin))
    return pd.concat(frames, ignore_index=True)


def interval_factors(backtest: pd.DataFrame, lower: float = 0.10, upper: float = 0.90) -> pd.DataFrame:
    """Empirical prediction-interval multipliers by forecast week: quantiles of actual / forecast."""
    b = backtest[backtest["forecast"] > 0].copy()
    b["week"] = (b["step"] - 1) // 7 + 1
    b["ratio"] = b["demand_units"] / b["forecast"]
    f = b.groupby("week")["ratio"].quantile([lower, upper]).unstack()
    f.columns = ["lower_factor", "upper_factor"]
    return f.reset_index()


def apply_intervals(forecast: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    out = forecast.copy()
    week = ((out["step"] - 1) // 7 + 1).clip(upper=int(factors["week"].max()))
    f = factors.set_index("week")
    out["lower"] = out["forecast"] * week.map(f["lower_factor"]).to_numpy()
    out["upper"] = out["forecast"] * week.map(f["upper_factor"]).to_numpy()
    return out
