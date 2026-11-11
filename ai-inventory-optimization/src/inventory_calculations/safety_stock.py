"""Safety stock and reorder point.

Formula (faculty slide):
    SS  = Z * sqrt( sigma_D^2 * L  +  D^2 * sigma_L^2 )
    ROP = D * L + SS

    Z        service-level factor (standard normal quantile of the cycle service level)
    D        average demand per day (units/day)
    sigma_D  standard deviation of daily demand (units/day)
    L        average lead time (days)
    sigma_L  standard deviation of lead time (days)

Units: demand is per DAY and lead time is in DAYS throughout.
Assumptions of the formula: daily demands independent, demand independent of lead time,
lead-time demand approximately normal. Phase 4 (Monte Carlo) checks the service level
actually achieved without these assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_processing.schema import SERIES_KEYS

MIN_DEMAND_OBS = 30   # fewer daily observations than this -> fallback for sigma_D
MIN_ORDERS = 10       # fewer observed orders than this -> fallback to supplier lead-time parameters


def z_score(service_level: float) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be in [0.5, 1)")
    return float(norm.ppf(service_level))


def safety_stock(z, sigma_d, lead_time, demand, sigma_l):
    """SS = Z * sqrt(sigma_D^2 * L + D^2 * sigma_L^2). Works on scalars or arrays."""
    for name, v in (("sigma_d", sigma_d), ("lead_time", lead_time), ("demand", demand), ("sigma_l", sigma_l)):
        if np.any(np.asarray(v, dtype=float) < 0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(np.asarray(lead_time, dtype=float) <= 0):
        raise ValueError("lead_time must be positive")
    return z * np.sqrt(np.square(sigma_d) * lead_time + np.square(demand) * np.square(sigma_l))


def reorder_point(demand, lead_time, ss):
    """ROP = D * L + SS."""
    return np.asarray(demand, dtype=float) * lead_time + ss


def implied_service_level(rop, demand, lead_time, sigma_d, sigma_l):
    """Cycle service level a given reorder point corresponds to under the same formula."""
    sd = np.sqrt(np.square(sigma_d) * lead_time + np.square(demand) * np.square(sigma_l))
    return norm.cdf((np.asarray(rop, dtype=float) - np.asarray(demand, dtype=float) * lead_time) / sd)


def demand_stats(clean: pd.DataFrame, as_of: pd.Timestamp | None = None, window_days: int | None = None) -> pd.DataFrame:
    """Per store-product mean and standard deviation of daily demand up to `as_of`.

    window_days=None uses all history; otherwise only the most recent `window_days`.
    Fallback (documented): a series with fewer than MIN_DEMAND_OBS days uses its own mean and
    the median coefficient of variation of its category for sigma_D; flagged in `demand_fallback`.
    """
    as_of = as_of or clean["date"].max()
    d = clean[clean["date"] <= as_of]
    if window_days:
        d = d[d["date"] > as_of - pd.Timedelta(days=window_days)]
    g = d.groupby(SERIES_KEYS + ["category"])["demand_units"]
    out = g.agg(demand_mean="mean", demand_std="std", demand_obs="count").reset_index()
    out["demand_fallback"] = out["demand_obs"] < MIN_DEMAND_OBS
    if out["demand_fallback"].any():
        full = clean[clean["date"] <= as_of].groupby(SERIES_KEYS + ["category"])["demand_units"].agg(["mean", "std"])
        cv = (full["std"] / full["mean"]).groupby("category").median()
        fb = out["demand_fallback"]
        out.loc[fb, "demand_std"] = out.loc[fb, "demand_mean"] * out.loc[fb, "category"].map(cv)
    return out.drop(columns="category")


def lead_time_stats(clean: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per store-product mean and standard deviation of observed lead times (orders placed up to `as_of`).

    Fallback (documented): fewer than MIN_ORDERS orders -> supplier parameters from the dataset.
    """
    as_of = as_of or clean["date"].max()
    d = clean[clean["date"] <= as_of]
    o = d[d["order_quantity_units"] > 0]
    obs = o.groupby(SERIES_KEYS)["lead_time_days_for_order"].agg(lt_mean="mean", lt_std="std", lt_orders="count")
    sup = d.groupby(SERIES_KEYS)[["supplier_id", "supplier_avg_lead_time_days", "supplier_lead_time_std_days"]].last()
    out = sup.join(obs).reset_index()
    out["lt_orders"] = out["lt_orders"].fillna(0).astype(int)
    out["lt_fallback"] = out["lt_orders"] < MIN_ORDERS
    fb = out["lt_fallback"]
    out.loc[fb, "lt_mean"] = out.loc[fb, "supplier_avg_lead_time_days"].astype(float)
    out.loc[fb, "lt_std"] = out.loc[fb, "supplier_lead_time_std_days"]
    return out


def forecast_inputs(future: pd.DataFrame, predictions: pd.DataFrame, model: str, lead: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking demand inputs ('dynamic' method).

    D       = average forecast demand per day over the coming lead time (rounded up to whole days)
    sigma_D = standard deviation of the model's out-of-sample one-day forecast errors for that series
    """
    f = future.merge(lead[SERIES_KEYS + ["lt_mean"]], on=SERIES_KEYS)
    f = f[f["step"] <= np.ceil(f["lt_mean"])]
    d = f.groupby(SERIES_KEYS)["forecast"].mean().rename("demand_mean")
    err = (predictions["actual"] - predictions[model])
    s = err.groupby([predictions[k] for k in SERIES_KEYS]).std().rename("demand_std")
    n = err.groupby([predictions[k] for k in SERIES_KEYS]).count().rename("demand_obs")
    out = pd.concat([d, s, n], axis=1).reset_index()
    out["demand_fallback"] = False
    return out


def compute_policy(demand: pd.DataFrame, lead: pd.DataFrame, service_level: float) -> pd.DataFrame:
    """Safety stock and reorder point per store-product from demand and lead-time statistics."""
    z = z_score(service_level)
    p = demand.merge(lead, on=SERIES_KEYS)
    p["service_level"] = service_level
    p["z"] = z
    p["safety_stock"] = safety_stock(z, p["demand_std"], p["lt_mean"], p["demand_mean"], p["lt_std"])
    p["reorder_point"] = reorder_point(p["demand_mean"], p["lt_mean"], p["safety_stock"])
    p["lead_time_demand"] = p["demand_mean"] * p["lt_mean"]
    return p


def service_level_curve(demand_mean, demand_std, lt_mean, lt_std, levels=None) -> pd.DataFrame:
    """Safety stock and reorder point across service levels for one item."""
    levels = np.asarray(levels if levels is not None else np.round(np.arange(0.80, 0.9951, 0.005), 4))
    z = norm.ppf(levels)
    ss = safety_stock(z, demand_std, lt_mean, demand_mean, lt_std)
    return pd.DataFrame({"service_level": levels, "z": z, "safety_stock": ss,
                         "reorder_point": reorder_point(demand_mean, lt_mean, ss)})


def rolling_reorder_point(series: pd.DataFrame, service_level: float, window_days: int, lt_mean: float,
                          lt_std: float) -> pd.DataFrame:
    """Reorder point recomputed every day from the trailing `window_days` of demand (one store-product)."""
    s = series.sort_values("date")
    mean = s["demand_units"].rolling(window_days).mean()
    std = s["demand_units"].rolling(window_days).std()
    out = pd.DataFrame({"date": s["date"], "demand_mean": mean, "demand_std": std}).dropna()
    out["safety_stock"] = safety_stock(z_score(service_level), out["demand_std"], lt_mean, out["demand_mean"], lt_std)
    out["reorder_point"] = reorder_point(out["demand_mean"], lt_mean, out["safety_stock"])
    return out
