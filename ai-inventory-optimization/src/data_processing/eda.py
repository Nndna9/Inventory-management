"""Exploratory summaries. Pure functions returning dataframes (used by report and dashboard)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.data_processing.schema import SERIES_KEYS


def demand_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per store-product demand statistics."""
    g = df.groupby(SERIES_KEYS + ["product_name", "category"])["demand_units"]
    out = g.agg(mean="mean", std="std", min="min", max="max").reset_index()
    out["cv"] = out["std"] / out["mean"]
    out["variance_to_mean"] = out["std"] ** 2 / out["mean"]
    out["skew"] = g.apply(lambda s: stats.skew(s)).to_numpy()
    return out


def stockout_summary(df: pd.DataFrame, by: str = "product_id") -> pd.DataFrame:
    keys = [by] + (["product_name"] if by == "product_id" else [])
    out = df.groupby(keys).agg(
        demand=("demand_units", "sum"), sold=("units_sold", "sum"), lost=("lost_sales_units", "sum"),
        stockout_days=("stockout_flag", "sum"), days=("stockout_flag", "size")).reset_index()
    out["fill_rate"] = out["sold"] / out["demand"]
    out["stockout_day_rate"] = out["stockout_days"] / out["days"]
    return out.sort_values("fill_rate")


def lead_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Observed lead times per supplier vs the supplier parameters in the dataset."""
    o = df[df.order_quantity_units > 0]
    out = o.groupby("supplier_id")["lead_time_days_for_order"].agg(
        orders="count", observed_mean="mean", observed_std="std", min="min", max="max").reset_index()
    ref = df.groupby("supplier_id")[["supplier_avg_lead_time_days", "supplier_lead_time_std_days"]].first().reset_index()
    return out.merge(ref, on="supplier_id")


def driver_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Average demand relative to each series' own mean (1.00 = typical day), by driver value."""
    rel = df["demand_units"] / df.groupby(SERIES_KEYS)["demand_units"].transform("mean")
    frames = []
    for col in ["promotion_flag", "discount_pct", "holiday_flag", "weekend_flag", "day_of_week", "weather_condition"]:
        t = rel.groupby(df[col]).agg(relative_demand="mean", rows="size").reset_index()
        t.columns = ["value", "relative_demand", "rows"]
        t.insert(0, "driver", col)
        t["value"] = t["value"].astype(str)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def monthly_demand(df: pd.DataFrame) -> pd.DataFrame:
    m = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
    return m.groupby("month").agg(demand=("demand_units", "sum"), sold=("units_sold", "sum"),
                                  lost=("lost_sales_units", "sum")).reset_index()


def lead_time_demand_dependence(df: pd.DataFrame) -> float:
    """Correlation between an order's lead time and the prior 7-day demand of that series."""
    d = df.sort_values(SERIES_KEYS + ["date"])
    recent = d.groupby(SERIES_KEYS)["demand_units"].transform(lambda s: s.rolling(7).mean())
    o = d.order_quantity_units > 0
    # compare within supplier so supplier mix does not create a spurious correlation
    lt = d.loc[o, "lead_time_days_for_order"] - d.loc[o].groupby("supplier_id")["lead_time_days_for_order"].transform("mean")
    rd = recent[o] / d.loc[o].groupby(SERIES_KEYS)["demand_units"].transform("mean")
    ok = lt.notna() & rd.notna()
    return float(np.corrcoef(lt[ok], rd[ok])[0, 1])


def seasonality_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weekday profile, month profile): average demand relative to the overall average of `df`."""
    overall = df["demand_units"].mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd = (df.groupby("day_of_week")["demand_units"].mean() / overall).rename("relative_demand").reset_index()
    wd["weekday"] = wd["day_of_week"].map(lambda i: names[i])
    mo = (df.groupby(df["date"].dt.month)["demand_units"].mean() / overall).rename("relative_demand").reset_index()
    mo.columns = ["month", "relative_demand"]
    mo["month_name"] = pd.to_datetime(mo["month"], format="%m").dt.strftime("%b")
    return wd, mo
