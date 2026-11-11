"""Single access layer for data-level facts. Dashboard pages and tests call these functions,
so a number on screen is always the number the backend computed."""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data_processing.schema import SERIES_KEYS


def as_of_date(df: pd.DataFrame) -> pd.Timestamp:
    setting = str(load_config()["as_of_date"])
    return df["date"].max() if setting == "latest" else min(pd.Timestamp(setting), df["date"].max())


def current_state(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per store-product: stock position at the close of the as-of day."""
    as_of = as_of or as_of_date(df)
    cols = SERIES_KEYS + ["store_city", "product_name", "category", "supplier_id", "unit_cost_aed",
                          "closing_inventory_units", "outstanding_order_units", "order_quantity_units",
                          "reorder_point_baseline_units", "order_up_to_level_units",
                          "supplier_avg_lead_time_days", "supplier_lead_time_std_days"]
    s = df.loc[df["date"] == as_of, cols].copy()
    s["on_order_units"] = s["outstanding_order_units"] + s["order_quantity_units"]  # after today's order
    s["inventory_position_units"] = s["closing_inventory_units"] + s["on_order_units"]
    s["on_hand_below_rop"] = s["closing_inventory_units"] <= s["reorder_point_baseline_units"]
    s["inventory_value_aed"] = s["closing_inventory_units"] * s["unit_cost_aed"]
    recent = df[(df["date"] <= as_of) & (df["date"] > as_of - pd.Timedelta(days=28))]
    avg = recent.groupby(SERIES_KEYS)["demand_units"].mean().rename("avg_daily_demand_28d").reset_index()
    s = s.merge(avg, on=SERIES_KEYS, how="left")
    s["days_of_cover"] = s["closing_inventory_units"] / s["avg_daily_demand_28d"]
    # status, most severe last so it wins
    s["status"] = "Healthy"
    s.loc[s["on_hand_below_rop"], "status"] = "Below reorder point, order placed"
    s.loc[s["on_hand_below_rop"] & (s["on_order_units"] == 0), "status"] = "Reorder needed"
    s.loc[s["on_hand_below_rop"] & (s["days_of_cover"] < s["supplier_avg_lead_time_days"]), "status"] = "Low cover"
    s.loc[s["closing_inventory_units"] == 0, "status"] = "Out of stock"
    return s.reset_index(drop=True)


def baseline_kpis(df: pd.DataFrame, as_of: pd.Timestamp | None = None, window_days: int = 30) -> dict:
    """Headline numbers for the existing ('before AI') policy, straight from the dataset."""
    as_of = as_of or as_of_date(df)
    hist = df[df["date"] <= as_of]
    recent = hist[hist["date"] > as_of - pd.Timedelta(days=window_days)]
    state = current_state(df, as_of)
    holding_rate = load_config()["costs"]["holding_rate_annual"]
    return {
        "as_of": as_of,
        "products": int(df["product_id"].nunique()),
        "stores": int(df["store_id"].nunique()),
        "series": int(len(state)),
        "inventory_units": int(state["closing_inventory_units"].sum()),
        "inventory_value_aed": float(state["inventory_value_aed"].sum()),
        "on_order_units": int(state["on_order_units"].sum()),
        "below_rop": int(state["on_hand_below_rop"].sum()),
        "out_of_stock_now": int((state["closing_inventory_units"] == 0).sum()),
        "low_cover": int((state["status"] == "Low cover").sum()),
        "reorder_needed": int((state["status"] == "Reorder needed").sum()),
        "total_demand": int(hist["demand_units"].sum()),
        "total_sold": int(hist["units_sold"].sum()),
        "total_lost": int(hist["lost_sales_units"].sum()),
        "fill_rate": float(hist["units_sold"].sum() / hist["demand_units"].sum()),
        "stockout_events": int(hist["stockout_flag"].sum()),
        "recent_window_days": window_days,
        "recent_demand": int(recent["demand_units"].sum()),
        "recent_stockout_events": int(recent["stockout_flag"].sum()),
        "recent_fill_rate": float(recent["units_sold"].sum() / recent["demand_units"].sum()),
        "lost_margin_aed": float((hist["lost_sales_units"] * hist["unit_margin_aed"]).sum()),
        "avg_inventory_value_aed": float((hist["closing_inventory_units"] * hist["unit_cost_aed"]).groupby(hist["date"]).sum().mean()),
        "annual_holding_cost_aed": float((hist["closing_inventory_units"] * hist["unit_cost_aed"]).groupby(hist["date"]).sum().mean() * holding_rate),
        "holding_rate_annual": holding_rate,
    }
