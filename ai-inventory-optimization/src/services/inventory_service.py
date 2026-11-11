"""Access layer for safety stock, reorder points and stock status. Pages and tests use only this."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.schema import SERIES_KEYS
from src.inventory_calculations import safety_stock as ssm
from src.services import forecast_service as fs
from src.services.data_service import current_state

METHODS = {"dynamic": "Dynamic (AI forecast)", "classic": "Classic (historical average)"}
STATUS_ORDER = ["Out of stock", "Reorder now", "Below reorder point, order placed", "Healthy"]


def dynamic_available() -> bool:
    return fs.artefacts_ready()


def policy_table(clean: pd.DataFrame, service_level: float, method: str = "dynamic",
                 window_days: int | None = None) -> pd.DataFrame:
    """One row per store-product: inputs, safety stock, reorder point, stock position, status, order suggestion.

    method 'classic': D and sigma_D from history (all of it, or the last `window_days`).
    method 'dynamic': D from the demand forecast over the coming lead time, sigma_D from forecast errors.
    Both use observed lead times and the same formula.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    lead = ssm.lead_time_stats(clean)
    hist = ssm.demand_stats(clean, window_days=window_days)
    if method == "dynamic":
        if not dynamic_available():
            raise RuntimeError("Forecast artefacts missing; run scripts/run_pipeline.py")
        meta = fs.load_meta()
        demand = ssm.forecast_inputs(fs.load_future(), fs.load_predictions(), meta["best_ml_model"], lead)
    else:
        demand = hist
    p = ssm.compute_policy(demand, lead, service_level)

    # existing ("before AI") policy, expressed in the same terms
    all_hist = ssm.demand_stats(clean).rename(columns={"demand_mean": "hist_mean", "demand_std": "hist_std"})
    p = p.merge(all_hist[SERIES_KEYS + ["hist_mean", "hist_std"]], on=SERIES_KEYS)
    state = current_state(clean)
    p = p.merge(state[SERIES_KEYS + ["store_city", "product_name", "category", "unit_cost_aed", "closing_inventory_units",
                                     "on_order_units", "inventory_position_units", "reorder_point_baseline_units",
                                     "order_up_to_level_units", "avg_daily_demand_28d"]], on=SERIES_KEYS)
    p["baseline_safety_stock"] = p["reorder_point_baseline_units"] - p["hist_mean"] * p["lt_mean"]
    p["baseline_service_level"] = ssm.implied_service_level(p["reorder_point_baseline_units"], p["hist_mean"],
                                                            p["lt_mean"], p["hist_std"], p["lt_std"])
    p["rop_change"] = p["reorder_point"] - p["reorder_point_baseline_units"]

    # order suggestion: keep the existing cycle stock (order-up-to minus reorder point), move it with the new ROP
    cycle = p["order_up_to_level_units"] - p["reorder_point_baseline_units"]
    p["order_up_to"] = p["reorder_point"] + cycle
    need = p["inventory_position_units"] <= p["reorder_point"]
    p["recommended_order_qty"] = np.where(need, np.ceil(p["order_up_to"] - p["inventory_position_units"]), 0).astype(int)
    p["recommended_order_value_aed"] = p["recommended_order_qty"] * p["unit_cost_aed"]

    p["status"] = "Healthy"
    p.loc[p["closing_inventory_units"] <= p["reorder_point"], "status"] = "Below reorder point, order placed"
    p.loc[need, "status"] = "Reorder now"
    p.loc[p["closing_inventory_units"] == 0, "status"] = "Out of stock"
    p["days_of_cover"] = p["closing_inventory_units"] / p["demand_mean"]
    p["method"] = method
    return p


def summary(policy: pd.DataFrame) -> dict:
    """Totals for the 'Before AI vs After AI' panel."""
    return {
        "series": len(policy),
        "safety_stock_units": float(policy["safety_stock"].sum()),
        "baseline_safety_stock_units": float(policy["baseline_safety_stock"].sum()),
        "safety_stock_value_aed": float((policy["safety_stock"] * policy["unit_cost_aed"]).sum()),
        "baseline_safety_stock_value_aed": float((policy["baseline_safety_stock"] * policy["unit_cost_aed"]).sum()),
        "rop_units": float(policy["reorder_point"].sum()),
        "baseline_rop_units": float(policy["reorder_point_baseline_units"].sum()),
        "baseline_service_level_mean": float(policy["baseline_service_level"].mean()),
        "baseline_service_level_min": float(policy["baseline_service_level"].min()),
        "baseline_service_level_max": float(policy["baseline_service_level"].max()),
        "raised": int((policy["rop_change"] > 0.5).sum()),
        "lowered": int((policy["rop_change"] < -0.5).sum()),
        "reorder_now": int((policy["status"] == "Reorder now").sum()),
        "out_of_stock": int((policy["status"] == "Out of stock").sum()),
        "order_units": int(policy["recommended_order_qty"].sum()),
        "order_value_aed": float(policy["recommended_order_value_aed"].sum()),
        "baseline_reorder_now": int((policy["inventory_position_units"] <= policy["reorder_point_baseline_units"]).sum()),
    }
