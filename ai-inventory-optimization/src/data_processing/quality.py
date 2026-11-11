"""Data quality checks. One function, used by the report, the tests and the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.schema import NON_NEGATIVE, NULLABLE, SCHEMA, SERIES_KEYS


def _row(check, severity, violations, detail):
    return {"check": check, "severity": severity, "passed": violations == 0,
            "violations": int(violations), "detail": detail}


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """severity 'error' = breaks calculations; 'warning' = usable but handled explicitly."""
    d = df.sort_values(SERIES_KEYS + ["date"])
    g = d.groupby(SERIES_KEYS, sort=False)
    rows = []

    required = [c for c in SCHEMA if c not in NULLABLE]
    rows.append(_row("No missing values in required columns", "error",
                     d[required].isna().sum().sum(), "Order columns are blank on no-order days by design"))
    rows.append(_row("No duplicate rows", "error", d.duplicated().sum(), ""))
    rows.append(_row("One row per store-product-day", "error",
                     d.duplicated(subset=SERIES_KEYS + ["date"]).sum(), ""))

    n_days = d["date"].nunique()
    rows.append(_row("Complete daily panel (no gaps)", "error", (g.size() != n_days).sum(),
                     f"{g.ngroups} series x {n_days} days"))
    rows.append(_row("No negative quantities or prices", "error", (d[NON_NEGATIVE] < 0).sum().sum(), ""))

    rows.append(_row("units_sold + lost_sales = demand", "error",
                     (d.units_sold + d.lost_sales_units != d.demand_units).sum(), ""))
    avail = d.opening_inventory_units + d.received_units
    rows.append(_row("units_sold = min(demand, opening + received)", "error",
                     (d.units_sold != np.minimum(d.demand_units, avail)).sum(),
                     "Receipts arrive before the day's sales"))
    rows.append(_row("closing = opening + received - sold", "error",
                     (d.closing_inventory_units != avail - d.units_sold).sum(), ""))
    prev = g["closing_inventory_units"].shift(1)
    rows.append(_row("opening = previous day's closing", "error",
                     ((d.opening_inventory_units != prev) & prev.notna()).sum(), ""))

    ordered = d.order_quantity_units > 0
    rows.append(_row("Order placed exactly when trigger flag = 1", "error",
                     (ordered != (d.reorder_trigger_flag == 1)).sum(), ""))
    rows.append(_row("Lead time present exactly when an order is placed", "error",
                     (d.lead_time_days_for_order.notna() != ordered).sum(), ""))
    o = d[ordered]
    rows.append(_row("Lead times positive", "error", (o.lead_time_days_for_order <= 0).sum(),
                     f"range {o.lead_time_days_for_order.min():.0f}-{o.lead_time_days_for_order.max():.0f} days"))
    rows.append(_row("expected_arrival = order date + lead time", "error",
                     ((o.expected_arrival_date - o.date).dt.days != o.lead_time_days_for_order).sum(), ""))

    ip = d.closing_inventory_units + d.outstanding_order_units
    rows.append(_row("Order triggered exactly when inventory position <= reorder point", "error",
                     ((ip <= d.reorder_point_baseline_units) != ordered).sum(),
                     "Confirms the existing policy: daily-review (s, S)"))
    rows.append(_row("Order quantity = order-up-to level - inventory position", "error",
                     (o.order_up_to_level_units - (o.closing_inventory_units + o.outstanding_order_units)
                      != o.order_quantity_units).sum(), ""))

    rec = d.loc[d.received_units > 0, SERIES_KEYS + ["date", "received_units"]]
    due = o[o.expected_arrival_date <= d["date"].max()]
    m = due.merge(rec, left_on=SERIES_KEYS + ["expected_arrival_date"], right_on=SERIES_KEYS + ["date"],
                  how="left", suffixes=("", "_rec"))
    rows.append(_row("Every due order is received in full on its arrival date", "error",
                     (m.received_units_rec != m.order_quantity_units).sum(),
                     f"{int((o.expected_arrival_date > d['date'].max()).sum())} orders still in transit at end of data"))

    rows.append(_row("Reorder point and order-up-to level fixed per store-product", "warning",
                     (g.reorder_point_baseline_units.nunique() > 1).sum()
                     + (g.order_up_to_level_units.nunique() > 1).sum(), ""))
    rows.append(_row("Discount only when promotion flag = 1", "warning",
                     ((d.discount_pct > 0) & (d.promotion_flag == 0)).sum(), ""))
    promo_no_disc = int(((d.promotion_flag == 1) & (d.discount_pct == 0)).sum())
    rows.append(_row("Promotions always carry a discount", "warning", promo_no_disc,
                     "Promotion flag and discount are treated as separate features"))
    w = d.groupby(["date", "store_id"])["weather_condition"].nunique()
    rows.append(_row("Weather consistent within a city-day", "warning", (w > 1).sum(),
                     "Weather differs across products in the same city on the same day; excluded from models"))
    return pd.DataFrame(rows)
