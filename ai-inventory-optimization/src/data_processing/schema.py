"""Expected schema and data dictionary for the inventory dataset.

`role` tells the forecasting module how a column may be used:
  key        - identifies the series or the day
  target     - what we forecast
  known      - known in advance for future dates (calendar, planned promotions)
  lag_only   - only observed after the fact; may be used lagged, never same-day
  excluded   - outcome of the same day's demand; using it would leak the target
  policy     - replenishment policy / supplier parameters (inventory modules)
"""
from __future__ import annotations

import pandas as pd

# column: (kind, role, description)
SCHEMA: dict[str, tuple[str, str, str]] = {
    "date": ("date", "key", "Calendar day (daily grain, 2024-01-01 to 2025-12-31)"),
    "store_id": ("str", "key", "Store identifier (5 stores)"),
    "store_city": ("str", "key", "Store city (UAE)"),
    "product_id": ("str", "key", "Product identifier (20 products)"),
    "product_name": ("str", "key", "Product name"),
    "category": ("str", "known", "Product category (4 categories)"),
    "supplier_id": ("str", "policy", "Supplier serving this store-product (fixed per store-product)"),
    "unit_cost_aed": ("num", "policy", "Purchase cost per unit, AED (fixed per product)"),
    "selling_price_aed": ("num", "known", "Selling price after discount, AED"),
    "competitor_price_aed": ("num", "lag_only", "Competitor price that day, AED"),
    "promotion_flag": ("int", "known", "1 if a promotion ran that day (assumed planned in advance)"),
    "discount_pct": ("num", "known", "Discount percentage applied (0, 5, 10, 15, 20)"),
    "holiday_flag": ("int", "known", "1 on holiday dates"),
    "day_of_week": ("int", "known", "0 = Monday ... 6 = Sunday"),
    "weekend_flag": ("int", "known", "1 on Friday and Saturday"),
    "weather_condition": ("str", "excluded", "Weather label. Inconsistent within a city-day and no demand effect; not used"),
    "annual_seasonality_index": ("num", "known", "Deterministic seasonal index by calendar date"),
    "demand_units": ("int", "target", "Unconstrained customer demand, units. FORECAST TARGET"),
    "units_sold": ("int", "excluded", "Fulfilled sales = min(demand, available stock)"),
    "lost_sales_units": ("int", "excluded", "Demand not served = demand - units sold"),
    "opening_inventory_units": ("int", "excluded", "On-hand stock at start of day"),
    "received_units": ("int", "excluded", "Units received that day (before sales)"),
    "closing_inventory_units": ("int", "excluded", "On-hand stock at end of day"),
    "outstanding_order_units": ("int", "excluded", "Units on order, not yet received, before that day's ordering decision"),
    "reorder_point_baseline_units": ("int", "policy", "Existing reorder point (fixed per store-product)"),
    "order_up_to_level_units": ("int", "policy", "Existing order-up-to level (fixed per store-product)"),
    "reorder_trigger_flag": ("int", "excluded", "1 if an order was placed at end of day"),
    "order_quantity_units": ("int", "excluded", "Units ordered at end of day"),
    "lead_time_days_for_order": ("num", "policy", "Realised lead time of the order placed that day (blank if no order)"),
    "expected_arrival_date": ("date", "policy", "Order date + lead time (blank if no order)"),
    "supplier_avg_lead_time_days": ("int", "policy", "Supplier mean lead time, days"),
    "supplier_lead_time_std_days": ("num", "policy", "Supplier lead-time standard deviation, days"),
}

DERIVED: dict[str, tuple[str, str, str]] = {
    "stockout_flag": ("int", "excluded", "DERIVED: 1 if lost_sales_units > 0"),
    "inventory_position_units": ("int", "excluded", "DERIVED: closing stock + outstanding orders (before that day's order)"),
    "list_price_aed": ("num", "known", "DERIVED: pre-discount price = selling price / (1 - discount)"),
    "unit_margin_aed": ("num", "policy", "DERIVED: selling price - unit cost"),
}

SERIES_KEYS = ["store_id", "product_id"]
NON_NEGATIVE = [c for c, (k, _, _) in SCHEMA.items() if k in ("num", "int")]
NULLABLE = ["lead_time_days_for_order", "expected_arrival_date"]


def data_dictionary() -> pd.DataFrame:
    rows = [(c, k, r, d, "original") for c, (k, r, d) in SCHEMA.items()]
    rows += [(c, k, r, d, "derived") for c, (k, r, d) in DERIVED.items()]
    return pd.DataFrame(rows, columns=["column", "type", "role", "description", "source"])
