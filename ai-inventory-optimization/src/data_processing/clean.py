"""Cleaning and derived columns. The raw file is never modified; output goes to data/processed/."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.schema import NON_NEGATIVE, SERIES_KEYS


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe, log of what was changed).

    Rules (all counted in the log so nothing is changed silently):
      1. drop exact duplicate rows
      2. drop duplicate (store, product, date) keys, keeping the first
      3. drop rows with a missing key or missing demand
      4. clip negative quantities/prices to zero
      5. add derived columns
    """
    log: dict[str, int] = {"rows_in": len(df)}
    out = df.copy()

    n = len(out)
    out = out.drop_duplicates()
    log["exact_duplicates_dropped"] = n - len(out)

    n = len(out)
    out = out.drop_duplicates(subset=SERIES_KEYS + ["date"], keep="first")
    log["duplicate_keys_dropped"] = n - len(out)

    n = len(out)
    out = out.dropna(subset=SERIES_KEYS + ["date", "demand_units"])
    log["missing_key_or_demand_dropped"] = n - len(out)

    neg = 0
    for col in NON_NEGATIVE:
        mask = out[col] < 0
        neg += int(mask.sum())
        if mask.any():
            out.loc[mask, col] = 0
    log["negative_values_clipped"] = neg

    out["stockout_flag"] = (out["lost_sales_units"] > 0).astype("int8")
    out["inventory_position_units"] = out["closing_inventory_units"] + out["outstanding_order_units"]
    out["list_price_aed"] = (out["selling_price_aed"] / (1 - out["discount_pct"] / 100)).round(2)
    out["unit_margin_aed"] = (out["selling_price_aed"] - out["unit_cost_aed"]).round(2)

    out = out.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    log["rows_out"] = len(out)
    return out, log
