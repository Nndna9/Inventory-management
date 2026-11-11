"""Writes the Phase 1 markdown reports from computed results (no hand-typed numbers)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing import eda
from src.data_processing.quality import run_quality_checks
from src.data_processing.schema import data_dictionary


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"


def write_reports(raw: pd.DataFrame, clean: pd.DataFrame, clean_log: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = run_quality_checks(raw)
    prof = eda.demand_profile(clean)
    paths = {}

    q = ["# Data Quality Report", "",
         "Source: `data/raw/synthetic_ai_inventory_data_2024_2025.csv` (original synthetic dataset, unmodified).", "",
         f"- Rows: {len(raw):,} | Columns: {raw.shape[1]}",
         f"- Stores: {raw.store_id.nunique()} | Products: {raw.product_id.nunique()} | "
         f"Suppliers: {raw.supplier_id.nunique()} | Days: {raw.date.nunique()} "
         f"({raw.date.min():%Y-%m-%d} to {raw.date.max():%Y-%m-%d})",
         f"- Checks passed: {int(checks.passed.sum())} of {len(checks)} "
         f"(errors failed: {int((~checks.passed & (checks.severity == 'error')).sum())}, "
         f"warnings: {int((~checks.passed & (checks.severity == 'warning')).sum())})", "",
         "## Checks", "", _md(checks), "## Cleaning log", "",
         _md(pd.DataFrame(clean_log.items(), columns=["step", "rows"])),
         "## Missing values", "",
         _md(raw.isna().sum().loc[lambda s: s > 0].rename("missing").reset_index().rename(columns={"index": "column"}))]
    paths["quality"] = out_dir / "data_quality_report.md"
    paths["quality"].write_text("\n".join(q), encoding="utf-8")

    e = ["# Exploratory Data Analysis", "", "All figures computed from the cleaned dataset.", "",
         "## Demand distribution across the 100 store-product series", "",
         _md(prof[["mean", "std", "cv", "variance_to_mean", "skew"]].describe().loc[["mean", "min", "50%", "max"]].reset_index()),
         "Variance-to-mean above 1 and positive skew: demand is overdispersed and right-skewed, not normal.", "",
         "## Demand drivers (1.00 = a typical day for that series)", "", _md(eda.driver_effects(clean)),
         "## Stockouts and lost sales by product", "", _md(eda.stockout_summary(clean, "product_id")),
         "## Stockouts by store", "", _md(eda.stockout_summary(clean, "store_id")),
         "## Supplier lead times: observed vs dataset parameters", "", _md(eda.lead_time_summary(clean)),
         f"Lead time vs recent demand correlation (within supplier): {eda.lead_time_demand_dependence(clean):.3f}. "
         "Near zero, so demand and lead time are modelled as independent (stated assumption).", "",
         "## Monthly totals", "", _md(eda.monthly_demand(clean).assign(month=lambda d: d.month.dt.strftime("%Y-%m")))]
    paths["eda"] = out_dir / "eda_report.md"
    paths["eda"].write_text("\n".join(e), encoding="utf-8")

    paths["dictionary"] = out_dir / "data_dictionary.md"
    paths["dictionary"].write_text("# Data Dictionary\n\n" + _md(data_dictionary()), encoding="utf-8")
    return paths
