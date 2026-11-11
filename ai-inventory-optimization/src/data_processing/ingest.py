"""Adding new data ("keep the possibility of adding new data").

New rows are validated against the existing history and stored as a separate file in
data/raw/updates/. The original dataset is never edited.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing.loader import DataValidationError, parse, updates_dir
from src.data_processing.quality import run_quality_checks
from src.data_processing.schema import SCHEMA, SERIES_KEYS


def validate_new_data(new: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Return the parsed new rows, or raise DataValidationError listing every problem found."""
    new = parse(new)[list(SCHEMA)]
    problems = []
    last = existing["date"].max()

    unknown_p = sorted(set(new["product_id"]) - set(existing["product_id"]))
    unknown_s = sorted(set(new["store_id"]) - set(existing["store_id"]))
    if unknown_p or unknown_s:
        problems.append(f"Unknown ids (no history to forecast from): products {unknown_p}, stores {unknown_s}")
    if new.duplicated(subset=SERIES_KEYS + ["date"]).any():
        problems.append("Duplicate store-product-day rows in the new file")
    if (new["date"] <= last).any():
        problems.append(f"{int((new['date'] <= last).sum())} rows are dated on or before {last:%Y-%m-%d}, "
                        "which is already in the data")
    else:
        expected = pd.date_range(last + pd.Timedelta(days=1), new["date"].max(), freq="D")
        series = existing[SERIES_KEYS].drop_duplicates()
        counts = new.groupby(SERIES_KEYS)["date"].nunique()
        if len(counts) != len(series) or (counts != len(expected)).any():
            problems.append(f"New data must contain every store-product for every day from {expected[0]:%Y-%m-%d} "
                            f"to {expected[-1]:%Y-%m-%d} with no gaps ({len(series)} series x {len(expected)} days)")
    if not problems:
        checks = run_quality_checks(pd.concat([existing[list(SCHEMA)], new], ignore_index=True))
        failed = checks[(~checks["passed"]) & (checks["severity"] == "error")]
        problems += [f"{r.check}: {r.violations} violations" for r in failed.itertuples()]
    if problems:
        raise DataValidationError("New data rejected:\n- " + "\n- ".join(problems))
    return new


def append_new_data(new: pd.DataFrame, existing: pd.DataFrame) -> Path:
    """Validate and store new rows as data/raw/updates/update_<from>_<to>.csv."""
    ok = validate_new_data(new, existing)
    updates_dir().mkdir(parents=True, exist_ok=True)
    target = updates_dir() / f"update_{ok['date'].min():%Y%m%d}_{ok['date'].max():%Y%m%d}.csv"
    ok.to_csv(target, index=False, date_format="%Y-%m-%d")
    return target
