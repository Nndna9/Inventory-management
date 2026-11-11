"""Loading and schema validation of the raw inventory dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import path
from src.data_processing.schema import SCHEMA, SERIES_KEYS


class DataValidationError(ValueError):
    """Raised when an input file cannot be used (missing columns, bad dates, empty)."""


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("File contains no rows.")


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and parse dates on an already-read table. Does not alter values."""
    df = df.copy()
    validate_schema(df)
    for col in ("date", "expected_arrival_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
    if df["date"].isna().any():
        raise DataValidationError(f"{int(df['date'].isna().sum())} rows have an unreadable date.")
    return df.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the ORIGINAL dataset only (never modified)."""
    csv_path = Path(csv_path) if csv_path else path("raw_csv")
    if not csv_path.exists():
        raise DataValidationError(f"Dataset not found: {csv_path}")
    return parse(pd.read_csv(csv_path))


def updates_dir() -> Path:
    return path("raw_csv").parent / "updates"


def load_all_raw() -> pd.DataFrame:
    """Original dataset plus any accepted update files in data/raw/updates/ (kept as separate files)."""
    frames = [load_raw()]
    if updates_dir().exists():
        frames += [parse(pd.read_csv(f)) for f in sorted(updates_dir().glob("*.csv"))]
    return pd.concat(frames, ignore_index=True).sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)


def load_processed() -> pd.DataFrame:
    p = path("processed")
    if not p.exists():
        raise DataValidationError("Processed data not found. Run: python scripts/run_pipeline.py")
    return pd.read_parquet(p)
