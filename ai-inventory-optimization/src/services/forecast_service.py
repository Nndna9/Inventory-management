"""Access layer for forecasting results. Pages and tests read forecasts only through these functions."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.data_processing.schema import SERIES_KEYS
from src.forecasting.future import apply_intervals, recursive_forecast
from src.forecasting.metrics import evaluate
from src.forecasting.train import artefact


def artefacts_ready() -> bool:
    return all(artefact(n).exists() for n in ("model", "meta", "predictions", "factors", "future"))


def load_meta() -> dict:
    return json.loads(artefact("meta").read_text(encoding="utf-8"))


def load_model():
    return joblib.load(artefact("model"))


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(artefact("predictions"))


def load_future() -> pd.DataFrame:
    return pd.read_parquet(artefact("future"))


def load_factors() -> pd.DataFrame:
    return pd.read_parquet(artefact("factors"))


def metrics_table(meta: dict, split: str = "test") -> pd.DataFrame:
    m = pd.DataFrame(meta["metrics"])
    return m[m["split"] == split].drop(columns="split").sort_values("wape").reset_index(drop=True)


def before_after(meta: dict) -> dict:
    """Overall test-period accuracy: 28-day average (before) vs best ML model (after)."""
    b, a = meta["test_metrics_before"], meta["test_metrics_after"]
    return {"before_model": meta["before_ai_model"], "after_model": meta["best_ml_model"], "before": b, "after": a,
            "wape_change": a["wape"] / b["wape"] - 1, "mae_change": a["mae"] / b["mae"] - 1,
            "rmse_change": a["rmse"] / b["rmse"] - 1}


def series_predictions(preds: pd.DataFrame, store_id: str, product_id: str, split: str = "test") -> pd.DataFrame:
    return preds[(preds["store_id"] == store_id) & (preds["product_id"] == product_id)
                 & (preds["split"] == split)].sort_values("date")


def series_accuracy(preds: pd.DataFrame, meta: dict, store_id: str, product_id: str) -> dict:
    p = series_predictions(preds, store_id, product_id)
    return {"before": evaluate(p["actual"], p[meta["before_ai_model"]]),
            "after": evaluate(p["actual"], p[meta["best_ml_model"]])}


def forecast_series(clean: pd.DataFrame, model, meta: dict, factors: pd.DataFrame, store_id: str, product_id: str,
                    horizon: int, promotion: bool = False, discount_pct: float = 0.0) -> pd.DataFrame:
    """Live forecast for one store-product under a promotion scenario, with 80% interval."""
    hist = clean[(clean["store_id"] == store_id) & (clean["product_id"] == product_id)]
    if hist.empty:
        raise ValueError(f"No history for {store_id} / {product_id}")
    ref = {"holidays": set(meta["calendar_reference"]["holidays"]),
           "season": {int(k): v for k, v in meta["calendar_reference"]["season"].items()}}
    fc = recursive_forecast(model, hist, horizon, ref, promotion=promotion, discount_pct=discount_pct)
    return apply_intervals(fc, factors)


def predicted_demand(future: pd.DataFrame, series: pd.DataFrame, days: int = 30) -> float:
    """Total forecast demand over the next `days` for the given store-product keys (no-promotion scenario)."""
    f = future[future["step"] <= days].merge(series[SERIES_KEYS].drop_duplicates(), on=SERIES_KEYS)
    return float(f["forecast"].sum())
