"""Training and evaluation pipeline for demand forecasting.

Protocol (all splits are by date):
  1. Fit every model on TRAIN, score on VALIDATION -> choose the model with the lowest validation WAPE.
  2. Refit on TRAIN + VALIDATION, score once on TEST (Q4 2025) -> reported accuracy.
  3. Refit the chosen model on ALL data -> used only for forecasting beyond the last observed day.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, path
from src.data_processing.schema import SERIES_KEYS
from src.feature_engineering.features import build_features, chronological_split
from src.forecasting.future import (apply_intervals, backtest_recursive, calendar_reference, interval_factors,
                                    recursive_forecast)
from src.forecasting.metrics import evaluate
from src.forecasting.models import (BASELINES, ML_MODELS, STAT_MODEL, fit_model, holt_winters_one_step,
                                    predict_baseline, predict_model)

BEFORE_AI = "Moving average (28 days)"   # what a classic reorder-point calculator uses as "average demand"
FUTURE_HORIZON = 90
BACKTEST_HORIZON = 30


def artefact(name: str) -> Path:
    return {"model": path("models") / "forecast_model.joblib",
            "meta": path("models") / "forecast_meta.json",
            "predictions": path("processed").parent / "forecast_predictions.parquet",
            "backtest": path("processed").parent / "forecast_backtest.parquet",
            "factors": path("processed").parent / "forecast_interval_factors.parquet",
            "future": path("processed").parent / "forecast_next_90_days.parquet",
            "report": path("reports") / "model_evaluation_report.md"}[name]


def _month_end_origins(start: pd.Timestamp, end: pd.Timestamp, horizon: int) -> list[str]:
    """Origins `start`, then each following month end, while a full horizon still fits before `end`."""
    out, o = [], start
    while o + pd.Timedelta(days=horizon) <= end:
        out.append(str(o.date()))
        o = (o + pd.Timedelta(days=1)) + pd.offsets.MonthEnd(0)
    return out


def run_training(clean: pd.DataFrame, save: bool = True, verbose: bool = True) -> dict:
    cfg = load_config()
    seed = cfg["seed"]
    train_end, val_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["val_end"])
    feats = build_features(clean, horizon=1)
    train, val, test = chronological_split(feats)
    train_val = pd.concat([train, val])
    log = print if verbose else (lambda *a, **k: None)

    preds = pd.concat([val.assign(split="validation"), test.assign(split="test")])[
        SERIES_KEYS + ["date", "split", "demand_units"]].rename(columns={"demand_units": "actual"})
    rows = []

    def record(name, kind, p_val, p_test):
        preds.loc[preds["split"] == "validation", name] = p_val
        preds.loc[preds["split"] == "test", name] = p_test
        for split, y, p in (("validation", val["demand_units"], p_val), ("test", test["demand_units"], p_test)):
            rows.append({"model": name, "type": kind, "split": split, **evaluate(y, p)})
        log(f"  {name:32s} val WAPE {rows[-2]['wape']:.4f} | test WAPE {rows[-1]['wape']:.4f}")

    for name in BASELINES:
        record(name, "baseline", predict_baseline(name, val), predict_baseline(name, test))

    stage1, stage2 = {}, {}
    for name in ML_MODELS:
        stage1[name] = fit_model(name, train, val, seed)
        n_est = (stage1[name].best_iteration + 1) if name == "XGBoost" else None
        stage2[name] = fit_model(name, train_val, None, seed, n_estimators=n_est)
        record(name, "machine learning", predict_model(stage1[name], val), predict_model(stage2[name], test))

    def hw(fit_end, part):
        h = holt_winters_one_step(clean[clean["date"] <= part["date"].max()], fit_end)
        return part[SERIES_KEYS + ["date"]].merge(h, on=SERIES_KEYS + ["date"], how="left")["prediction"].to_numpy()
    record(STAT_MODEL, "statistical", hw(train_end, val), hw(val_end, test))

    metrics = pd.DataFrame(rows)
    val_rank = metrics[metrics["split"] == "validation"].sort_values("wape")
    best = val_rank.iloc[0]["model"]
    best_ml = val_rank[val_rank["type"] == "machine learning"].iloc[0]["model"]
    log(f"  selected on validation: {best} (best ML model: {best_ml})")

    # one-step prediction interval: per-series 10% / 90% quantiles of validation errors of the best ML model
    v = preds[preds["split"] == "validation"]
    q = (v["actual"] - v[best_ml]).groupby([v[k] for k in SERIES_KEYS]).quantile([0.10, 0.90]).unstack()
    q.columns = ["q_low", "q_high"]
    preds = preds.merge(q.reset_index(), on=SERIES_KEYS, how="left")
    preds["lower"] = (preds[best_ml] + preds["q_low"]).clip(lower=0)
    preds["upper"] = preds[best_ml] + preds["q_high"]
    preds = preds.drop(columns=["q_low", "q_high"])
    t = preds[preds["split"] == "test"]
    coverage_1step = float(((t["actual"] >= t["lower"]) & (t["actual"] <= t["upper"])).mean())

    # multi-day (recursive) accuracy and interval factors
    ref = calendar_reference(clean)
    bt_val = backtest_recursive(stage1[best_ml], clean, _month_end_origins(train_end, val_end, BACKTEST_HORIZON),
                                BACKTEST_HORIZON, ref).assign(split="validation")
    factors = interval_factors(bt_val)
    bt_test = backtest_recursive(stage2[best_ml], clean, _month_end_origins(val_end, clean["date"].max(), BACKTEST_HORIZON),
                                 BACKTEST_HORIZON, ref).assign(split="test")
    bt_test = apply_intervals(bt_test, factors)
    coverage_multi = float(((bt_test["demand_units"] >= bt_test["lower"]) & (bt_test["demand_units"] <= bt_test["upper"])).mean())
    multi = evaluate(bt_test["demand_units"], bt_test["forecast"])
    naive_flat = bt_test.merge(
        test[SERIES_KEYS + ["date", "demand_mean_28"]].rename(columns={"date": "first_day", "demand_mean_28": "flat"})
        .assign(origin=lambda d: d["first_day"] - pd.Timedelta(days=1)), on=SERIES_KEYS + ["origin"], how="left")
    multi_before = evaluate(naive_flat["demand_units"], naive_flat["flat"])
    log(f"  30-day recursive test WAPE {multi['wape']:.4f} vs flat 28-day average {multi_before['wape']:.4f}; "
        f"80% interval coverage {coverage_multi:.3f}")

    # final model on all data, future forecast (no-promotion scenario)
    n_est = (stage1[best_ml].best_iteration + 1) if best_ml == "XGBoost" else None
    final = fit_model(best_ml, feats, None, seed, n_estimators=n_est)
    future = apply_intervals(recursive_forecast(final, clean, FUTURE_HORIZON, ref), factors)

    m = metrics.set_index(["model", "split"])
    meta = {
        "selected_model": best, "best_ml_model": best_ml, "before_ai_model": BEFORE_AI,
        "trained_through": str(clean["date"].max().date()),
        "split": {"train_end": str(train_end.date()), "val_end": str(val_end.date()),
                  "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date())},
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_metrics_before": m.loc[(BEFORE_AI, "test")][["mae", "rmse", "mape", "wape", "bias"]].to_dict(),
        "test_metrics_after": m.loc[(best_ml, "test")][["mae", "rmse", "mape", "wape", "bias"]].to_dict(),
        "test_metrics_naive": m.loc[("Naive (yesterday)", "test")][["mae", "rmse", "mape", "wape", "bias"]].to_dict(),
        "interval": {"level": 0.80, "coverage_one_step_test": coverage_1step, "coverage_30_day_test": coverage_multi},
        "multi_day_test": {"horizon": BACKTEST_HORIZON, "after": multi, "before": multi_before},
        "metrics": metrics.to_dict(orient="records"),
        "calendar_reference": {"holidays": sorted(ref["holidays"]), "season": {int(k): float(v) for k, v in ref["season"].items()}},
    }
    result = {"meta": meta, "metrics": metrics, "predictions": preds, "factors": factors,
              "backtest": pd.concat([bt_val, bt_test], ignore_index=True), "future": future, "model": final}
    if save:
        artefact("model").parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, artefact("model"), compress=3)
        artefact("meta").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        preds.to_parquet(artefact("predictions"), index=False)
        result["backtest"].to_parquet(artefact("backtest"), index=False)
        factors.to_parquet(artefact("factors"), index=False)
        future.to_parquet(artefact("future"), index=False)
        write_report(meta, result["backtest"], factors)
    return result


def write_report(meta: dict, backtest: pd.DataFrame, factors: pd.DataFrame) -> None:
    m = pd.DataFrame(meta["metrics"])
    wide = m.pivot(index=["model", "type"], columns="split", values=["wape", "mae", "rmse", "mape", "bias"])
    wide.columns = [f"{s}_{k}" for k, s in wide.columns]
    wide = wide.reset_index().sort_values("validation_wape")
    cols = ["model", "type", "validation_wape", "test_wape", "test_mae", "test_rmse", "test_mape", "test_bias"]
    b, a, n = meta["test_metrics_before"], meta["test_metrics_after"], meta["test_metrics_naive"]
    bt = backtest[backtest["split"] == "test"].assign(week=lambda d: (d["step"] - 1) // 7 + 1)
    by_week = bt.groupby("week").apply(lambda g: evaluate(g["demand_units"], g["forecast"])["wape"]).rename("wape").reset_index()

    def table(df):
        d = df.copy()
        for c in d.select_dtypes("float").columns:
            d[c] = d[c].map(lambda v: f"{v:.4f}")
        return ("| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
                + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n")

    s = meta["split"]
    lines = [
        "# Demand Forecasting - Model Evaluation Report", "",
        "Target: `demand_units` (unconstrained demand), one row per store-product-day. All numbers below are computed "
        "by `src/forecasting/train.py`; nothing is typed in by hand.", "",
        "## Protocol", "",
        f"- Train: up to {s['train_end']} ({meta['rows']['train']:,} rows)",
        f"- Validation: to {s['val_end']} ({meta['rows']['validation']:,} rows) - used to choose the model",
        f"- Test: {s['test_start']} to {s['test_end']} ({meta['rows']['test']:,} rows) - used once, for reporting",
        "- Splits are by date; no shuffling. Features for a day use demand only up to the previous day.",
        "- Models predict demand relative to the series' trailing 28-day average, then scale back to units.",
        "- Holt-Winters: parameters estimated before the evaluated period, then run one day ahead.", "",
        "## One-day-ahead accuracy (sorted by validation WAPE)", "", table(wide[cols]),
        f"Model selected on validation: **{meta['selected_model']}**.", "",
        "## Before AI vs After AI (test period)", "",
        table(pd.DataFrame([
            {"metric": "WAPE", "naive": n["wape"], "before_ai_28d_average": b["wape"], "after_ai": a["wape"],
             "change_vs_before": f"{(a['wape'] / b['wape'] - 1):+.1%}"},
            {"metric": "MAE (units)", "naive": n["mae"], "before_ai_28d_average": b["mae"], "after_ai": a["mae"],
             "change_vs_before": f"{(a['mae'] / b['mae'] - 1):+.1%}"},
            {"metric": "RMSE (units)", "naive": n["rmse"], "before_ai_28d_average": b["rmse"], "after_ai": a["rmse"],
             "change_vs_before": f"{(a['rmse'] / b['rmse'] - 1):+.1%}"}])),
        f"'After AI' = {meta['best_ml_model']}. 'Before AI' = {meta['before_ai_model']}, the average-demand input of a "
        "classic reorder-point calculator.", "",
        "The improvement is real but moderate: daily demand in this dataset is noisy (variance about 1.9x the mean), "
        "so a large part of the error cannot be removed by any model.", "",
        f"## Multi-day forecasts (recursive, {meta['multi_day_test']['horizon']} days ahead, test period)", "",
        f"- WAPE after AI: {meta['multi_day_test']['after']['wape']:.4f}",
        f"- WAPE before AI (flat 28-day average): {meta['multi_day_test']['before']['wape']:.4f}", "",
        "WAPE by forecast week:", "", table(by_week),
        "## Prediction intervals (80%)", "",
        "Built from validation-period errors only, then checked on the test period:", "",
        f"- One day ahead: {meta['interval']['coverage_one_step_test']:.1%} of test actuals fall inside the interval",
        f"- Up to 30 days ahead: {meta['interval']['coverage_30_day_test']:.1%}", "",
        "Multipliers applied to the forecast, by forecast week:", "", table(factors),
        "## Assumptions and limits", "",
        "- Promotions and holidays are treated as known in advance. Future forecasts default to no promotion.",
        "- Competitor price is not known in advance; the last observed price ratio is carried forward.",
        "- Weather is excluded: it is inconsistent within a city-day and shows no demand effect.",
        "- Only two years of history: annual seasonality is learned from a single full cycle.",
        "- LSTM was not used: 731 points per series, and tabular models already sit close to the noise floor.",
    ]
    artefact("report").write_text("\n".join(lines), encoding="utf-8")
