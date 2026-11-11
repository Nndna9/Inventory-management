"""End-to-end rebuild of every artefact. Used by scripts/run_pipeline.py and by the dashboard's retrain button."""
from __future__ import annotations

from src.config import path
from src.data_processing.clean import clean
from src.data_processing.loader import load_all_raw
from src.data_processing.report import write_reports
from src.feature_engineering.features import build_features


def phase1(verbose: bool = True):
    raw = load_all_raw()  # original + accepted updates
    cleaned, log = clean(raw)
    path("processed").parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(path("processed"), index=False)
    feats = build_features(cleaned, horizon=1)
    feats.to_parquet(path("features"), index=False)
    write_reports(raw, cleaned, log, path("reports"))
    if verbose:
        print(f"[phase 1] clean rows: {len(cleaned):,} | feature rows: {len(feats):,} | {log}")
    return cleaned


def phase2(cleaned, verbose: bool = True) -> dict:
    from src.forecasting.train import run_training

    if verbose:
        print("[phase 2] training and evaluating forecasting models")
    out = run_training(cleaned, verbose=verbose)
    if verbose:
        print(f"[phase 2] selected: {out['meta']['selected_model']}")
    return out


def rebuild(verbose: bool = True) -> dict:
    cleaned = phase1(verbose)
    return phase2(cleaned, verbose)
