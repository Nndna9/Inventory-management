"""Page 2: Demand Forecasting."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import MUTED, SERIES, before_after, data_label, get_data, page_header, sidebar_filters, style
from src.data_processing import eda
from src.data_processing.ingest import append_new_data
from src.data_processing.loader import DataValidationError, load_all_raw
from src.services import forecast_service as fs

TYPE_COLOUR = {"baseline": SERIES[1], "statistical": SERIES[2], "machine learning": SERIES[0]}


@st.cache_resource(show_spinner=False)
def _model(stamp: float):
    return fs.load_model()


@st.cache_data(show_spinner=False)
def _artefacts(stamp: float):
    return fs.load_meta(), fs.load_predictions(), fs.load_factors()


@st.cache_data(show_spinner="Forecasting...")
def _forecast(stamp: float, store_id: str, product_id: str, horizon: int, promotion: bool, discount: float):
    meta, _, factors = _artefacts(stamp)
    return fs.forecast_series(get_data(), _model(stamp), meta, factors, store_id, product_id, horizon, promotion, discount)


def _stamp() -> float:
    from src.forecasting.train import artefact
    return artefact("meta").stat().st_mtime


def render() -> None:
    page_header("Demand Forecasting", "Daily demand forecast by store and product, with model comparison.")
    df = sidebar_filters(get_data(), period=False)
    if not fs.artefacts_ready():
        st.warning("Forecasting models have not been built yet. Run `python scripts/run_pipeline.py`, then reload.")
        return
    stamp = _stamp()
    meta, preds, _ = _artefacts(stamp)

    c = st.columns([1, 1.4, 1.2, 1.2])
    stores = df[["store_id", "store_city"]].drop_duplicates().sort_values("store_id")
    store_id = c[0].selectbox("Store", stores["store_id"], key="fc_store",
                              format_func=lambda s: stores.set_index("store_id").loc[s, "store_city"])
    products = df[["product_id", "product_name"]].drop_duplicates().sort_values("product_id")
    product_id = c[1].selectbox("Product", products["product_id"], key="fc_product",
                                format_func=lambda p: products.set_index("product_id").loc[p, "product_name"])
    horizon = c[2].slider("Forecast horizon (days)", 7, 90, 30, step=1, key="fc_horizon")
    history_days = c[3].slider("History shown (days)", 30, 365, 120, step=30, key="fc_history")
    c = st.columns([1, 1.4, 2.4])
    promotion = c[0].toggle("Assume a promotion", value=False, key="fc_promo",
                            help="Scenario: a promotion runs on every day of the forecast horizon.")
    discount = c[1].select_slider("Discount (%)", options=[0, 5, 10, 15, 20], value=0, key="fc_discount",
                                  disabled=not promotion)

    full = get_data()
    series = full[(full.store_id == store_id) & (full.product_id == product_id)].sort_values("date")
    fc = _forecast(stamp, store_id, product_id, horizon, promotion, float(discount))
    acc = fs.series_accuracy(preds, meta, store_id, product_id)
    last_actual = float(series.tail(horizon)["demand_units"].sum())

    k = st.columns(4)
    k[0].metric(f"Forecast demand, next {horizon} days", f"{fc['forecast'].sum():,.0f}",
                delta=f"{fc['forecast'].sum() / last_actual - 1:+.1%} vs last {horizon} days", delta_color="off")
    k[1].metric("Average per day", f"{fc['forecast'].mean():.1f}")
    k[2].metric("80% range, total", f"{fc['lower'].sum():,.0f} - {fc['upper'].sum():,.0f}",
                help="Sum of daily lower and upper bounds. Daily bounds come from past forecast errors.")
    k[3].metric("Forecast error on test (WAPE)", f"{acc['after']['wape']:.1%}",
                delta=f"{(acc['after']['wape'] - acc['before']['wape']) * 100:+.1f} pts vs 28-day average",
                delta_color="inverse")

    st.subheader("History and forecast")
    hist = series.tail(history_days)
    fig = go.Figure()
    fig.add_scatter(x=fc["date"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["date"], y=fc["lower"], line=dict(width=0), fill="tonexty", fillcolor="rgba(235,104,52,0.18)",
                    name="80% interval", hoverinfo="skip")
    fig.add_scatter(x=hist["date"], y=hist["demand_units"], name="Actual demand", line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=fc["date"], y=fc["forecast"], name="Forecast", line=dict(color=SERIES[1], width=2))
    st.plotly_chart(style(fig, height=380, y_title="Units per day"), width="stretch")
    data_label("model")

    ba = fs.before_after(meta)
    before_after([
        {"metric": "Method", "before": ba["before_model"], "after": ba["after_model"]},
        {"metric": "Forecast error, all series (WAPE)", "before": f"{ba['before']['wape']:.1%}",
         "after": f"{ba['after']['wape']:.1%}", "change": f"{ba['wape_change']:+.1%}"},
        {"metric": "Average miss per day, all series (units)", "before": f"{ba['before']['mae']:.2f}",
         "after": f"{ba['after']['mae']:.2f}", "change": f"{ba['mae_change']:+.1%}"},
        {"metric": "This store-product (WAPE)", "before": f"{acc['before']['wape']:.1%}",
         "after": f"{acc['after']['wape']:.1%}",
         "change": f"{acc['after']['wape'] / acc['before']['wape'] - 1:+.1%}"},
        {"metric": "Reacts to promotions, holidays, weekends", "before": "No", "after": "Yes"},
    ], note=f"Measured on the test period ({meta['split']['test_start']} to {meta['split']['test_end']}), which no model "
            "saw during training. The gain is moderate because daily demand is noisy.")

    t_actual, t_models, t_season, t_data = st.tabs(["Actual vs predicted", "Model comparison", "Seasonality", "Add new data"])

    with t_actual:
        p = fs.series_predictions(preds, store_id, product_id)
        days = st.slider("Test days shown", 14, len(p), min(60, len(p)), step=1, key="fc_test_days")
        p = p.tail(days)
        fig = go.Figure()
        fig.add_scatter(x=p["date"], y=p["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=p["date"], y=p["lower"], line=dict(width=0), fill="tonexty", fillcolor="rgba(235,104,52,0.18)",
                        name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=p["date"], y=p["actual"], name="Actual", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=p["date"], y=p[meta["best_ml_model"]], name=f"Predicted ({meta['best_ml_model']})",
                        line=dict(color=SERIES[1], width=2))
        fig.add_scatter(x=p["date"], y=p[meta["before_ai_model"]], name="28-day average (before AI)",
                        line=dict(color=MUTED, width=1.5, dash="dot"))
        st.plotly_chart(style(fig, y_title="Units per day"), width="stretch")
        st.caption(f"One-day-ahead predictions. Interval coverage on the full test set: "
                   f"{meta['interval']['coverage_one_step_test']:.1%} (target 80%).")

    with t_models:
        m = fs.metrics_table(meta, "test")
        fig = go.Figure()
        for kind, colour in TYPE_COLOUR.items():
            part = m[m["type"] == kind]
            fig.add_bar(y=part["model"], x=part["wape"] * 100, orientation="h", name=kind.capitalize(), marker_color=colour,
                        text=[f"{v:.1%}" for v in part["wape"]], textposition="outside", cliponaxis=False)
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(categoryorder="array", categoryarray=m["model"][::-1].tolist())
        fig.update_xaxes(range=[0, m["wape"].max() * 115])
        st.plotly_chart(style(fig, height=360, x_title="Test WAPE (%), lower is better"), width="stretch")
        st.dataframe(
            m.rename(columns={"model": "Model", "type": "Type", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE",
                              "wape": "WAPE", "bias": "Bias"}),
            hide_index=True, width="stretch",
            column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                           "RMSE": st.column_config.NumberColumn(format="%.2f"),
                           "MAPE": st.column_config.NumberColumn(format="percent"),
                           "WAPE": st.column_config.NumberColumn(format="percent"),
                           "Bias": st.column_config.NumberColumn(format="%.2f")})
        md = meta["multi_day_test"]
        st.caption(f"Model chosen on validation data: {meta['selected_model']}. All models scored on the same test days. "
                   f"Forecasting {md['horizon']} days ahead: WAPE {md['after']['wape']:.1%} vs "
                   f"{md['before']['wape']:.1%} for a flat 28-day average.")

    with t_season:
        wd, mo = eda.seasonality_profile(series)
        l, r = st.columns(2)
        with l:
            st.markdown("**By weekday**")
            fig = go.Figure(go.Bar(x=wd["weekday"], y=wd["relative_demand"] * 100, marker_color=SERIES[0]))
            fig.update_layout(hovermode="closest")
            fig.add_hline(y=100, line_color=MUTED, line_width=1)
            st.plotly_chart(style(fig, height=300, y_title="% of average day"), width="stretch")
        with r:
            st.markdown("**By month**")
            fig = go.Figure(go.Bar(x=mo["month_name"], y=mo["relative_demand"] * 100, marker_color=SERIES[0]))
            fig.update_layout(hovermode="closest")
            fig.add_hline(y=100, line_color=MUTED, line_width=1)
            st.plotly_chart(style(fig, height=300, y_title="% of average day"), width="stretch")
        data_label("original")

    with t_data:
        st.markdown("Upload new daily rows (same 32 columns as the original file) to extend the history and update forecasts.")
        st.caption(f"Data currently runs to {full['date'].max():%d %b %Y}. New rows must start the next day, cover every "
                   "store-product, and pass the stock-balance checks. They are saved separately in data/raw/updates/.")
        up = st.file_uploader("New data (CSV)", type="csv", key="fc_upload")
        if up is not None and st.button("Validate, add and retrain", type="primary", key="fc_retrain"):
            try:
                target = append_new_data(pd.read_csv(up), load_all_raw())
            except (DataValidationError, pd.errors.ParserError, ValueError) as e:
                st.error(str(e))
            else:
                with st.spinner("Rebuilding data and retraining models (about 2 minutes)..."):
                    from src.pipeline import rebuild
                    rebuild(verbose=False)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Added {target.name}. Forecasts now use the extended history. Reload the page.")
