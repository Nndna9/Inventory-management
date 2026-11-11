"""Page 1: Executive Overview."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (SERIES, STATUS, before_after, data_label, get_data, get_raw, page_header,
                              sidebar_filters, style)
from src.data_processing import eda
from src.data_processing.quality import run_quality_checks
from src.data_processing.schema import data_dictionary
from src.services import forecast_service as fs
from src.services.data_service import baseline_kpis, current_state

STATUS_COLOUR = {"Healthy": STATUS["good"], "Below reorder point, order placed": STATUS["warning"],
                 "Reorder needed": STATUS["serious"], "Low cover": STATUS["serious"], "Out of stock": STATUS["critical"]}
ALERT_STATUSES = ["Out of stock", "Low cover", "Reorder needed"]


def render() -> None:
    df = sidebar_filters(get_data())
    k = baseline_kpis(df)
    state = current_state(df)
    page_header("Executive Overview",
                f"Position at close of {k['as_of']:%d %b %Y} | {k['stores']} stores, {k['products']} products, "
                f"{k['series']} store-product combinations")

    tab_overview, tab_quality, tab_dict = st.tabs(["Overview", "Data quality and EDA", "Data dictionary"])

    with tab_overview:
        c = st.columns(4)
        c[0].metric("Products", k["products"])
        c[1].metric("Inventory on hand (units)", f"{k['inventory_units']:,}")
        c[2].metric("Total demand in period (units)", f"{k['total_demand']:,}")
        if fs.artefacts_ready():
            meta = fs.load_meta()
            predicted = fs.predicted_demand(fs.load_future(), state, days=30)
            c[3].metric("Predicted demand, next 30 days", f"{predicted:,.0f}",
                        delta=f"{predicted / k['recent_demand'] - 1:+.1%} vs last 30 days", delta_color="off",
                        help=f"Forecast from {meta['trained_through']} for the selected stores and products, "
                             "no-promotion scenario.")
        else:
            meta = None
            c[3].metric("Predicted demand, next 30 days", "Not built", help="Run python scripts/run_pipeline.py")
        c = st.columns(4)
        c[0].metric("On hand at or below reorder point", k["below_rop"], help="Store-product combinations at or below the existing reorder point. An order is already placed for all but "
                         f"{k['reorder_needed']}.")
        c[1].metric("Stockout events in period", f"{k['stockout_events']:,}", help="Store-product-days with lost sales.")
        c[2].metric("Fill rate in period", f"{k['fill_rate']:.1%}")
        c[3].metric("Holding cost per year (AED)", f"{k['annual_holding_cost_aed']:,.0f}",
                    help=f"ASSUMED rate: {k['holding_rate_annual']:.0%} of average inventory value per year.")
        data_label("derived")

        before_after([
            {"metric": "Fill rate", "before": f"{k['fill_rate']:.2%}", "pending": "Phase 5"},
            {"metric": "Lost sales (units, period)", "before": f"{k['total_lost']:,}", "pending": "Phase 5"},
            {"metric": "Lost margin (AED, period)", "before": f"{k['lost_margin_aed']:,.0f}", "pending": "Phase 5"},
            {"metric": "Average inventory value (AED)", "before": f"{k['avg_inventory_value_aed']:,.0f}", "pending": "Phase 5"},
            ({"metric": "Forecast error (WAPE, test period)", "before": f"{meta['test_metrics_before']['wape']:.1%}",
              "after": f"{meta['test_metrics_after']['wape']:.1%}",
              "change": f"{meta['test_metrics_after']['wape'] / meta['test_metrics_before']['wape'] - 1:+.1%}"}
             if meta else {"metric": "Forecast error (WAPE)", "before": "Not built", "pending": "Not built"}),
        ], note="'Before AI' is the fixed reorder point and order-up-to level already in the dataset. "
                "'After AI' is filled in only from tested model and simulation output.")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Demand served and lost by month")
            m = eda.monthly_demand(df)
            fig = go.Figure()
            fig.add_bar(x=m["month"], y=m["sold"], name="Units sold", marker_color=SERIES[0])
            fig.add_bar(x=m["month"], y=m["lost"], name="Lost sales", marker_color=SERIES[1])
            fig.update_layout(barmode="stack", legend_traceorder="normal")
            st.plotly_chart(style(fig, y_title="Units"), width="stretch")
        with right:
            st.subheader(f"Inventory health at {k['as_of']:%d %b %Y}")
            counts = state["status"].value_counts().reindex(list(STATUS_COLOUR), fill_value=0)
            fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", text=counts.values,
                                   textposition="outside", cliponaxis=False,
                                   marker_color=[STATUS_COLOUR[s] for s in counts.index]))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(range=[0, max(counts.max() * 1.15, 1)])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, x_title="Store-product combinations"), width="stretch")

        st.subheader("Alerts")
        top_n = st.slider("Alerts to show", min_value=5, max_value=50, value=10, step=5, key="overview_top_n")
        alerts_all = state[state["status"].isin(ALERT_STATUSES)].sort_values("days_of_cover")
        alerts = alerts_all.head(top_n)
        if alerts.empty:
            st.success("Nothing is out of stock, short of cover, or waiting for an order.")
        else:
            st.dataframe(
                alerts[["status", "store_city", "product_name", "closing_inventory_units", "reorder_point_baseline_units",
                        "on_order_units", "days_of_cover", "supplier_id", "supplier_avg_lead_time_days"]]
                .rename(columns={"status": "Status", "store_city": "Store", "product_name": "Product",
                                 "closing_inventory_units": "On hand", "reorder_point_baseline_units": "Reorder point",
                                 "on_order_units": "On order", "days_of_cover": "Days of cover",
                                 "supplier_id": "Supplier", "supplier_avg_lead_time_days": "Avg lead time (days)"}),
                hide_index=True, width="stretch",
                column_config={"Days of cover": st.column_config.NumberColumn(format="%.1f")})
            st.caption(f"Showing {len(alerts)} of {len(alerts_all)} alerts, most urgent first. "
                       "Days of cover = on-hand stock / average daily demand over the last 28 days. "
                       "Low cover = fewer days of cover than the supplier's average lead time.")

    with tab_quality:
        checks = run_quality_checks(get_raw())
        errors = int((~checks.passed & (checks.severity == "error")).sum())
        warns = int((~checks.passed & (checks.severity == "warning")).sum())
        c = st.columns(3)
        c[0].metric("Checks passed", f"{int(checks.passed.sum())} of {len(checks)}")
        c[1].metric("Errors", errors)
        c[2].metric("Warnings (handled)", warns)
        st.dataframe(checks.assign(passed=checks.passed.map({True: "Pass", False: "Flag"})), hide_index=True, width="stretch")

        l, r = st.columns(2)
        with l:
            st.subheader("What moves demand")
            eff = eda.driver_effects(df)
            eff = eff[eff.driver.isin(["promotion_flag", "holiday_flag", "weekend_flag"]) & (eff.value == "1")]
            names = {"promotion_flag": "Promotion", "holiday_flag": "Holiday", "weekend_flag": "Weekend (Fri-Sat)"}
            fig = go.Figure(go.Bar(x=[names[d] for d in eff.driver], y=(eff.relative_demand - 1) * 100,
                                   marker_color=SERIES[0], text=[f"{v:+.0f}%" for v in (eff.relative_demand - 1) * 100],
                                   textposition="outside"))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(style(fig, y_title="Demand vs a typical day (%)"), width="stretch")
        with r:
            st.subheader("Supplier lead times (days)")
            lt = eda.lead_time_summary(df)
            fig = go.Figure(go.Bar(x=lt.supplier_id, y=lt.observed_mean, marker_color=SERIES[0], name="Mean lead time",
                                   error_y=dict(type="data", array=lt.observed_std, color="#52514e", thickness=1.5)))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(style(fig, y_title="Days (bar = mean, whisker = 1 std)"), width="stretch")
        st.subheader("Lowest fill rates by product")
        so = eda.stockout_summary(df, "product_id").head(10)
        st.dataframe(so[["product_name", "demand", "lost", "stockout_days", "fill_rate"]]
                     .rename(columns={"product_name": "Product", "demand": "Demand", "lost": "Lost sales",
                                      "stockout_days": "Stockout days", "fill_rate": "Fill rate"}),
                     hide_index=True, width="stretch",
                     column_config={"Fill rate": st.column_config.NumberColumn(format="percent")})
        data_label("original")

    with tab_dict:
        st.dataframe(data_dictionary(), hide_index=True, width="stretch", height=560)
        st.caption("Role 'excluded' = same-day outcome; never used as a forecasting input.")
