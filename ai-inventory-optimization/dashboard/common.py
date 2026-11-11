"""Shared dashboard helpers: cached data access, chart styling, reusable components."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_processing.clean import clean
from src.data_processing.loader import load_raw
from src.config import path

# Validated categorical order (fixed, never cycled) and reserved status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


@st.cache_data(show_spinner="Loading data...")
def get_data() -> pd.DataFrame:
    """Cleaned dataset. Uses the processed parquet when present, else builds it from the raw CSV."""
    p = path("processed")
    if p.exists():
        return pd.read_parquet(p)
    return clean(load_raw())[0]


@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    return load_raw()


def style(fig: go.Figure, height: int = 340, y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    """One look for every chart: thin marks, recessive grid, text in ink colours."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=13), colorway=SERIES,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=MUTED)),
        hovermode="x unified", bargap=0.25)
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=x_title, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title, tickfont=dict(color=MUTED))
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def data_label(kind: str) -> None:
    """Every page states where its numbers come from."""
    labels = {
        "original": "Source: original synthetic dataset (2024-2025).",
        "derived": "Source: calculated from the original synthetic dataset.",
        "model": "Source: model predictions trained on the original synthetic dataset.",
        "simulated": "Source: simulation. Results are synthetic, not measured.",
    }
    st.caption(labels[kind])


def before_after(rows: list[dict], note: str | None = None) -> None:
    """'Before AI vs After AI' insight panel.

    rows: [{"metric": str, "before": str, "after": str | None, "change": str | None}]
    `after=None` shows the phase that will supply the value instead of a made-up number.
    """
    with st.container(border=True):
        st.markdown("**Before AI vs After AI**")
        table = pd.DataFrame([{
            "Metric": r["metric"],
            "Before AI (existing rule)": r["before"],
            "After AI": r.get("after") or r.get("pending", "Pending"),
            "Change": r.get("change") or "-",
        } for r in rows])
        st.dataframe(table, hide_index=True, width="stretch")
        if note:
            st.caption(note)


def pending(page: str, phase: str, what: str) -> None:
    page_header(page, what)
    st.info(f"This page is switched on in {phase}. Nothing is shown here until the backend module passes its tests.")


def sidebar_filters(df: pd.DataFrame, period: bool = True) -> pd.DataFrame:
    """Global filters shown in the sidebar on every page. Returns the filtered dataset.

    period=False hides the date slider (used by pages about the current stock position).

    Selections are kept in session state, so they persist when switching pages.
    The end of the date range acts as "today" for stock-position figures.
    """
    with st.sidebar:
        st.markdown("#### Filters")
        cities = sorted(df["store_city"].unique())
        cats = sorted(df["category"].unique())
        stores = st.multiselect("Store", cities, default=cities, key="f_store")
        categories = st.multiselect("Category", cats, default=cats, key="f_category")
        products = sorted(df.loc[df["category"].isin(categories or cats), "product_name"].unique())
        chosen = st.multiselect("Product", products, default=[], key="f_product", placeholder="All products")
        lo, hi = df["date"].min().date(), df["date"].max().date()
        if period:
            start, end = st.slider("Period", min_value=lo, max_value=hi, value=(lo, hi), format="MMM YYYY", key="f_period")
        else:  # pages that describe today's position always use the full history
            start, end = lo, hi
    out = df[df["store_city"].isin(stores) & df["category"].isin(categories)
             & df["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if chosen:
        out = out[out["product_name"].isin(chosen)]
    if out.empty:
        st.warning("No data matches the current filters. Select at least one store and one category.")
        st.stop()
    return out


SERVICE_LEVELS = [0.90, 0.95, 0.975, 0.99]


def policy_controls(prefix: str):
    """Service level + calculation method controls shared by the inventory pages. Returns (level, method, window)."""
    from src.services import inventory_service as inv

    c = st.columns([1.3, 1.5, 1.4])
    level = c[0].select_slider("Target service level", options=SERVICE_LEVELS, value=0.95, key="policy_service_level",
                               format_func=lambda v: f"{v:.1%}".replace(".0%", "%"))
    methods = list(inv.METHODS) if inv.dynamic_available() else ["classic"]
    method = c[1].radio("Demand input", methods, format_func=inv.METHODS.get, horizontal=True, key="policy_method",
                        help="Dynamic: forecast demand over the coming lead time and forecast-error variability. "
                             "Classic: historical average and standard deviation.")
    window = None
    if method == "classic":
        window = c[2].slider("History used (days)", 30, 730, 730, step=30, key="policy_window",
                             help="Classic method only: how much recent history feeds the average and standard deviation.")
    return level, method, window
