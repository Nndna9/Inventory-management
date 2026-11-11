"""AI-Powered Inventory Management & Supply Chain Optimization System.

Run from the project root:   streamlit run app.py
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import forecasting, inventory, overview, safety_stock  # noqa: E402
from dashboard.common import pending  # noqa: E402

st.set_page_config(page_title="AI Inventory Optimization", page_icon=":material/inventory_2:", layout="wide")


def _pending(title: str, phase: str, what: str):
    fn = partial(pending, title, phase, what)
    fn.__name__ = title  # st.Page needs a name for callables
    return fn


PAGES = [
    st.Page(overview.render, title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(forecasting.render, title="Demand Forecasting", icon=":material/trending_up:", url_path="forecasting"),
    st.Page(inventory.render, title="Inventory & Reorder", icon=":material/inventory:", url_path="inventory"),
    st.Page(safety_stock.render, title="Dynamic Safety Stock", icon=":material/shield:", url_path="safety-stock"),
    st.Page(_pending("Monte Carlo Risk Simulation", "Phase 4", "Stockout risk under uncertain demand and lead time."),
            title="Monte Carlo Simulation", icon=":material/casino:", url_path="simulation"),
    st.Page(_pending("Reorder Point Optimization", "Phase 5", "Bayesian optimization of reorder parameters."),
            title="ROP Optimization", icon=":material/tune:", url_path="optimization"),
    st.Page(_pending("Bullwhip Effect Analysis", "Phase 6", "Demand amplification across the supply chain."),
            title="Bullwhip Effect", icon=":material/ssid_chart:", url_path="bullwhip"),
    st.Page(_pending("AI Recommendations", "Phase 7", "Prioritised actions from the analytics."),
            title="AI Recommendations", icon=":material/lightbulb:", url_path="recommendations"),
]

with st.sidebar:
    st.markdown("### AI Inventory Optimization")
    st.caption("All data is synthetic.")

st.navigation(PAGES, position="sidebar").run()
