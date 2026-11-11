"""Page 4: Dynamic Safety Stock."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (MUTED, SERIES, before_after, data_label, get_data, page_header, policy_controls,
                              sidebar_filters, style)
from src.inventory_calculations import safety_stock as ssm
from src.services import inventory_service as inv


@st.cache_data(show_spinner="Calculating safety stock...")
def _policy(level: float, method: str, window: int | None, stamp: str):
    return inv.policy_table(get_data(), level, method, window)


def render() -> None:
    page_header("Dynamic Safety Stock", "Safety stock and reorder point from demand and lead-time variability.")
    df = sidebar_filters(get_data(), period=False)
    level, method, window = policy_controls("ss")
    stamp = str(get_data()["date"].max())
    keys = df[["store_id", "product_id"]].drop_duplicates()
    policy = _policy(level, method, window, stamp).merge(keys, on=["store_id", "product_id"])

    c = st.columns(2)
    stores = policy[["store_id", "store_city"]].drop_duplicates().sort_values("store_id")
    store_id = c[0].selectbox("Store", stores["store_id"], key="ss_store",
                              format_func=lambda s: stores.set_index("store_id").loc[s, "store_city"])
    in_store = policy[policy["store_id"] == store_id].sort_values("product_id")
    product_id = c[1].selectbox("Product", in_store["product_id"], key="ss_product",
                                format_func=lambda p: in_store.set_index("product_id").loc[p, "product_name"])
    row = in_store.set_index("product_id").loc[product_id]

    k = st.columns(4)
    k[0].metric("Demand variability (std, units/day)", f"{row.demand_std:.1f}",
                help="Dynamic: spread of forecast errors. Classic: spread of daily demand.")
    k[1].metric("Lead-time variability (std, days)", f"{row.lt_std:.2f}", help=f"Average lead time {row.lt_mean:.1f} days, "
                f"from {int(row.lt_orders)} past orders.")
    k[2].metric("Safety stock (units)", f"{row.safety_stock:.0f}",
                delta=f"{row.safety_stock - row.baseline_safety_stock:+.0f} vs existing", delta_color="off")
    k[3].metric("Reorder point (units)", f"{row.reorder_point:.0f}",
                delta=f"{row.rop_change:+.0f} vs existing {row.reorder_point_baseline_units:.0f}", delta_color="off")

    with st.expander("How this number is calculated", expanded=False):
        dem_part = row.demand_std ** 2 * row.lt_mean
        lt_part = row.demand_mean ** 2 * row.lt_std ** 2
        st.latex(r"SS = Z \times \sqrt{\sigma_D^2 \, L + D^2 \, \sigma_L^2} \qquad ROP = D \times L + SS")
        st.markdown(
            f"- Z = **{row.z:.3f}** (service level {level:.1%})\n"
            f"- D = **{row.demand_mean:.2f}** units/day, σD = **{row.demand_std:.2f}** units/day\n"
            f"- L = **{row.lt_mean:.2f}** days, σL = **{row.lt_std:.2f}** days\n"
            f"- SS = {row.z:.3f} × √({row.demand_std:.2f}² × {row.lt_mean:.2f} + {row.demand_mean:.2f}² × {row.lt_std:.2f}²)"
            f" = {row.z:.3f} × √({dem_part:.1f} + {lt_part:.1f}) = **{row.safety_stock:.1f}** units\n"
            f"- ROP = {row.demand_mean:.2f} × {row.lt_mean:.2f} + {row.safety_stock:.1f} = **{row.reorder_point:.1f}** units\n"
            f"- Lead-time uncertainty accounts for **{lt_part / (dem_part + lt_part):.0%}** of the variance; "
            f"demand uncertainty for {dem_part / (dem_part + lt_part):.0%}.")
        if row.demand_fallback or row.lt_fallback:
            st.warning("Fallback used: too little history for this item, so category-level variability or supplier "
                       "lead-time parameters were applied.")
        st.caption("Demand is per day and lead time is in days. The formula assumes independent daily demand and "
                   "roughly normal lead-time demand; the Monte Carlo page tests the result without those assumptions.")

    s = inv.summary(policy)
    lt_share = float((policy.demand_mean ** 2 * policy.lt_std ** 2 /
                      (policy.demand_std ** 2 * policy.lt_mean + policy.demand_mean ** 2 * policy.lt_std ** 2)).mean())
    before_after([
        {"metric": "Rule", "before": "One fixed reorder point per item", "after": f"{inv.METHODS[method]}, {level:.1%} target"},
        {"metric": "Service level the reorder point is sized for",
         "before": f"{s['baseline_service_level_mean']:.0%} on average "
                   f"({s['baseline_service_level_min']:.0%} to {s['baseline_service_level_max']:.0%})",
         "after": f"{level:.1%} for every item"},
        {"metric": "Total safety stock (units)", "before": f"{s['baseline_safety_stock_units']:,.0f}",
         "after": f"{s['safety_stock_units']:,.0f}",
         "change": f"{s['safety_stock_units'] / s['baseline_safety_stock_units'] - 1:+.0%}"},
        {"metric": "Safety stock value (AED)", "before": f"{s['baseline_safety_stock_value_aed']:,.0f}",
         "after": f"{s['safety_stock_value_aed']:,.0f}",
         "change": f"{s['safety_stock_value_aed'] / s['baseline_safety_stock_value_aed'] - 1:+.0%}"},
        {"metric": "Reorder points raised / lowered", "before": "-", "after": f"{s['raised']} / {s['lowered']}"},
    ], note=f"For the {s['series']} store-product combinations in the current filter. The existing reorder points protect "
            f"unevenly; the formula sizes every item to the same target. Lead-time uncertainty drives about {lt_share:.0%} "
            "of the requirement, so a better forecast alone does not shrink safety stock much. Whether the extra stock "
            "pays off is tested by simulation (Monte Carlo and Optimization pages), not assumed here.")

    left, right = st.columns(2)
    with left:
        st.subheader("Service-level sensitivity")
        curve = ssm.service_level_curve(row.demand_mean, row.demand_std, row.lt_mean, row.lt_std)
        marks = ssm.service_level_curve(row.demand_mean, row.demand_std, row.lt_mean, row.lt_std,
                                        levels=[0.90, 0.95, 0.975, 0.99])
        fig = go.Figure()
        fig.add_scatter(x=curve.service_level * 100, y=curve.safety_stock, name="Safety stock",
                        line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=marks.service_level * 100, y=marks.safety_stock, mode="markers+text", showlegend=False,
                        marker=dict(color=SERIES[0], size=9, line=dict(color="white", width=2)),
                        text=[f"{v:.0f}" for v in marks.safety_stock], textposition="top left")
        fig.add_hline(y=row.baseline_safety_stock, line_color=MUTED, line_dash="dot", line_width=1.5,
                      annotation_text="Existing", annotation_position="bottom right")
        st.plotly_chart(style(fig, y_title="Safety stock (units)", x_title="Service level (%)"), width="stretch")
    with right:
        st.subheader("Reorder point over time")
        win = st.slider("Rolling window (days)", 28, 180, 60, step=4, key="ss_roll_window")
        series = get_data().query("store_id == @store_id and product_id == @product_id")
        roll = ssm.rolling_reorder_point(series, level, win, row.lt_mean, row.lt_std)
        fig = go.Figure()
        fig.add_scatter(x=roll.date, y=roll.reorder_point, name=f"Recalculated daily ({win}-day window)",
                        line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=roll.date, y=np.full(len(roll), row.reorder_point_baseline_units), name="Existing fixed reorder point",
                        line=dict(color=MUTED, width=1.5, dash="dot"))
        st.plotly_chart(style(fig, height=290, y_title="Reorder point (units)"), width="stretch")

    st.subheader(f"Safety stock by product, {row.store_city}")
    fig = go.Figure()
    fig.add_bar(x=in_store.product_name, y=in_store.baseline_safety_stock, name="Existing (before AI)", marker_color=SERIES[1])
    fig.add_bar(x=in_store.product_name, y=in_store.safety_stock, name=f"Formula at {level:.1%} (after AI)", marker_color=SERIES[0])
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(style(fig, height=380, y_title="Safety stock (units)"), width="stretch")
    data_label("model" if method == "dynamic" else "derived")
