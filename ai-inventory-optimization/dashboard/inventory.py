"""Page 3: Inventory & Reorder Management."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (MUTED, SERIES, STATUS, before_after, data_label, get_data, page_header, policy_controls,
                              sidebar_filters, style)
from src.services import inventory_service as inv

STATUS_COLOUR = {"Out of stock": STATUS["critical"], "Reorder now": STATUS["serious"],
                 "Below reorder point, order placed": STATUS["warning"], "Healthy": STATUS["good"]}


@st.cache_data(show_spinner="Calculating reorder points...")
def _policy(level: float, method: str, window: int | None, stamp: str):
    return inv.policy_table(get_data(), level, method, window)


def render() -> None:
    data = get_data()
    as_of = data["date"].max()
    page_header("Inventory & Reorder Management", f"Stock position at close of {as_of:%d %b %Y} against reorder points.")
    df = sidebar_filters(data, period=False)
    level, method, window = policy_controls("inv")
    keys = df[["store_id", "product_id"]].drop_duplicates()
    policy = _policy(level, method, window, str(as_of)).merge(keys, on=["store_id", "product_id"])
    s = inv.summary(policy)

    k = st.columns(5)
    k[0].metric("Current stock (units)", f"{policy.closing_inventory_units.sum():,.0f}")
    k[1].metric("On order (units)", f"{policy.on_order_units.sum():,.0f}")
    k[2].metric("Reorder now", s["reorder_now"], help="Stock on hand plus on order is at or below the reorder point.")
    k[3].metric("Out of stock", s["out_of_stock"])
    k[4].metric("Recommended orders (units)", f"{s['order_units']:,}", help=f"Value AED {s['order_value_aed']:,.0f} at unit cost.")

    before_after([
        {"metric": "Items needing a new order today", "before": str(s["baseline_reorder_now"]), "after": str(s["reorder_now"]),
         "change": f"{s['reorder_now'] - s['baseline_reorder_now']:+d}"},
        {"metric": "Total reorder point (units)", "before": f"{s['baseline_rop_units']:,.0f}", "after": f"{s['rop_units']:,.0f}",
         "change": f"{s['rop_units'] / s['baseline_rop_units'] - 1:+.0%}"},
        {"metric": "Order quantity to place today (units)", "before": "0", "after": f"{s['order_units']:,}"},
    ], note="'Before' is the fixed reorder point in the dataset: every item at or below it already has an order placed, so "
            "it asks for nothing more today. 'After' applies the reorder points from the Dynamic Safety Stock page; a higher "
            "target service level triggers orders earlier.")

    st.subheader("Low-stock alerts and product status")
    c = st.columns([2.2, 1])
    shown = c[0].multiselect("Status", inv.STATUS_ORDER, default=inv.STATUS_ORDER[:3], key="inv_status")
    max_cover = c[1].slider("Max days of cover", 0.0, 30.0, 30.0, step=0.5, key="inv_cover",
                            help="Show only items with at most this many days of stock on hand.")
    table = policy[policy.status.isin(shown) & (policy.days_of_cover <= max_cover)].copy()
    table["status"] = table["status"].astype("category").cat.set_categories(inv.STATUS_ORDER, ordered=True)
    table = table.sort_values(["status", "days_of_cover"])
    if table.empty:
        st.success("No items match these filters.")
    else:
        st.dataframe(
            table[["status", "store_city", "product_name", "closing_inventory_units", "on_order_units",
                   "inventory_position_units", "safety_stock", "reorder_point", "reorder_point_baseline_units",
                   "days_of_cover", "recommended_order_qty"]]
            .rename(columns={"status": "Status", "store_city": "Store", "product_name": "Product",
                             "closing_inventory_units": "Current stock", "on_order_units": "On order",
                             "inventory_position_units": "Inventory position", "safety_stock": "Safety stock",
                             "reorder_point": "Reorder point", "reorder_point_baseline_units": "Existing reorder point",
                             "days_of_cover": "Days of cover", "recommended_order_qty": "Recommended order"}),
            hide_index=True, width="stretch", height=380,
            column_config={"Safety stock": st.column_config.NumberColumn(format="%.0f"),
                           "Reorder point": st.column_config.NumberColumn(format="%.0f"),
                           "Days of cover": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"{len(table)} of {len(policy)} store-product combinations. Inventory position = current stock + on order. "
                   "Recommended order = order-up-to level minus inventory position, when the position is at or below the reorder point.")

    left, right = st.columns([3, 2])
    with left:
        stores = policy[["store_id", "store_city"]].drop_duplicates().sort_values("store_id")
        store_id = st.selectbox("Store", stores["store_id"], key="inv_store",
                                format_func=lambda x: stores.set_index("store_id").loc[x, "store_city"])
        one = policy[policy.store_id == store_id].sort_values("product_id")
        st.subheader("Stock against reorder point")
        fig = go.Figure()
        fig.add_bar(x=one.product_name, y=one.closing_inventory_units, name="Current stock", marker_color=SERIES[0])
        fig.add_bar(x=one.product_name, y=one.on_order_units, name="On order", marker_color=SERIES[2])
        fig.add_scatter(x=one.product_name, y=one.reorder_point, mode="markers", name="Reorder point",
                        marker=dict(symbol="line-ew", size=22, line=dict(color="#0b0b0b", width=2.5)))
        fig.update_layout(barmode="stack", bargap=0.35)
        st.plotly_chart(style(fig, height=400, y_title="Units"), width="stretch")
    with right:
        st.subheader("Status mix")
        counts = policy["status"].value_counts().reindex(inv.STATUS_ORDER, fill_value=0)
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", text=counts.values, textposition="outside",
                               cliponaxis=False, marker_color=[STATUS_COLOUR[x] for x in counts.index]))
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, max(counts.max() * 1.15, 1)])
        st.plotly_chart(style(fig, height=300, x_title="Store-product combinations"), width="stretch")

    st.subheader("Stock history")
    c = st.columns([2, 2])
    product_id = c[0].selectbox("Product", one["product_id"], key="inv_product",
                                format_func=lambda x: one.set_index("product_id").loc[x, "product_name"])
    days = c[1].slider("Days shown", 30, 365, 90, step=15, key="inv_days")
    row = one.set_index("product_id").loc[product_id]
    hist = data[(data.store_id == store_id) & (data.product_id == product_id)].sort_values("date").tail(days)
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.closing_inventory_units, name="Closing stock", line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=hist.date, y=np.full(len(hist), row.reorder_point), name="Reorder point (after AI)",
                    line=dict(color=SERIES[1], width=2))
    fig.add_scatter(x=hist.date, y=np.full(len(hist), row.reorder_point_baseline_units), name="Existing reorder point",
                    line=dict(color=MUTED, width=1.5, dash="dot"))
    st.plotly_chart(style(fig, height=320, y_title="Units"), width="stretch")
    st.caption("The stock history is what happened under the existing rule. The new reorder point is drawn for comparison; "
               "how stock would have behaved under it is simulated on the Monte Carlo page.")
    data_label("model" if method == "dynamic" else "derived")
