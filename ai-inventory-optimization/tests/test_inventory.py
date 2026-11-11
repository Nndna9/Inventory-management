import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.inventory_calculations import safety_stock as ssm
from src.services import forecast_service as fs
from src.services import inventory_service as inv


def test_z_scores():
    assert ssm.z_score(0.90) == pytest.approx(1.2816, abs=1e-4)
    assert ssm.z_score(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert ssm.z_score(0.975) == pytest.approx(1.9600, abs=1e-4)
    assert ssm.z_score(0.99) == pytest.approx(2.3263, abs=1e-4)
    for bad in (0.3, 1.0, 1.2):
        with pytest.raises(ValueError):
            ssm.z_score(bad)


def test_safety_stock_and_rop_match_hand_calculation():
    # Z = 1.645, sigma_D = 5/day, L = 4 days, D = 20/day, sigma_L = 1 day
    # SS = 1.645 * sqrt(25 * 4 + 400 * 1) = 1.645 * sqrt(500) = 1.645 * 22.3607 = 36.783
    ss = ssm.safety_stock(1.645, 5, 4, 20, 1)
    assert ss == pytest.approx(36.783, abs=1e-3)
    assert ssm.reorder_point(20, 4, ss) == pytest.approx(116.783, abs=1e-3)   # 20 * 4 + 36.783


def test_special_cases_of_the_formula():
    assert ssm.safety_stock(2.0, 5, 4, 20, 0) == pytest.approx(2.0 * 5 * 2)     # no lead-time uncertainty: Z*sigma_D*sqrt(L)
    assert ssm.safety_stock(2.0, 0, 4, 20, 1.5) == pytest.approx(2.0 * 20 * 1.5)  # no demand uncertainty: Z*D*sigma_L


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ssm.safety_stock(1.645, -1, 4, 20, 1)
    with pytest.raises(ValueError):
        ssm.safety_stock(1.645, 5, 0, 20, 1)      # lead time must be positive
    with pytest.raises(ValueError):
        ssm.safety_stock(1.645, 5, 4, 20, -0.5)


def test_implied_service_level_inverts_the_formula():
    ss = ssm.safety_stock(norm.ppf(0.95), 5, 4, 20, 1)
    assert ssm.implied_service_level(20 * 4 + ss, 20, 4, 5, 1) == pytest.approx(0.95)


def test_demand_stats_match_pandas(cleaned):
    st = ssm.demand_stats(cleaned).set_index(["store_id", "product_id"])
    s = cleaned[(cleaned.store_id == "ST002") & (cleaned.product_id == "P005")].demand_units
    assert st.loc[("ST002", "P005"), "demand_mean"] == pytest.approx(s.mean())
    assert st.loc[("ST002", "P005"), "demand_std"] == pytest.approx(s.std())
    assert not st.demand_fallback.any()
    recent = ssm.demand_stats(cleaned, window_days=60).set_index(["store_id", "product_id"])
    assert recent.loc[("ST002", "P005"), "demand_mean"] == pytest.approx(s.tail(60).mean())
    assert (recent.demand_obs == 60).all()


def test_short_history_uses_documented_fallback(cleaned):
    st = ssm.demand_stats(cleaned, window_days=14)                 # 14 < MIN_DEMAND_OBS
    assert st.demand_fallback.all() and st.demand_std.notna().all() and (st.demand_std > 0).all()
    few = cleaned[cleaned.date <= "2024-01-20"]                    # almost no orders yet
    lt = ssm.lead_time_stats(few)
    assert lt.lt_fallback.all()
    assert (lt.lt_mean == lt.supplier_avg_lead_time_days).all() and (lt.lt_std == lt.supplier_lead_time_std_days).all()


def test_lead_time_stats_match_observed_orders(cleaned):
    lt = ssm.lead_time_stats(cleaned).set_index(["store_id", "product_id"])
    o = cleaned[(cleaned.store_id == "ST001") & (cleaned.product_id == "P001") & (cleaned.order_quantity_units > 0)]
    assert lt.loc[("ST001", "P001"), "lt_mean"] == pytest.approx(o.lead_time_days_for_order.mean())
    assert lt.loc[("ST001", "P001"), "lt_std"] == pytest.approx(o.lead_time_days_for_order.std())
    assert (lt.lt_mean > 0).all()


@pytest.mark.parametrize("method", ["classic", "dynamic"])
def test_policy_table_is_consistent(cleaned, method):
    if method == "dynamic" and not fs.artefacts_ready():
        pytest.skip("no forecast artefacts")
    p = inv.policy_table(cleaned, 0.95, method)
    assert len(p) == 100 and not p[["safety_stock", "reorder_point", "demand_mean", "demand_std"]].isna().any().any()
    np.testing.assert_allclose(p.reorder_point, p.demand_mean * p.lt_mean + p.safety_stock)
    np.testing.assert_allclose(p.safety_stock, p.z * np.sqrt(p.demand_std ** 2 * p.lt_mean + p.demand_mean ** 2 * p.lt_std ** 2))
    assert (p.safety_stock > 0).all() and (p.recommended_order_qty >= 0).all()
    need = p.inventory_position_units <= p.reorder_point
    assert ((p.recommended_order_qty > 0) == need).all()             # order suggested exactly when position <= ROP
    assert (p.loc[need, "inventory_position_units"] + p.loc[need, "recommended_order_qty"]
            >= p.loc[need, "order_up_to"] - 1e-9).all()
    assert set(p.status) <= set(inv.STATUS_ORDER)


def test_higher_service_level_means_more_safety_stock(cleaned):
    levels = [0.90, 0.95, 0.975, 0.99]
    totals = [inv.policy_table(cleaned, lv, "classic").safety_stock.sum() for lv in levels]
    assert totals == sorted(totals) and totals[0] < totals[-1]
    curve = ssm.service_level_curve(20, 5, 4, 1)
    assert curve.safety_stock.is_monotonic_increasing


def test_rolling_reorder_point(small):
    one = small[small.product_id == "P001"]
    roll = ssm.rolling_reorder_point(one, 0.95, 60, 3.0, 1.4)
    last = one.sort_values("date").demand_units.tail(60)
    expected = last.mean() * 3.0 + ssm.safety_stock(ssm.z_score(0.95), last.std(), 3.0, last.mean(), 1.4)
    assert roll.reorder_point.iloc[-1] == pytest.approx(expected)
    assert len(roll) == len(one) - 59


def test_invalid_method_raises(cleaned):
    with pytest.raises(ValueError):
        inv.policy_table(cleaned, 0.95, "magic")
