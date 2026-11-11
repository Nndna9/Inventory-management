"""Headless dashboard tests: every page renders without an exception and shows backend numbers."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.services.data_service import baseline_kpis

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["overview", "forecasting", "inventory", "safety-stock", "simulation", "optimization", "bullwhip",
         "recommendations"]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_overview_renders_without_errors(app):
    assert not app.exception, app.exception
    assert app.title[0].value == "Executive Overview"


def test_overview_metrics_match_backend(app, cleaned):
    k = baseline_kpis(cleaned)
    shown = {m.label: m.value for m in app.metric}
    assert shown["Inventory on hand (units)"] == f"{k['inventory_units']:,}"
    assert shown["Fill rate in period"] == f"{k['fill_rate']:.1%}"
    assert shown["Stockout events in period"] == f"{k['stockout_events']:,}"
    assert shown["Errors"] == "0"


def test_app_registers_all_eight_pages():
    src = Path(APP).read_text(encoding="utf-8")
    assert src.count("st.Page(") == 8
    for url in PAGES:
        assert f'url_path="{url}"' in src


def test_pending_page_renders_and_claims_nothing():
    def page():
        from dashboard.common import pending
        pending("Monte Carlo Risk Simulation", "Phase 2", "Stockout risk.")

    at = AppTest.from_function(page, default_timeout=60)
    at.run()
    assert not at.exception
    assert "Phase 2" in at.info[0].value
    assert len(at.metric) == 0  # no numbers shown before the backend exists


def test_filters_change_metrics_to_match_backend(cleaned):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.multiselect(key="f_store").set_value(["Dubai"]).run()
    assert not at.exception
    k = baseline_kpis(cleaned[cleaned.store_city == "Dubai"])
    shown = {m.label: m.value for m in at.metric}
    assert shown["Inventory on hand (units)"] == f"{k['inventory_units']:,}"
    assert shown["Total demand in period (units)"] == f"{k['total_demand']:,}"


def test_empty_filter_shows_warning_not_error():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.multiselect(key="f_store").set_value([]).run()
    assert not at.exception
    assert "No data matches" in at.warning[0].value
    assert len(at.metric) == 0


def test_alert_slider_limits_rows():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.slider(key="overview_top_n").set_value(5).run()
    assert not at.exception
    assert any("Showing 5 of" in c.value for c in at.caption)


# ---- Demand Forecasting page ----------------------------------------------------------------

def _forecasting_page():
    from dashboard import forecasting
    forecasting.render()


@pytest.fixture(scope="module")
def fc_page():
    from src.services import forecast_service as fs
    if not fs.artefacts_ready():
        pytest.skip("run scripts/run_pipeline.py first")
    at = AppTest.from_function(_forecasting_page, default_timeout=180)
    at.run()
    return at


def _number(text):
    return float(text.replace(",", "").replace("%", ""))


def test_forecasting_page_renders_and_matches_backend(fc_page, cleaned):
    from src.services import forecast_service as fs
    assert not fc_page.exception, fc_page.exception
    meta = fs.load_meta()
    live = fs.forecast_series(cleaned, fs.load_model(), meta, fs.load_factors(), "ST001", "P001", 30)
    shown = {m.label: m.value for m in fc_page.metric}
    assert shown["Forecast demand, next 30 days"] == f"{live.forecast.sum():,.0f}"
    acc = fs.series_accuracy(fs.load_predictions(), meta, "ST001", "P001")
    assert shown["Forecast error on test (WAPE)"] == f"{acc['after']['wape']:.1%}"
    table = fc_page.dataframe[0].value
    assert table.loc[1, "After AI"] == f"{meta['test_metrics_after']['wape']:.1%}"
    assert table.loc[1, "Before AI (existing rule)"] == f"{meta['test_metrics_before']['wape']:.1%}"


def test_forecasting_controls_update_the_forecast(fc_page):
    at = fc_page
    base = _number({m.label: m.value for m in at.metric}["Forecast demand, next 30 days"])
    at.slider(key="fc_horizon").set_value(60).run()
    assert not at.exception
    longer = _number({m.label: m.value for m in at.metric}["Forecast demand, next 60 days"])
    assert longer > base
    at.toggle(key="fc_promo").set_value(True).run()
    at.select_slider(key="fc_discount").set_value(10).run()
    assert not at.exception
    promo = _number({m.label: m.value for m in at.metric}["Forecast demand, next 60 days"])
    assert promo > longer
    at.selectbox(key="fc_product").set_value("P006").run()
    at.selectbox(key="fc_store").set_value("ST003").run()
    assert not at.exception


# ---- Inventory pages --------------------------------------------------------------------------

def _safety_page():
    from dashboard import safety_stock
    safety_stock.render()


def _inventory_page():
    from dashboard import inventory
    inventory.render()


def test_safety_stock_page_matches_backend_and_reacts_to_service_level(cleaned):
    from src.services import inventory_service as inv
    at = AppTest.from_function(_safety_page, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    method = at.radio(key="policy_method").value
    row = inv.policy_table(cleaned, 0.95, method).query("store_id == 'ST001' and product_id == 'P001'").iloc[0]
    shown = {m.label: m.value for m in at.metric}
    assert shown["Safety stock (units)"] == f"{row.safety_stock:.0f}"
    assert shown["Reorder point (units)"] == f"{row.reorder_point:.0f}"
    at.select_slider(key="policy_service_level").set_value(0.99).run()
    assert not at.exception
    higher = inv.policy_table(cleaned, 0.99, method).query("store_id == 'ST001' and product_id == 'P001'").iloc[0]
    shown = {m.label: m.value for m in at.metric}
    assert shown["Safety stock (units)"] == f"{higher.safety_stock:.0f}"
    assert higher.safety_stock > row.safety_stock
    at.radio(key="policy_method").set_value("classic").run()
    at.slider(key="policy_window").set_value(90).run()
    assert not at.exception
    classic = inv.policy_table(cleaned, 0.99, "classic", 90).query("store_id == 'ST001' and product_id == 'P001'").iloc[0]
    assert {m.label: m.value for m in at.metric}["Safety stock (units)"] == f"{classic.safety_stock:.0f}"


def test_inventory_page_matches_backend_and_filters(cleaned):
    from src.services import inventory_service as inv
    at = AppTest.from_function(_inventory_page, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    method = at.radio(key="policy_method").value
    s = inv.summary(inv.policy_table(cleaned, 0.95, method))
    shown = {m.label: m.value for m in at.metric}
    assert shown["Reorder now"] == str(s["reorder_now"])
    assert shown["Recommended orders (units)"] == f"{s['order_units']:,}"
    at.multiselect(key="inv_status").set_value(["Healthy"]).run()
    at.slider(key="inv_cover").set_value(1.0).run()
    at.multiselect(key="f_store").set_value(["Sharjah"]).run()
    assert not at.exception
    sharjah = inv.summary(inv.policy_table(cleaned[cleaned.store_city == "Sharjah"], 0.95, method))
    assert {m.label: m.value for m in at.metric}["Reorder now"] == str(sharjah["reorder_now"])
