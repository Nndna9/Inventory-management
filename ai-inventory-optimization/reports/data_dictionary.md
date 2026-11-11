# Data Dictionary

| column | type | role | description | source |
|---|---|---|---|---|
| date | date | key | Calendar day (daily grain, 2024-01-01 to 2025-12-31) | original |
| store_id | str | key | Store identifier (5 stores) | original |
| store_city | str | key | Store city (UAE) | original |
| product_id | str | key | Product identifier (20 products) | original |
| product_name | str | key | Product name | original |
| category | str | known | Product category (4 categories) | original |
| supplier_id | str | policy | Supplier serving this store-product (fixed per store-product) | original |
| unit_cost_aed | num | policy | Purchase cost per unit, AED (fixed per product) | original |
| selling_price_aed | num | known | Selling price after discount, AED | original |
| competitor_price_aed | num | lag_only | Competitor price that day, AED | original |
| promotion_flag | int | known | 1 if a promotion ran that day (assumed planned in advance) | original |
| discount_pct | num | known | Discount percentage applied (0, 5, 10, 15, 20) | original |
| holiday_flag | int | known | 1 on holiday dates | original |
| day_of_week | int | known | 0 = Monday ... 6 = Sunday | original |
| weekend_flag | int | known | 1 on Friday and Saturday | original |
| weather_condition | str | excluded | Weather label. Inconsistent within a city-day and no demand effect; not used | original |
| annual_seasonality_index | num | known | Deterministic seasonal index by calendar date | original |
| demand_units | int | target | Unconstrained customer demand, units. FORECAST TARGET | original |
| units_sold | int | excluded | Fulfilled sales = min(demand, available stock) | original |
| lost_sales_units | int | excluded | Demand not served = demand - units sold | original |
| opening_inventory_units | int | excluded | On-hand stock at start of day | original |
| received_units | int | excluded | Units received that day (before sales) | original |
| closing_inventory_units | int | excluded | On-hand stock at end of day | original |
| outstanding_order_units | int | excluded | Units on order, not yet received, before that day's ordering decision | original |
| reorder_point_baseline_units | int | policy | Existing reorder point (fixed per store-product) | original |
| order_up_to_level_units | int | policy | Existing order-up-to level (fixed per store-product) | original |
| reorder_trigger_flag | int | excluded | 1 if an order was placed at end of day | original |
| order_quantity_units | int | excluded | Units ordered at end of day | original |
| lead_time_days_for_order | num | policy | Realised lead time of the order placed that day (blank if no order) | original |
| expected_arrival_date | date | policy | Order date + lead time (blank if no order) | original |
| supplier_avg_lead_time_days | int | policy | Supplier mean lead time, days | original |
| supplier_lead_time_std_days | num | policy | Supplier lead-time standard deviation, days | original |
| stockout_flag | int | excluded | DERIVED: 1 if lost_sales_units > 0 | derived |
| inventory_position_units | int | excluded | DERIVED: closing stock + outstanding orders (before that day's order) | derived |
| list_price_aed | num | known | DERIVED: pre-discount price = selling price / (1 - discount) | derived |
| unit_margin_aed | num | policy | DERIVED: selling price - unit cost | derived |
