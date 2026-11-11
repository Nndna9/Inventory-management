# Data Quality Report

Source: `data/raw/synthetic_ai_inventory_data_2024_2025.csv` (original synthetic dataset, unmodified).

- Rows: 73,100 | Columns: 32
- Stores: 5 | Products: 20 | Suppliers: 8 | Days: 731 (2024-01-01 to 2025-12-31)
- Checks passed: 18 of 20 (errors failed: 0, warnings: 2)

## Checks

| check | severity | passed | violations | detail |
|---|---|---|---|---|
| No missing values in required columns | error | True | 0 | Order columns are blank on no-order days by design |
| No duplicate rows | error | True | 0 |  |
| One row per store-product-day | error | True | 0 |  |
| Complete daily panel (no gaps) | error | True | 0 | 100 series x 731 days |
| No negative quantities or prices | error | True | 0 |  |
| units_sold + lost_sales = demand | error | True | 0 |  |
| units_sold = min(demand, opening + received) | error | True | 0 | Receipts arrive before the day's sales |
| closing = opening + received - sold | error | True | 0 |  |
| opening = previous day's closing | error | True | 0 |  |
| Order placed exactly when trigger flag = 1 | error | True | 0 |  |
| Lead time present exactly when an order is placed | error | True | 0 |  |
| Lead times positive | error | True | 0 | range 1-16 days |
| expected_arrival = order date + lead time | error | True | 0 |  |
| Order triggered exactly when inventory position <= reorder point | error | True | 0 | Confirms the existing policy: daily-review (s, S) |
| Order quantity = order-up-to level - inventory position | error | True | 0 |  |
| Every due order is received in full on its arrival date | error | True | 0 | 63 orders still in transit at end of data |
| Reorder point and order-up-to level fixed per store-product | warning | True | 0 |  |
| Discount only when promotion flag = 1 | warning | True | 0 |  |
| Promotions always carry a discount | warning | False | 5095 | Promotion flag and discount are treated as separate features |
| Weather consistent within a city-day | warning | False | 3655 | Weather differs across products in the same city on the same day; excluded from models |

## Cleaning log

| step | rows |
|---|---|
| rows_in | 73100 |
| exact_duplicates_dropped | 0 |
| duplicate_keys_dropped | 0 |
| missing_key_or_demand_dropped | 0 |
| negative_values_clipped | 0 |
| rows_out | 73100 |

## Missing values

| column | missing |
|---|---|
| lead_time_days_for_order | 65472 |
| expected_arrival_date | 65472 |
