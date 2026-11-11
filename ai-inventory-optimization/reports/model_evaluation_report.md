# Demand Forecasting - Model Evaluation Report

Target: `demand_units` (unconstrained demand), one row per store-product-day. All numbers below are computed by `src/forecasting/train.py`; nothing is typed in by hand.

## Protocol

- Train: up to 2025-06-30 (51,900 rows)
- Validation: to 2025-09-30 (9,200 rows) - used to choose the model
- Test: 2025-10-01 to 2025-12-31 (9,200 rows) - used once, for reporting
- Splits are by date; no shuffling. Features for a day use demand only up to the previous day.
- Models predict demand relative to the series' trailing 28-day average, then scale back to units.
- Holt-Winters: parameters estimated before the evaluated period, then run one day ahead.

## One-day-ahead accuracy (sorted by validation WAPE)

| model | type | validation_wape | test_wape | test_mae | test_rmse | test_mape | test_bias |
|---|---|---|---|---|---|---|---|
| Linear Regression | machine learning | 0.1901 | 0.1817 | 4.2658 | 5.7314 | 0.2280 | -0.5424 |
| XGBoost | machine learning | 0.1909 | 0.1831 | 4.2990 | 5.7410 | 0.2353 | -0.0323 |
| Random Forest | machine learning | 0.1913 | 0.1838 | 4.3144 | 5.7624 | 0.2363 | 0.0373 |
| Holt-Winters (weekly) | statistical | 0.2043 | 0.2064 | 4.8459 | 6.7324 | 0.2481 | -0.9552 |
| Moving average (28 days) | baseline | 0.2099 | 0.2131 | 5.0011 | 6.9408 | 0.2575 | -0.8252 |
| Moving average (7 days) | baseline | 0.2165 | 0.2204 | 5.1727 | 7.0718 | 0.2715 | -0.2304 |
| Same weekday average (4 weeks) | baseline | 0.2250 | 0.2268 | 5.3226 | 7.3481 | 0.2716 | -1.0044 |
| Naive (yesterday) | baseline | 0.2836 | 0.2838 | 6.6620 | 9.1252 | 0.3427 | -0.0604 |

Model selected on validation: **Linear Regression**.

## Before AI vs After AI (test period)

| metric | naive | before_ai_28d_average | after_ai | change_vs_before |
|---|---|---|---|---|
| WAPE | 0.2838 | 0.2131 | 0.1817 | -14.7% |
| MAE (units) | 6.6620 | 5.0011 | 4.2658 | -14.7% |
| RMSE (units) | 9.1252 | 6.9408 | 5.7314 | -17.4% |

'After AI' = Linear Regression. 'Before AI' = Moving average (28 days), the average-demand input of a classic reorder-point calculator.

The improvement is real but moderate: daily demand in this dataset is noisy (variance about 1.9x the mean), so a large part of the error cannot be removed by any model.

## Multi-day forecasts (recursive, 30 days ahead, test period)

- WAPE after AI: 0.1824
- WAPE before AI (flat 28-day average): 0.2218

WAPE by forecast week:

| week | wape |
|---|---|
| 1 | 0.1851 |
| 2 | 0.1845 |
| 3 | 0.1839 |
| 4 | 0.1782 |
| 5 | 0.1763 |

## Prediction intervals (80%)

Built from validation-period errors only, then checked on the test period:

- One day ahead: 76.9% of test actuals fall inside the interval
- Up to 30 days ahead: 81.3%

Multipliers applied to the forecast, by forecast week:

| week | lower_factor | upper_factor |
|---|---|---|
| 1 | 0.6991 | 1.3950 |
| 2 | 0.7032 | 1.3873 |
| 3 | 0.6970 | 1.3771 |
| 4 | 0.6856 | 1.4033 |
| 5 | 0.6836 | 1.4161 |

## Assumptions and limits

- Promotions and holidays are treated as known in advance. Future forecasts default to no promotion.
- Competitor price is not known in advance; the last observed price ratio is carried forward.
- Weather is excluded: it is inconsistent within a city-day and shows no demand effect.
- Only two years of history: annual seasonality is learned from a single full cycle.
- LSTM was not used: 731 points per series, and tabular models already sit close to the noise floor.