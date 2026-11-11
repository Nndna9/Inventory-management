# Exploratory Data Analysis

All figures computed from the cleaned dataset.

## Demand distribution across the 100 store-product series

| index | mean | std | cv | variance_to_mean | skew |
|---|---|---|---|---|---|
| mean | 23.123 | 6.528 | 0.303 | 1.879 | 0.567 |
| min | 8.179 | 3.256 | 0.227 | 1.230 | 0.262 |
| 50% | 20.137 | 5.950 | 0.295 | 1.739 | 0.565 |
| max | 60.715 | 14.303 | 0.416 | 3.386 | 0.880 |

Variance-to-mean above 1 and positive skew: demand is overdispersed and right-skewed, not normal.

## Demand drivers (1.00 = a typical day for that series)

| driver | value | relative_demand | rows |
|---|---|---|---|
| promotion_flag | 0 | 0.972 | 66039 |
| promotion_flag | 1 | 1.265 | 7061 |
| discount_pct | 0.0 | 0.992 | 71134 |
| discount_pct | 5.0 | 1.279 | 550 |
| discount_pct | 10.0 | 1.277 | 705 |
| discount_pct | 15.0 | 1.258 | 516 |
| discount_pct | 20.0 | 1.278 | 195 |
| holiday_flag | 0 | 0.994 | 70100 |
| holiday_flag | 1 | 1.139 | 3000 |
| weekend_flag | 0 | 0.959 | 52300 |
| weekend_flag | 1 | 1.102 | 20800 |
| day_of_week | 0 | 0.960 | 10500 |
| day_of_week | 1 | 0.957 | 10500 |
| day_of_week | 2 | 0.961 | 10500 |
| day_of_week | 3 | 0.962 | 10400 |
| day_of_week | 4 | 1.102 | 10400 |
| day_of_week | 5 | 1.103 | 10400 |
| day_of_week | 6 | 0.956 | 10400 |
| weather_condition | Cloudy | 1.000 | 14508 |
| weather_condition | Hot | 1.016 | 23580 |
| weather_condition | Rainy | 0.961 | 7352 |
| weather_condition | Sunny | 0.997 | 27660 |

## Stockouts and lost sales by product

| product_id | product_name | demand | sold | lost | stockout_days | days | fill_rate | stockout_day_rate |
|---|---|---|---|---|---|---|---|---|
| P004 | Laptop Stand | 49839 | 48793 | 1046 | 103 | 3655 | 0.979 | 0.028 |
| P020 | Fitness Gloves | 61783 | 60569 | 1214 | 106 | 3655 | 0.980 | 0.029 |
| P003 | Bluetooth Speaker | 34019 | 33378 | 641 | 86 | 3655 | 0.981 | 0.024 |
| P015 | Kitchen Scale | 46570 | 45727 | 843 | 82 | 3655 | 0.982 | 0.022 |
| P014 | Cushion Cover | 58417 | 57363 | 1054 | 83 | 3655 | 0.982 | 0.023 |
| P007 | Ballpoint Pen Pack | 173878 | 170901 | 2977 | 76 | 3655 | 0.983 | 0.021 |
| P019 | Skipping Rope | 89588 | 88103 | 1485 | 90 | 3655 | 0.983 | 0.025 |
| P006 | Notebook A5 | 190546 | 187452 | 3094 | 82 | 3655 | 0.984 | 0.022 |
| P001 | Wireless Mouse | 63769 | 62740 | 1029 | 78 | 3655 | 0.984 | 0.021 |
| P005 | Power Bank | 42175 | 41506 | 669 | 74 | 3655 | 0.984 | 0.020 |
| P002 | USB-C Cable | 128358 | 126344 | 2014 | 79 | 3655 | 0.984 | 0.022 |
| P012 | Storage Box | 67904 | 66855 | 1049 | 79 | 3655 | 0.985 | 0.022 |
| P011 | Water Bottle | 91326 | 89948 | 1378 | 81 | 3655 | 0.985 | 0.022 |
| P010 | Marker Set | 100016 | 98558 | 1458 | 77 | 3655 | 0.985 | 0.021 |
| P013 | LED Desk Lamp | 39555 | 39000 | 555 | 65 | 3655 | 0.986 | 0.018 |
| P009 | Printer Paper | 103424 | 101992 | 1432 | 69 | 3655 | 0.986 | 0.019 |
| P008 | Desk Organizer | 78645 | 77578 | 1067 | 69 | 3655 | 0.986 | 0.019 |
| P017 | Resistance Bands | 89810 | 88600 | 1210 | 75 | 3655 | 0.987 | 0.021 |
| P018 | Sports Bottle | 125808 | 124147 | 1661 | 77 | 3655 | 0.987 | 0.021 |
| P016 | Yoga Mat | 54891 | 54226 | 665 | 67 | 3655 | 0.988 | 0.018 |

## Stockouts by store

| store_id | demand | sold | lost | stockout_days | days | fill_rate | stockout_day_rate |
|---|---|---|---|---|---|---|---|
| ST005 | 297696 | 292301 | 5395 | 349 | 14620 | 0.982 | 0.024 |
| ST004 | 273648 | 268985 | 4663 | 328 | 14620 | 0.983 | 0.022 |
| ST002 | 384060 | 377949 | 6111 | 336 | 14620 | 0.984 | 0.023 |
| ST003 | 319413 | 314647 | 4766 | 313 | 14620 | 0.985 | 0.021 |
| ST001 | 415504 | 409898 | 5606 | 272 | 14620 | 0.987 | 0.019 |

## Supplier lead times: observed vs dataset parameters

| supplier_id | orders | observed_mean | observed_std | min | max | supplier_avg_lead_time_days | supplier_lead_time_std_days |
|---|---|---|---|---|---|---|---|
| SUP001 | 716 | 3.042 | 1.368 | 1.000 | 7.000 | 3 | 1.410 |
| SUP002 | 963 | 5.998 | 1.985 | 1.000 | 16.000 | 6 | 2.000 |
| SUP003 | 811 | 2.975 | 0.970 | 1.000 | 6.000 | 3 | 0.930 |
| SUP004 | 1135 | 5.978 | 2.100 | 1.000 | 13.000 | 6 | 2.170 |
| SUP005 | 1123 | 6.053 | 1.959 | 1.000 | 12.000 | 6 | 1.900 |
| SUP006 | 1042 | 5.985 | 1.036 | 3.000 | 9.000 | 6 | 0.980 |
| SUP007 | 955 | 6.923 | 1.400 | 2.000 | 12.000 | 7 | 1.320 |
| SUP008 | 883 | 4.867 | 2.008 | 1.000 | 12.000 | 5 | 2.100 |

Lead time vs recent demand correlation (within supplier): 0.007. Near zero, so demand and lead time are modelled as independent (stated assumption).

## Monthly totals

| month | demand | sold | lost |
|---|---|---|---|
| 2024-01 | 66613 | 66458 | 155 |
| 2024-02 | 67504 | 66962 | 542 |
| 2024-03 | 76334 | 75405 | 929 |
| 2024-04 | 73923 | 72178 | 1745 |
| 2024-05 | 74420 | 73472 | 948 |
| 2024-06 | 67929 | 67273 | 656 |
| 2024-07 | 64041 | 63745 | 296 |
| 2024-08 | 59647 | 59445 | 202 |
| 2024-09 | 54753 | 54694 | 59 |
| 2024-10 | 57137 | 56662 | 475 |
| 2024-11 | 65942 | 65378 | 564 |
| 2024-12 | 70763 | 70164 | 599 |
| 2025-01 | 74798 | 73851 | 947 |
| 2025-02 | 72815 | 70431 | 2384 |
| 2025-03 | 85409 | 81581 | 3828 |
| 2025-04 | 82987 | 79555 | 3432 |
| 2025-05 | 83901 | 81119 | 2782 |
| 2025-06 | 74981 | 73559 | 1422 |
| 2025-07 | 71880 | 71023 | 857 |
| 2025-08 | 67275 | 67033 | 242 |
| 2025-09 | 61321 | 61226 | 95 |
| 2025-10 | 63919 | 63774 | 145 |
| 2025-11 | 73127 | 72274 | 853 |
| 2025-12 | 78902 | 76518 | 2384 |
