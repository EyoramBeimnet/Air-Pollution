# NYC Air-Pollution Model: 2014+ Data Extension Handoff

- **Repository:** `EyoramBeimnet/Air-Pollution`
- **Branch:** `expanded-data` (not merged to `main`)
- **Date:** November 3, 2026

## 1. Goal

Extend the history of the borough-year, next-year AQI model from 2019 back to 2014, then improve its accuracy. The earlier model had very little data to learn from: only 25 to 30 usable rows.

## 2. Summary of Work Completed

### 2.1 Data extended back to 2014

| Source | What was done |
|---|---|
| Weather | Daily Open-Meteo data for 2014 to 2025 was downloaded by hand for the five boroughs and compiled with `nyc_daily_weather_compiler.py`. `collect_nyc_weather.py` now has a `--from-daily` flag that skips the API and rebuilds `nyc_weather_borough_year_2014_2025.csv` from that daily file. |
| ACS commuting | `collect_historical_acs.py` was failing its uniqueness check because of a leftover hard-coded "30 records" from the old 2019 to 2024 range. It now expects 55. The output file must be named `nyc_acs_commuting_2014_2024.csv` so `prepare_model_data.py` can find it. |
| Tree plantings | Dropped from the models. The data is only reliable from 2016, and the counts partly reflect recording practice rather than actual plantings. |
| MTA ridership | Still passed through the staging table (data starts in 2017), but no model uses it. |

### 2.2 Pipeline scripts updated for 2014+

Many places were hard-coded to 2019, so each stage needed edits.

| Script | Changes |
|---|---|
| `review_ev_and_stage_data.py` | `YEARS` widened to 2014 to 2025. The year check in `normalize_keys()` now uses `YEARS`. Master filename updated. Baseline year filter widened. Tree plantings removed from `BASELINE_FEATURES`. The EV file path stays hard-coded to its own year range, because EV data starts in 2019. |
| `finalize_air_pollution_handoff.py` | Master grid is 60 rows (2014 to 2025). Candidate table is 55 rows (feature years 2014 to 2024). Splits are train 2014 to 2020, validation 2021 to 2022, test 2023 to 2024. Tree plantings removed from the feature lists and checks. Report text and printed summaries now use computed values. |
| `train_aqi_models.py` | Extended splits and year filter widened to 2014 to 2024. `ROLLING_MIN_TRAIN_YEARS` set to 5 for both experiments so the rolling backtests cover the same years. Tree plantings removed. Ablation experiments added (see 2.3). |
| `evaluate_aqi_predictions.py` | No changes needed. |

A hard-coded year filter in `main()` of the training script kept the Extended experiment at 30 rows until it was fixed, so check it if the row count looks wrong.

### 2.3 Ablation experiments

`train_aqi_models.py` now runs a base feature set (last observed AQI plus CAMPD NOx) with one feature group added at a time: weather, and drive-alone commuting percentage. Experiment names avoid colons because the script builds file names from them, and Windows does not allow colons in file names.

## 3. How to Rerun the Pipeline

Run from the project root. Each stage reads the output of the previous stage, so keep this order:

```powershell
python scripts\collect_nyc_weather.py --from-daily
python scripts\prepare_model_data.py
python scripts\review_ev_and_stage_data.py
python scripts\finalize_air_pollution_handoff.py
python scripts\train_aqi_models.py
python scripts\evaluate_aqi_predictions.py
```

Before the first run, confirm that `datasets/raw/acs/nyc_acs_commuting_2014_2024.csv` exists and that the old `nyc_acs_commuting_2019_2024.csv` has been removed.

Expected sizes after a successful run:

- Staging table: 55 supervised rows (5 boroughs x 11 feature years, 2014 to 2024)
- Splits: 35 train, 10 validation, 10 test
- Rolling backtest: 30 rows per experiment

## 4. Results

Rolling backtest across 30 rows (6 test years, 2019 to 2024), shown as improvement over the no-change baseline (baseline MAE 3.433):

| Features | Random Forest | XGBoost |
|---|---|---|
| Historical core (AQI, CAMPD, traffic) | +2.8% | +10.3% |
| AQI + CAMPD only | +6.7% | +11.8% |
| AQI + CAMPD + weather | +7.9% | **+16.3%** |
| AQI + CAMPD + drive-alone | +2.7% | +10.4% |
| Extended (weather + drive-alone) | +5.0% | +14.3% |

Test and validation results for XGBoost, same comparison:

| Features | Test (10 rows, baseline 4.0) | Validation (10 rows, baseline 2.2) |
|---|---|---|
| AQI + CAMPD only | +28.8% | +4.0% |
| AQI + CAMPD + weather | +27.2% | +11.9% |
| AQI + CAMPD + drive-alone | +23.0% | +5.7% |
| Extended (weather + drive-alone) | +25.2% | +4.9% |
| Historical core | +21.8% | +9.0% |

### Findings

- **Weather helps.** It is the only feature group that improves both the rolling backtest and validation. XGBoost with AQI, CAMPD, and weather is the best candidate so far.
- **Drive-alone percentage does not help.** Adding it lowers the backtest score each time it is added. The apparent gain of the Extended experiment over Historical core came from weather, not from ACS commuting data.
- **Traffic does not help.** The traffic sample is also flagged as nonrepresentative in the finalize audit.
- **XGBoost beats Random Forest** in every experiment.
- **Earlier results were misleading.** The previous "20% better" Extended figure came from only three backtest folds. The six-fold numbers above are the reliable ones.

## 5. Caveats

- The dataset has only 55 rows. The test set is effectively two independent feature years, because all five boroughs move together in a given year. Differences of about 0.1 AQI points of MAE are within noise.
- ACS five-year estimates may not be published in time for a real next-year forecast, so `drive_alone_pct` must be removed or lagged in any deployment.
- Traffic observations are a nonrepresentative sample with year-varying road segments.
- EV registrations are missing for 2024 to 2025. Do not treat the missing values as zero or backfill from current snapshots.
- No scalers or imputers are fitted during the finalize stage. The audit is structural only and does not certify the model as leakage-free or deployment-ready.
- Do not commit API keys or very large raw downloads to GitHub.

## 6. Open Items

1. **Decide whether to drop drive-alone.** The ablation suggests removing it, leaving AQI, CAMPD, and weather. This is a one-line change to `EXTENDED_FEATURES` in `train_aqi_models.py`, plus matching edits in `finalize_air_pollution_handoff.py` (the `FEATURES` list, the 0 to 100 range check, and the ACS warning).
2. **Increase the sample size.** This is the largest remaining opportunity:
   - Build a county-month table with monthly AQI as the target, monthly weather, and annual features repeated across each year's months. This would give roughly 700 rows.
   - Add neighboring counties (Hudson, Essex, and Bergen in New Jersey, plus Nassau, Suffolk, and Westchester), since every source is keyed by county FIPS.
3. **Merge to `main`** only after the new results are confirmed.

---

# Air Pollution Project — Monthly AQI Pipeline Progress

Last Updated: November 10, 2026\
Project Directory: `D:\Air-Pollution`\
Git Branch: `expanded-data`\
Scope: Build a monthly AQI prediction pipeline using monthly weather features while keeping the existing annual pipeline unchanged.

## 1. Project Objective

Extend the existing annual AQI prediction project with a separate monthly forecasting workflow.

The monthly workflow should:

- Predict monthly AQI for the five NYC boroughs.
- Use historical AQI and monthly weather features.
- Forecast one month ahead.
- Use chronological training, validation, and test splits.
- Compare Random Forest and XGBoost against baseline models.
- Preserve the existing annual dataset, scripts, and results.

### NYC Borough and County FIPS

| Borough                  | County FIPS |
| ------------------------ | ----------- |
| Bronx                    | 36005       |
| Brooklyn (Kings)         | 36047       |
| Manhattan (New York)     | 36061       |
| Queens                   | 36081       |
| Staten Island (Richmond) | 36085       |

## 2. Data Sources Located

### Daily AQI Dataset

File:

`datasets\raw\daily_aqi\nyc_daily_aqi_2014_2025.csv`

This dataset was downloaded from EPA daily AQI data for 2014–2025.

### Monthly AQI Targets

File:

`datasets\processed\monthly\monthly_aqi_targets.csv`

Columns:

- county_fips
- borough
- target_month
- aqi_median_daily
- aqi_mean_daily
- aqi_days_observed
- calendar_days_in_month
- aqi_coverage_pct
- meets_coverage

Example:

```
county_fips,borough,target_month,aqi_median_daily,aqi_mean_daily,aqi_days_observed,calendar_days_in_month,aqi_coverage_pct,meets_coverage
36005,Bronx,2014-01-01,65.0,66.74193548387096,31,31,1.0,True
36005,Bronx,2014-02-01,60.0,62.82142857142857,28,28,1.0,True
```

### Daily Weather Dataset

File:

`datasets\raw\weather\nyc_weather_daily_2014_2025.csv`

Columns:

- county_fips
- borough
- date
- temp_mean_c
- temp_max_c
- precipitation_mm
- wind_speed_mean_kmh
- relative_humidity_mean_pct
- sample_latitude
- sample_longitude
- year

Example:

```
county_fips,borough,date,temp_mean_c,temp_max_c,precipitation_mm,wind_speed_mean_kmh,relative_humidity_mean_pct,sample_latitude,sample_longitude,year
36005,Bronx,2014-01-01,-1.7,1.2,0.0,11.2,58,40.75,-73.75,2014
36005,Bronx,2014-01-02,-2.5,0.4,5.0,23.8,82,40.75,-73.75,2014
```

Other weather files found:

- `datasets\processed\clean_weather_borough_year.csv`
- `datasets\raw\weather\nyc_weather_borough_year_2014_2025.csv`
- `datasets\raw\weather\nyc_weather_borough_year_2019_2025.csv`
- `datasets\raw\weather\nyc_weather_daily_2019_2025.csv`
- Individual county weather files under `datasets\raw\weather\individual_counties_2014-2025\`

The combined daily weather file is the selected source for monthly aggregation.

## 3. Monthly Pipeline Files

The following files currently exist under `datasets\processed\monthly`:

```
monthly_aqi_coverage_audit.csv
monthly_aqi_model_STAGING.csv
monthly_aqi_targets.csv

model_results\
    evaluation_by_borough.csv
    evaluation_by_month.csv
    evaluation_by_month_of_year.csv
    evaluation_predictions.csv
    evaluation_summary.csv
    features_used.csv
    predictions.csv
    training_metrics.csv
```

### Current Staging Dataset

File:

`datasets\processed\monthly\monthly_aqi_model_STAGING.csv`

Columns:

- county_fips
- borough
- target_month
- aqi_median_daily
- aqi_mean_daily
- aqi_days_observed
- calendar_days_in_month
- aqi_coverage_pct
- meets_coverage
- aqi_lag_1
- aqi_lag_12
- issue_month
- target_month_number
- target_year

### Current Model Features

File:

`datasets\processed\monthly\model_results\features_used.csv`

Current features:

```
aqi_lag_1
aqi_lag_12
target_month_number
```

Important: Monthly weather has not yet been incorporated into the training dataset.

## 4. Monthly Pipeline Execution

The following commands were previously executed successfully:

```
py -m pip install pandas numpy scikit-learn xgboost requests

py scripts/download_daily_aqi.py --start-year 2014 --end-year 2025

py scripts/prepare_monthly_aqi.py

py scripts/prepare_monthly_model_data.py

py scripts/train_monthly_aqi_models.py

py scripts/evaluate_monthly_aqi_predictions.py
```

### Daily AQI Download Results

| Year  | Daily Rows |
| ----- | ---------- |
| 2014  | 1,825      |
| 2015  | 1,825      |
| 2016  | 1,830      |
| 2017  | 1,804      |
| 2018  | 1,823      |
| 2019  | 1,825      |
| 2020  | 1,827      |
| 2021  | 1,824      |
| 2022  | 1,824      |
| 2023  | 1,823      |
| 2024  | 1,830      |
| 2025  | 1,825      |
| Total | 21,885     |

### Monthly Dataset Results

- 720 monthly target rows were initially generated.
- 719 rows passed the 75% coverage threshold.
- 719 rows were used in the earlier staging dataset.

### Chronological Splits

| Split      | Date Range         | Rows |
| ---------- | ------------------ | ---- |
| Training   | 2014-02 to 2022-05 | 499  |
| Validation | 2022-06 to 2024-02 | 105  |
| Testing    | 2024-03 to 2025-12 | 110  |

These results are from the original monthly pipeline before weather integration.

## 5. Existing Monthly Model Results

The original monthly models produced the following results:

| Model         | Split      | Rows | MAE    | RMSE   | R²      |
| ------------- | ---------- | ---- | ------ | ------ | ------- |
| Persistence   | Test       | 110  | 5.8545 | 7.7133 | -0.0487 |
| Persistence   | Validation | 105  | 6.8190 | 8.1670 | -0.3252 |
| Random Forest | Test       | 110  | 4.3415 | 5.4233 | 0.4816  |
| Random Forest | Train      | 499  | 3.7249 | 4.7717 | 0.6778  |
| Random Forest | Validation | 105  | 4.5466 | 5.9213 | 0.3034  |
| XGBoost       | Test       | 110  | 4.2117 | 5.1726 | 0.5284  |
| XGBoost       | Train      | 499  | 3.5607 | 4.6090 | 0.6994  |
| XGBoost       | Validation | 105  | 4.3396 | 5.5820 | 0.3809  |

### Initial Interpretation

- XGBoost had the best test performance.
- XGBoost achieved approximately 28.1% lower MAE than the persistence baseline.
- XGBoost achieved approximately 32.9% lower RMSE than the persistence baseline.
- The results are encouraging but do not establish that the model is ready for operational forecasting.
- The test set contains 110 borough-month observations, representing 22 months across five boroughs.

These results should be preserved as the pre-weather baseline.

## 6. Existing Monthly Scripts

The following scripts exist under `scripts`:

```
download_daily_aqi.py
prepare_monthly_aqi.py
prepare_monthly_model_data.py
train_monthly_aqi_models.py
evaluate_monthly_aqi_predictions.py
```

The following two scripts were inspected on November 4, 2026:

### prepare_monthly_model_data.py

Current behavior:

- Reads monthly AQI targets.
- Creates `aqi_lag_1` and `aqi_lag_12`.
- Creates `issue_month`, `target_month_number`, and `target_year`.
- Has an optional weather input.
- Expects weather to already be aggregated into `weather_*` columns.
- Does not currently aggregate the daily weather dataset automatically.

### train_monthly_aqi_models.py

Current behavior:

- Reads the monthly staging dataset.
- Automatically selects eligible numeric features.
- Splits data chronologically.
- Trains Random Forest and XGBoost.
- Evaluates a persistence baseline using `aqi_lag_1`.
- Saves predictions, metrics, and feature lists.



## 7. CNN Experiment Review (added November 10, 2026)
 
We ran a 1D CNN on the monthly data (`scripts/train_monthly_cnn.py`, results written up in `notebooks/cnnFindings.md`, branch `expanded-data`). The CNN beat Persistence, Random Forest, and XGBoost on the held-out test set (5-seed ensemble: test MAE 3.87, RMSE 4.70, R² 0.61, vs. XGBoost MAE 4.14, RMSE 5.09, R² 0.54).
 
This section records the follow-up review: how likely the gain is overfitting, whether the model is more complex than needed, and whether the gain holds up across different time periods.
 
### 7.1 Reasons for concern
 
- **Capacity vs. data.** The full CNN has about 5,100 parameters and trains on 498 rows (about 10 parameters per row). The rows are also correlated (consecutive months within a borough, and five boroughs moving together in a month), so the effective sample size is smaller than 498.
- **The reported edge over XGBoost was thin.** The test bootstrap difference vs. XGBoost had a 90% CI of about -0.74 to -0.00, and on validation the CI included zero.
- **The comparison was not equal.** The CNN was a 5-seed ensemble, while Random Forest and XGBoost were single fits. The CNN also saw a 12-month history, while the trees saw only lag 1, lag 12, and month number.
- **Persistence is a weak baseline.** Its test R² is below zero, so most models that shrink toward the mean beat it by a lot.
- **Single test window.** The original test set was 110 rows from one period (2024-03 to 2025-12).
What the CNN script already did well: chronological splits, train-only scaling, validation used only for early stopping, a residual head (predicts change from last month), L2, dropout, and seed averaging.
 
### 7.2 New scripts
 
Both scripts are run from the project root and need `tensorflow-cpu`, `xgboost`, and `scikit-learn`. They import from `train_monthly_cnn.py`, so keep them in the same `scripts` folder. Neither modifies existing files. Neither uses weather features (the weather input is `None`), so the CNN here sees AQI history, an observed-mask channel, month sine/cosine, and borough.
 
| Script | Purpose | Output |
|---|---|---|
| `scripts/diagnose_cnn_overfit.py` | Trains CNNs of different sizes and window lengths on the same rows and splits as the training script (3 seeds averaged), next to Persistence and a ridge baseline. Reports parameters, best epoch, and train/validation/test RMSE. | Printed table (terminal only) |
| `scripts/rolling_origin_backtest.py` | For each test year 2020 to 2025: validate on the previous year (early stopping only), train on everything earlier. Compares Persistence, ridge, XGBoost, and the CNN (8 filters, window 12). XGBoost uses the same settings and features as `train_monthly_cnn.py` (lag 1, lag 12, month number) and is averaged over the same 3 seeds. | `notebooks/cnn_rolling_backtest.csv` |
 
```powershell
python scripts\diagnose_cnn_overfit.py
python scripts\rolling_origin_backtest.py
```
 
### 7.3 Check 1: model size and window (single split)
 
RMSE in AQI points, 498 train rows, 3-seed average. Two runs on two machines:
 
| Model | Params | Run A Train / Val / Test | Run B Train / Val / Test |
|---|---|---|---|
| Persistence | 0 | 8.09 / 8.17 / 7.71 | 8.09 / 8.17 / 7.71 |
| Ridge (lag1, lag12, month) | 5 | 5.79 / 6.06 / 5.22 | 5.79 / 6.06 / 5.22 |
| CNN filters=16, window=12 | 5137 | 3.92 / 5.16 / 4.77 | 4.17 / 5.32 / 4.85 |
| CNN filters=8, window=12 | 2329 | 3.72 / 5.12 / 4.81 | 3.97 / 5.09 / 4.75 |
| CNN filters=4, window=12 | 1213 | 3.97 / 5.14 / 4.79 | 3.86 / 5.18 / 4.89 |
| CNN filters=8, window=6 | 2329 | 4.06 / 5.41 / 5.43 | 4.15 / 5.44 / 5.53 |
| CNN filters=4, window=3 | 1213 | 4.06 / 5.38 / 5.22 | 4.13 / 5.50 / 5.18 |
 
- **Model size does not matter.** The 1,213-parameter CNN matches the 5,137-parameter one on test. The ranking among 4, 8, and 16 filters flipped between runs, so those differences are seed noise (about +/-0.1 to 0.15 RMSE).
- **Overfitting is mild.** The train-to-test gap for the 12-month models is about 0.7 to 1.1 RMSE (train about 4, test about 4.8). Early stopping chose epochs from 23 to 84.
- **The 12-month window drives the gain.** Windows of 6 and 3 months score about the same as, or worse than, ridge (5.2 to 5.5). The 12-month models are about 0.35 to 0.45 better than ridge.
- **The test period is easier than validation.** Persistence and ridge also score lower on test than on train and validation, so compare models against each other rather than reading absolute test numbers.
### 7.4 Check 2: rolling-origin backtest (six test years)
 
RMSE in AQI points (Run A; Run B on a second machine differed by a few hundredths for Persistence, ridge, and XGBoost, and by up to about 0.2 for the CNN).
 
| Test year | Train rows | Persistence | Ridge | XGBoost | CNN | CNN - Ridge | CNN - XGB |
|---|---|---|---|---|---|---|---|
| 2020 | 293 | 7.99 | 5.87 | 5.49 | 4.78 | -1.08 | -0.71 |
| 2021 | 353 | 7.07 | 5.92 | 5.16 | 6.30 | +0.38 | +1.14 |
| 2022 | 413 | 6.39 | 4.45 | 4.51 | 4.37 | -0.07 | -0.14 |
| 2023 | 473 | 8.88 | 6.70 | 6.05 | 5.76 | -0.93 | -0.28 |
| 2024 | 533 | 7.28 | 5.02 | 4.65 | 4.81 | -0.22 | +0.16 |
| 2025 | 593 | 8.22 | 5.71 | 5.63 | 5.15 | -0.56 | -0.48 |
 
Negative gap means the CNN is better.
 
| | Run A | Run B |
|---|---|---|
| CNN beats ridge | 5 of 6 years, mean gap -0.42 | 4 of 6 years, mean gap -0.35 |
| CNN beats XGBoost | 4 of 6 years, mean gap -0.05 | 3 of 6 years, mean gap +0.02 |
 
- **The CNN's edge over ridge is real but small** (about 0.35 to 0.42 RMSE).
- **XGBoost gets about the same improvement over ridge** (about 0.37 RMSE) from three simple features. Most of the gain comes from a nonlinear model using lag 1, lag 12, and month, not from the CNN's longer window.
- **The CNN and XGBoost are tied.** The mean gap is about zero and the standard deviation (about 0.65) is far larger than the mean. The CNN's run-to-run wobble is as large as its whole margin.
- **The CNN is more volatile.** It has the largest wins (2020, 2025) and the largest loss (2021, about +1.1 to +1.3 vs. XGBoost) in both runs.
- **2020 inflates the CNN's average edge over ridge.** It is the COVID year and the largest CNN win. Excluding it, the Run A mean gap vs. ridge is about -0.28.
- **2021 is the CNN's consistent weak spot.** A possible explanation is that its validation year is 2020, an unusual year used for early stopping. This was not tested.
### 7.5 Conclusions
 
1. The CNN is not badly overfit, but it is more complex than it needs to be.
2. Against simple baselines the CNN gain is real but modest. Against XGBoost it is effectively a tie.
3. **Recommendation:** use XGBoost as the main monthly model (simpler, faster, steadier). Keep the CNN as a documented alternative if the project needs a deep-learning component, and if so use the small version (`filters` 4 or 8, `window` 12).
4. Report rolling-backtest results rather than a single split, and note that 2020 is an unusual year.
5. Freeze hyperparameters before looking at test results again, and record that in `notebooks/cnnFindings.md`.
### 7.6 Limitations
 
- Six overlapping folds with 60 test rows each. Training sets overlap across folds, so the folds are not independent.
- Only 3 seeds per model. Differences smaller than about 0.15 RMSE are within noise.
- No hyperparameter tuning for the CNN or XGBoost.
- The ridge baseline is basic (lag 1, lag 12, month sine and cosine).
- Weather features were not used in these checks.
- The `model_results_cnn/` output files are not committed to the branch, so the first-run results in `cnnFindings.md` were not independently re-checked beyond the runs above.
### 7.7 Files added or changed
 
- `scripts/diagnose_cnn_overfit.py` (new)
- `scripts/rolling_origin_backtest.py` (new)
- `notebooks/cnn_rolling_backtest.csv` (generated by running the backtest)
- `notebooks/cnn_overfitting_and_backtest_analysis.md` (written analysis of sections 7.1 to 7.6)
- `progress.md` (this section added)
## 8. Open Items (monthly CNN)
 
1. Add monthly weather to the rolling backtest. The annual ablations found weather was the only feature group that helped, and the monthly checks above ran without it.
2. Decide whether the CNN stays in the project, given the tie with XGBoost.
3. If keeping the CNN, run more seeds (for example 10) per fold to see whether the 2021 loss is a seed effect, and test whether a different validation year changes the 2021 result.
4. Tune XGBoost and the CNN lightly (same budget for each) before the final comparison.
5. Commit the new scripts and notebook files to `expanded-data`.