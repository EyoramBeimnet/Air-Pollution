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