# Monthly AQI CNN: Overfitting Check and Rolling-Origin Backtest

Branch reviewed: `expanded-data` (`scripts/train_monthly_cnn.py`, `notebooks/cnnFindings.md`).
Scripts added for this analysis: `scripts/diagnose_cnn_overfit.py`, `scripts/rolling_origin_backtest.py`.
All errors are RMSE in AQI points. Lower is better.

## 1. Question

A teammate's CNN improved on the earlier monthly models. How likely is it that the improvement comes from overfitting the training data, and is the model more complex than it needs to be?

## 2. Summary

- The CNN is **not badly overfit**, but it is **more complex than needed**. A CNN with about a quarter of the parameters scores the same on test.
- The CNN's gain over simple baselines is real but modest: about **0.35 to 0.45 RMSE better than ridge**.
- Against **XGBoost the CNN is effectively tied**. Across six test years the mean gap is about zero, and the year-to-year swings are much larger than that.
- **Recommendation:** use XGBoost as the main model (simpler, faster, steadier). Keep the CNN as a documented alternative if the project needs a deep-learning component.

## 3. Why overfitting was a concern

- **Capacity vs. data.** The full CNN has about 5,100 parameters and trains on 498 rows (roughly 10 parameters per row). Those rows are also correlated: consecutive months within a borough, and the five boroughs within a month, move together. The effective sample size is smaller than 498.
- **The reported edge over XGBoost was thin.** In `cnnFindings.md` the test bootstrap difference vs. XGBoost had a 90% CI of about -0.74 to -0.00, and on validation the CI included zero.
- **The original comparison was not equal.** The CNN was a 5-seed ensemble while Random Forest and XGBoost were single fits, and the CNN saw a 12-month history while the trees saw only lag 1, lag 12, and month number.
- **Persistence is a weak baseline.** Its test R² is below zero, so almost any model that shrinks toward the mean beats it by a lot.
- **One test window.** The original test set was 110 rows from a single period (March 2024 to December 2025).

What the original script already did well: chronological splits, train-only scaling, validation used only for early stopping, a residual head, L2, dropout, and seed averaging.

## 4. Check 1: model size and window (single split)

`diagnose_cnn_overfit.py` trains CNNs of different sizes on the same rows and splits as the training script, with 3 seeds averaged, next to simple baselines. Train rows: 498.

Run A (first run):

| Model | Params | Best epoch | Train | Val | Test | Test - Train |
|---|---|---|---|---|---|---|
| Persistence | 0 | - | 8.09 | 8.17 | 7.71 | -0.37 |
| Ridge (lag1, lag12, month) | 5 | - | 5.79 | 6.06 | 5.22 | -0.57 |
| CNN filters=16, window=12 | 5137 | 31 | 3.92 | 5.16 | 4.77 | 0.86 |
| CNN filters=8, window=12 | 2329 | 76 | 3.72 | 5.12 | 4.81 | 1.10 |
| CNN filters=4, window=12 | 1213 | 69 | 3.97 | 5.14 | 4.79 | 0.82 |
| CNN filters=8, window=6 | 2329 | 39 | 4.06 | 5.41 | 5.43 | 1.37 |
| CNN filters=4, window=3 | 1213 | 83 | 4.06 | 5.38 | 5.22 | 1.16 |

Run B (second run on a different machine):

| Model | Params | Best epoch | Train | Val | Test | Test - Train |
|---|---|---|---|---|---|---|
| Persistence | 0 | - | 8.09 | 8.17 | 7.71 | -0.37 |
| Ridge (lag1, lag12, month) | 5 | - | 5.79 | 6.06 | 5.22 | -0.57 |
| CNN filters=16, window=12 | 5137 | 23 | 4.17 | 5.32 | 4.85 | 0.67 |
| CNN filters=8, window=12 | 2329 | 44 | 3.97 | 5.09 | 4.75 | 0.78 |
| CNN filters=4, window=12 | 1213 | 84 | 3.86 | 5.18 | 4.89 | 1.04 |
| CNN filters=8, window=6 | 2329 | 34 | 4.15 | 5.44 | 5.53 | 1.38 |
| CNN filters=4, window=3 | 1213 | 66 | 4.13 | 5.50 | 5.18 | 1.04 |

Findings:

1. **Size does not matter.** The 1,213-parameter CNN matches the 5,137-parameter one on test. The ranking among 4, 8, and 16 filters flipped between runs, so those differences are seed noise (roughly +/-0.1 to 0.15 RMSE). The smaller model is enough.
2. **Overfitting is mild.** The train-to-test gap for the 12-month models is about 0.7 to 1.1 RMSE (train about 4, test about 4.8). There is a gap but no collapse. Early stopping chose epochs from 23 to 84, so the models do learn beyond persistence.
3. **The 12-month window drives the gain.** Windows of 6 and 3 months score about the same as, or worse than, ridge (5.2 to 5.5). The 12-month models are about 0.35 to 0.45 better than ridge, which suggests the benefit comes from seeing a full seasonal cycle.
4. **The test period is easier than validation.** Persistence and ridge also score lower on test than on train and validation. Compare models against each other rather than reading absolute test numbers.

## 5. Check 2: rolling-origin backtest (six test years)

`rolling_origin_backtest.py` repeats the split for each test year from 2020 to 2025. For a given test year, the previous year is the validation set (early stopping only) and everything earlier is training data. The CNN is the 8-filter, window-12 model. XGBoost uses the same settings and features as `train_monthly_cnn.py` (lag 1, lag 12, month number; 200 trees, depth 3, learning rate 0.03). Both the CNN and XGBoost average 3 seeds.

Run A:

| Test year | Train rows | Persistence | Ridge | XGBoost | CNN | CNN - Ridge | CNN - XGB |
|---|---|---|---|---|---|---|---|
| 2020 | 293 | 7.99 | 5.87 | 5.49 | 4.78 | -1.08 | -0.71 |
| 2021 | 353 | 7.07 | 5.92 | 5.16 | 6.30 | +0.38 | +1.14 |
| 2022 | 413 | 6.39 | 4.45 | 4.51 | 4.37 | -0.07 | -0.14 |
| 2023 | 473 | 8.88 | 6.70 | 6.05 | 5.76 | -0.93 | -0.28 |
| 2024 | 533 | 7.28 | 5.02 | 4.65 | 4.81 | -0.22 | +0.16 |
| 2025 | 593 | 8.22 | 5.71 | 5.63 | 5.15 | -0.56 | -0.48 |

Run B:

| Test year | Train rows | Persistence | Ridge | XGBoost | CNN | CNN - Ridge | CNN - XGB |
|---|---|---|---|---|---|---|---|
| 2020 | 293 | 7.99 | 5.87 | 5.48 | 4.93 | -0.94 | -0.55 |
| 2021 | 353 | 7.07 | 5.92 | 5.19 | 6.45 | +0.52 | +1.26 |
| 2022 | 413 | 6.39 | 4.45 | 4.49 | 4.56 | +0.11 | +0.07 |
| 2023 | 473 | 8.88 | 6.70 | 6.05 | 5.74 | -0.96 | -0.31 |
| 2024 | 533 | 7.28 | 5.02 | 4.65 | 4.81 | -0.22 | +0.16 |
| 2025 | 593 | 8.22 | 5.71 | 5.62 | 5.10 | -0.61 | -0.52 |

Summary (negative gap = CNN better):

| | Run A | Run B |
|---|---|---|
| CNN beats ridge | 5 of 6 years, mean gap -0.42 | 4 of 6 years, mean gap -0.35 |
| CNN beats XGBoost | 4 of 6 years, mean gap -0.05 | 3 of 6 years, mean gap +0.02 |

Findings:

1. **The CNN's edge over ridge is real but small** (about 0.35 to 0.42 RMSE), consistent with the single-split result.
2. **XGBoost gets about the same improvement over ridge** (about 0.37 RMSE) from three simple features. Most of the gain comes from a nonlinear model using lag 1, lag 12, and month, not from the CNN's longer window.
3. **The CNN and XGBoost are tied.** The mean gap is about zero and the standard deviation (about 0.65) is far larger than the mean. The CNN's seed-to-seed wobble between runs (0.1 to 0.2 RMSE) is as large as its whole margin.
4. **The CNN is more volatile.** It has the largest wins (2020, 2025) and the largest loss (2021, about +1.1 to +1.3 vs. XGBoost) in both runs.
5. **2020 inflates the CNN's average advantage over ridge.** It is the COVID year and the largest CNN win. Excluding it, the mean gap vs. ridge in Run A is about -0.28.
6. **2021 is the CNN's consistent weak spot.** One possible explanation is that its validation year is 2020, an unusual year used for early stopping. This was not tested.

## 6. Recommendations

1. Use **XGBoost** as the primary monthly model.
2. If the CNN is kept, use the **small version** (`filters=4` or `8`, `window=12`).
3. Report results from the **rolling backtest** rather than a single split, and note that 2020 is an unusual year.
4. Freeze hyperparameters before looking at test results again, and record that in `cnnFindings.md`.

## 7. Limitations

- Six overlapping folds with 60 test rows each. Training sets overlap across folds, so the folds are not independent.
- Only 3 seeds per model. Differences smaller than about 0.15 RMSE are within noise.
- No hyperparameter tuning for either the CNN or XGBoost. A tuned model could shift the comparison, though the CNN-XGBoost gap would need to change a lot to be distinguishable from noise.
- The ridge baseline is basic (lag 1, lag 12, month sine and cosine).
- The original `model_results_cnn/` output files are not committed to the branch, so the first-run results in `cnnFindings.md` were not independently re-checked here beyond the runs above.

## 8. Reproducing

Run from the repo root with `tensorflow-cpu`, `xgboost`, and `scikit-learn` installed:

```
python scripts/diagnose_cnn_overfit.py
python scripts/rolling_origin_backtest.py
```

The backtest writes `notebooks/cnn_rolling_backtest.csv`. Both scripts need `train_monthly_cnn.py` in the same folder, and the backtest also imports `diagnose_cnn_overfit.py`. Expect CNN numbers to differ by a few hundredths to a couple of tenths between machines, but the same pattern of results.