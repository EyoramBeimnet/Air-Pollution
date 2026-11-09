# Monthly Next-Month AQI — 1D CNN Experiment Findings

**Date:** 2026-11-08  
**Repo branch:** `expanded-data`  
**Script:** `scripts/train_monthly_cnn.py`  
**Outputs:** `datasets/processed/monthly/model_results_cnn/`

This experiment compares a 1D convolutional neural network (CNN) against the existing monthly baselines (Persistence, Random Forest, XGBoost) for **next-month median AQI** at the NYC borough level.

---

## Setup (fair comparison)

- **Same rows:** Only borough-months where both the target month and the previous month have observed AQI (same filter as the original monthly trainer).
- **Same chronological splits:** ~70% train / 15% validation / 15% test by target month (identical rule to `train_monthly_aqi_models.py`).
- **No leakage:** CNN input is the previous 12 calendar months only; the target month is never in the window. Gaps are past-only forward-filled and flagged with an observed-mask channel. Scalers use train-period statistics only.
- **Residual head:** CNN predicts the change from last month’s AQI (starts at persistence, learns corrections).
- **Ensemble:** Predictions averaged over 5 random seeds; validation used only for early stopping.
- **Classical models:** Re-trained on the identical rows/splits with the same hyperparameters as the original monthly script.

This run did **not** modify any original GitHub files. All new outputs live under `model_results_cnn/`.

---

## Held-out test results

| Model | n_rows | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|--------|-------|--------|------|
| **1D CNN (5-seed ensemble)** | 110 | **3.87** | **4.70** | **0.61** |
| XGBoost | 110 | 4.14 | 5.09 | 0.54 |
| Random Forest | 110 | 4.27 | 5.33 | 0.50 |
| Persistence (last month) | 110 | 5.85 | 7.71 | −0.05 |

**Validation** showed the same ranking: CNN best, then XGBoost, Random Forest, Persistence.

Relative to persistence on test:

- RMSE reduced by about **39%**
- MAE reduced by about **34%**
- R² moved from slightly negative to **0.61**

---

## Bootstrap check (is the gap real?)

Month-block bootstrap of **RMSE(CNN) − RMSE(other)** (negative = CNN better):

| Split | CNN vs | Median Δ RMSE | 90% CI | % resamples CNN better |
|-------|--------|---------------|--------|-------------------------|
| **test** | Persistence | **−2.92** | −4.70 to −1.15 | **99.9%** |
| **test** | Random Forest | **−0.63** | −1.01 to −0.21 | **99.4%** |
| **test** | XGBoost | **−0.39** | −0.74 to −0.00 | **95.1%** |
| validation | Random Forest | −0.73 | −1.35 to −0.04 | 95.6% |
| validation | XGBoost | −0.41 | −0.94 to +0.17 | 87.9% |

On the **test** set, the CNN advantage over Persistence and Random Forest is clearly supported (entire CI negative). Over XGBoost the edge is smaller but still favors the CNN in ~95% of resamples, with the upper CI near zero.

---

## Interpretation

1. **CNN is the strongest of the four models** on this monthly next-month borough AQI task under the controlled comparison above.
2. Beating persistence by a large margin shows the model is using more than simple month-to-month carry-forward.
3. Beating RF and XGBoost on the same features/splits indicates the sequential 12-month window + residual design adds useful signal.
4. Sample size remains limited (5 boroughs × months). Metrics should be treated as experimental evidence, not proof of operational forecast skill.
5. Point-in-time availability of any optional weather inputs was not re-audited in this run (weather was optional; core comparison is AQI history + calendar).

---

## Files produced

| File | Contents |
|------|----------|
| `training_metrics.csv` | MAE, RMSE, R², within-±5 AQI %, category accuracy by model & split |
| `predictions.csv` | Row-level actual vs predicted (compatible with `evaluate_monthly_aqi_predictions.py`) |
| `cnn_vs_baselines_bootstrap.csv` | Month-block bootstrap of RMSE differences |
| `evaluation_summary.csv` | Evaluator overall metrics |
| `evaluation_by_borough.csv` | Borough-level metrics |
| `evaluation_by_month.csv` | Month-level metrics |

---

## How this was run

```bat
python scripts\download_daily_aqi.py --start-year 2014 --end-year 2025
python scripts\prepare_monthly_aqi.py
python scripts\train_monthly_cnn.py
python scripts\evaluate_monthly_aqi_predictions.py ^
  --input datasets\processed\monthly\model_results_cnn\predictions.csv ^
  --output-dir datasets\processed\monthly\model_results_cnn
