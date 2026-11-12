"""Rolling-origin backtest: does the CNN beat ridge and XGBoost in every test year, or just one?
Run from the project root:
    python scripts/rolling_origin_backtest.py

For each test year: validate on the year before it, train on everything earlier.
Reuses prepare-style splitting and run_cnn from diagnose_cnn_overfit.py.
Saves notebooks/cnn_rolling_backtest.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
import diagnose_cnn_overfit as D
import train_monthly_cnn as T

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]  # 2020 = COVID year, read it separately
FILTERS, WINDOW = 8, 12
XGB_FEATURES = ["aqi_lag_1", "aqi_lag_12", "target_month_number"]  # same as train_monthly_cnn.py


def prepare_fold(test_year):
    """Train: before (test_year-1). Validation: test_year-1. Test: test_year."""
    panels, wcols = T.build_panels(T.load_targets(D.DATA), None, WINDOW)
    samples = T.list_samples(panels)
    table = T.tabular_frame(panels, samples, wcols)
    year = pd.to_datetime(table.target_month).dt.year
    table["split"] = np.select([year < test_year - 1, year == test_year - 1, year == test_year],
                               ["train", "validation", "test"], default="unused")

    val_start = pd.Timestamp(test_year - 1, 1, 1)
    train_aqi = np.concatenate([p["aqi"][p["index"] < val_start] for p in panels.values()])
    mu, sd = np.nanmean(train_aqi), np.nanstd(train_aqi)

    data = {}
    for s in D.SPLITS:
        chosen = [x for x, sp in zip(samples, table.split) if sp == s]
        data[s] = T.make_tensors(panels, chosen, wcols, WINDOW, (mu, sd, [], []), sorted(panels))
    return table, data, mu, sd


def ridge_x(f):
    m = f.target_month_number
    return pd.DataFrame({"lag1": f.aqi_lag_1, "lag12": f.aqi_lag_12.fillna(f.aqi_lag_1),
                         "sin": np.sin(2 * np.pi * m / 12), "cos": np.cos(2 * np.pi * m / 12)})


def fit_xgb(train, test):
    """Same settings as train_monthly_cnn.py; averaged over D.SEEDS seeds like the CNN."""
    preds = []
    for seed in range(D.SEEDS):
        model = make_pipeline(SimpleImputer(strategy="median"), XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=42 + seed, n_jobs=-1))
        model.fit(train[XGB_FEATURES], train.aqi_median_daily)
        preds.append(model.predict(test[XGB_FEATURES]))
    return np.mean(preds, axis=0)


def main():
    rows = []
    for year in TEST_YEARS:
        table, data, mu, sd = prepare_fold(year)
        train, test = table[table.split == "train"], table[table.split == "test"]
        actual = data["test"][3] * sd + mu

        ridge = make_pipeline(SimpleImputer(strategy="median"), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        ridge.fit(ridge_x(train), train.aqi_median_daily)
        pred, _, _ = D.run_cnn(data, mu, sd, WINDOW, FILTERS)

        rows.append({"test_year": year, "train_rows": len(train), "test_rows": len(test),
                     "persistence": D.rmse(actual, test.aqi_lag_1),
                     "ridge": D.rmse(actual, ridge.predict(ridge_x(test))),
                     "xgboost": D.rmse(actual, fit_xgb(train, test)),
                     "cnn": D.rmse(actual, pred["test"])})
        print("done:", year, flush=True)

    res = pd.DataFrame(rows)
    res["cnn_minus_ridge"] = res.cnn - res.ridge  # negative = CNN better
    res["cnn_minus_xgb"] = res.cnn - res.xgboost
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nCNN beats ridge in {(res.cnn_minus_ridge < 0).sum()} of {len(res)} years; "
          f"mean gap {res.cnn_minus_ridge.mean():.2f}, std {res.cnn_minus_ridge.std():.2f}")
    print(f"CNN beats XGBoost in {(res.cnn_minus_xgb < 0).sum()} of {len(res)} years; "
          f"mean gap {res.cnn_minus_xgb.mean():.2f}, std {res.cnn_minus_xgb.std():.2f}")

    out_path = Path(__file__).parent.parent / "notebooks" / "cnn_rolling_backtest.csv"
    res.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()