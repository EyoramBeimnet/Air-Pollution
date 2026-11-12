"""Overfitting check for the monthly CNN. Run from the project root:
    python scripts/diagnose_cnn_overfit.py

Trains CNNs of different sizes on the same rows/splits as train_monthly_cnn.py,
and prints parameter count, best epoch, and train/validation/test RMSE next to
simple baselines. Big train-vs-test gaps = overfitting. If a small model matches
the big one on test, use the small one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
import train_monthly_cnn as T

DATA = Path("datasets/processed/monthly/monthly_aqi_targets.csv")
SEEDS = 3
CONFIGS = [(16, 12), (8, 12), (4, 12), (8, 6), (4, 3)]  # (filters, window)
SPLITS = ("train", "validation", "test")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def prepare(window):
    """Same rows, split rule and train-only scaling as train_monthly_cnn.py."""
    panels, wcols = T.build_panels(T.load_targets(DATA), None, window)
    samples = T.list_samples(panels)
    table = T.tabular_frame(panels, samples, wcols)
    months = sorted(table.target_month.unique())
    m_train, m_val, m_test = T.split_months(months, 0.70, 0.15)
    lookup = {**{m: "train" for m in m_train}, **{m: "validation" for m in m_val},
              **{m: "test" for m in m_test}}
    table["split"] = table.target_month.map(lookup)

    train_end = pd.Timestamp(m_train[-1])
    train_aqi = np.concatenate([p["aqi"][p["index"] <= train_end] for p in panels.values()])
    mu, sd = np.nanmean(train_aqi), np.nanstd(train_aqi)

    data = {}
    for s in SPLITS:
        chosen = [x for x, sp in zip(samples, table.split) if sp == s]
        data[s] = T.make_tensors(panels, chosen, wcols, window, (mu, sd, [], []), sorted(panels))
    return table, data, mu, sd


def run_cnn(data, mu, sd, window, filters):
    

    n_ch, n_static = data["train"][0].shape[2], data["train"][1].shape[1]
    preds = {s: [] for s in SPLITS}
    best_epochs = []
    for seed in range(SEEDS):
        keras.utils.set_random_seed(42 + seed)
        model = T.build_cnn(window, n_ch, n_static, filters, 0.2, 1e-3, True)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                             restore_best_weights=True)
        hist = model.fit(list(data["train"][:3]), data["train"][3],
                         validation_data=(list(data["validation"][:3]), data["validation"][3]),
                         epochs=300, batch_size=32, callbacks=[stop], verbose=0)
        best_epochs.append(int(np.argmin(hist.history["val_loss"])) + 1)
        for s in SPLITS:
            preds[s].append(model.predict(list(data[s][:3]), verbose=0).ravel())
        params = model.count_params()
        keras.backend.clear_session()
    out = {s: np.mean(preds[s], axis=0) * sd + mu for s in SPLITS}  # back to AQI units
    return out, params, best_epochs


def main():
    rows = []
    table, data, mu, sd = prepare(window=12)
    actual = {s: data[s][3] * sd + mu for s in SPLITS}

    # baselines (same rows as the CNN)
    frames = {s: table[table.split == s] for s in SPLITS}
    rows.append({"model": "Persistence", "params": 0, "best_epoch": "-",
                 **{s: rmse(actual[s], frames[s].aqi_lag_1) for s in SPLITS}})

    def ridge_x(f):
        m = f.target_month_number
        return pd.DataFrame({"lag1": f.aqi_lag_1, "lag12": f.aqi_lag_12.fillna(f.aqi_lag_1),
                             "sin": np.sin(2 * np.pi * m / 12), "cos": np.cos(2 * np.pi * m / 12)})
    ridge = make_pipeline(SimpleImputer(strategy="median"), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    ridge.fit(ridge_x(frames["train"]), frames["train"].aqi_median_daily)
    rows.append({"model": "Ridge (lag1, lag12, month)", "params": 5, "best_epoch": "-",
                 **{s: rmse(actual[s], ridge.predict(ridge_x(frames[s]))) for s in SPLITS}})

    # CNNs of decreasing size
    for filters, window in CONFIGS:
        table, data, mu, sd = prepare(window)
        pred, params, epochs = run_cnn(data, mu, sd, window, filters)
        rows.append({"model": f"CNN filters={filters} window={window}", "params": params,
                     "best_epoch": int(np.mean(epochs)),
                     **{s: rmse(actual[s], pred[s]) for s in SPLITS}})
        print("done:", rows[-1]["model"], flush=True)

    res = pd.DataFrame(rows).rename(columns={"train": "train_RMSE",
                                             "validation": "val_RMSE", "test": "test_RMSE"})
    res["test_minus_train"] = res.test_RMSE - res.train_RMSE
    print(f"\nTrain rows: {len(actual['train'])}  (RMSE in AQI points, {SEEDS}-seed average)")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out_path = Path(__file__).parent.parent / "notebooks" / "cnn_overfit_check.csv"
    res.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()