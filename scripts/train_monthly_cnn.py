#!/usr/bin/env python3
"""
python -m pip install tensorflow-cpu xgboost scikit-learn pandas

--input datasets/processed/monthly/model_results_cnn/predictions.csv \
--output-dir datasets/processed/monthly/model_results_cnn

Design:
  * Input to the CNN is the previous WINDOW calendar months of AQI (plus a
    "was observed" mask, month-of-year sin/cos, and optional weather) for the
    borough. The target month itself is never in the window.
  * Months that failed the coverage filter are gaps. They are forward-filled
    from earlier months only and flagged by the mask channel (no future data).
  * Scalers use statistics from months up to the end of the TRAIN split only.
  * The CNN predicts the change from last month's AQI (residual head), which
    starts it at the persistence baseline and lets it learn only the correction.
  * Validation is used only for early stopping. Test is never touched during
    training. Predictions are averaged over several random seeds.
  * RF / XGBoost use the same hyper-parameters as train_monthly_aqi_models.py
    and the same feature set (aqi_lag_1, aqi_lag_12, month number, weather_*),
    except lags are calendar-correct (see note in the README / summary).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

COUNTIES = {"36005": "Bronx", "36047": "Kings", "36061": "New York",
            "36081": "Queens", "36085": "Richmond"}
# EPA AQI category upper bounds: Good<=50, Moderate<=100, USG<=150, Unhealthy<=200, ...
CATEGORY_EDGES = [51, 101, 151, 201, 301]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def aqi_category(values) -> np.ndarray:
    return np.digitize(np.round(np.asarray(values, dtype=float)), CATEGORY_EDGES)


def score(y, pred, tol: float) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred) if len(y) > 1 else np.nan,
        # "Accuracy" is not a regression metric, so two explicit definitions:
        f"within_{tol:g}_AQI_pct": float(np.mean(np.abs(y - pred) <= tol) * 100),
        "AQI_category_accuracy_pct": float(np.mean(aqi_category(y) == aqi_category(pred)) * 100),
    }


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def load_targets(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"county_fips": "string"})
    need = {"county_fips", "borough", "target_month", "aqi_median_daily"}
    if need - set(d.columns):
        raise ValueError(f"{path} missing columns: {sorted(need - set(d.columns))}")
    d["county_fips"] = d.county_fips.astype("string").str.zfill(5)
    d["target_month"] = pd.to_datetime(d.target_month).dt.to_period("M").dt.to_timestamp()
    d["aqi_median_daily"] = pd.to_numeric(d.aqi_median_daily, errors="coerce")
    d = d.dropna(subset=["aqi_median_daily"])
    if d.duplicated(["county_fips", "target_month"]).any():
        raise ValueError("Duplicate county/month rows.")
    if not d.county_fips.isin(COUNTIES).all():
        raise ValueError("Targets contain non-NYC county FIPS codes.")
    return d


def load_weather(path: Path | None) -> pd.DataFrame | None:
    """Weather rows describe their OWN month (same file you pass to
    prepare_monthly_model_data.py --weather, un-shifted)."""
    if path is None:
        return None
    w = pd.read_csv(path, dtype={"county_fips": "string"})
    cols = [c for c in w.columns if c.startswith("weather_")]
    if not cols or {"county_fips", "target_month"} - set(w.columns):
        raise ValueError("Weather needs county_fips, target_month and weather_* columns.")
    w["county_fips"] = w.county_fips.astype("string").str.zfill(5)
    w["target_month"] = pd.to_datetime(w.target_month).dt.to_period("M").dt.to_timestamp()
    if w.duplicated(["county_fips", "target_month"]).any():
        raise ValueError("Duplicate weather county/month rows.")
    for c in cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    return w[["county_fips", "target_month", *cols]]


def build_panels(targets, weather, window):
    """One calendar-complete series per borough, padded `window` months at the
    front so the first real months still have a (masked) window."""
    start = targets.target_month.min() - pd.DateOffset(months=window)
    index = pd.date_range(start, targets.target_month.max(), freq="MS")
    wcols = [] if weather is None else [c for c in weather.columns if c.startswith("weather_")]
    panels = {}
    for fips in sorted(targets.county_fips.unique()):
        sub = targets[targets.county_fips == fips].set_index("target_month")
        aqi = sub.aqi_median_daily.reindex(index).to_numpy(float)
        wx = np.full((len(index), len(wcols)), np.nan)
        if wcols:
            ws = weather[weather.county_fips == fips].set_index("target_month")
            wx = ws[wcols].reindex(index).to_numpy(float)
        panels[fips] = {"borough": sub.borough.iloc[0], "aqi": aqi, "wx": wx, "index": index}
    return panels, wcols


def list_samples(panels):
    """(fips, position) for every month whose target AND previous month were
    observed: the same filter the existing trainer applies via aqi_lag_1."""
    out = []
    for fips, p in panels.items():
        for t in range(1, len(p["aqi"])):
            if np.isfinite(p["aqi"][t]) and np.isfinite(p["aqi"][t - 1]):
                out.append((fips, t))
    return sorted(out, key=lambda s: (panels[s[0]]["index"][s[1]], s[0]))


def tabular_frame(panels, samples, wcols):
    rows = []
    for fips, t in samples:
        p = panels[fips]
        r = {"county_fips": fips, "borough": p["borough"], "target_month": p["index"][t],
             "aqi_median_daily": p["aqi"][t], "aqi_lag_1": p["aqi"][t - 1],
             "aqi_lag_12": p["aqi"][t - 12] if t >= 12 else np.nan,
             "target_month_number": p["index"][t].month}
        for k, c in enumerate(wcols):
            r[c] = p["wx"][t - 1, k]  # prior month's weather, as in prepare_monthly_model_data.py
        rows.append(r)
    return pd.DataFrame(rows)


def split_months(months, train_fraction, validation_fraction):
    """Identical rule to train_monthly_aqi_models.py."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split fractions.")
    if len(months) < 18:
        raise ValueError(f"Only {len(months)} target months; need at least 18.")
    i = max(1, int(len(months) * train_fraction))
    j = min(len(months) - 1, max(i + 1, int(len(months) * (train_fraction + validation_fraction))))
    return months[:i], months[i:j], months[j:]


def make_tensors(panels, samples, wcols, window, stats, fips_order):
    """Build CNN inputs. Everything is scaled with TRAIN-only statistics."""
    mu, sd, wmu, wsd = stats
    channel_cache = {}
    for fips, p in panels.items():
        observed = np.isfinite(p["aqi"])
        filled = pd.Series(p["aqi"]).ffill().to_numpy()  # past-only forward fill
        scaled = np.nan_to_num((filled - mu) / sd, nan=0.0)
        months = p["index"].month.to_numpy()
        cols = [scaled, observed.astype(float),
                np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12)]
        for k in range(len(wcols)):
            cols.append(np.nan_to_num((p["wx"][:, k] - wmu[k]) / wsd[k], nan=0.0))
        channel_cache[fips] = (np.stack(cols, axis=1).astype("float32"), scaled)
    seq, static, base, y = [], [], [], []
    for fips, t in samples:
        chans, scaled = channel_cache[fips]
        seq.append(chans[t - window:t])
        m = panels[fips]["index"][t].month
        static.append([*(float(fips == f) for f in fips_order),
                       np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
        base.append([scaled[t - 1]])
        y.append((panels[fips]["aqi"][t] - mu) / sd)
    return (np.asarray(seq, "float32"), np.asarray(static, "float32"),
            np.asarray(base, "float32"), np.asarray(y, "float32"))


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
def build_cnn(window, n_channels, n_static, filters, dropout, l2, residual):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    seq_in = keras.Input((window, n_channels), name="sequence")
    static_in = keras.Input((n_static,), name="static")
    base_in = keras.Input((1,), name="last_month_aqi")
    reg = regularizers.l2(l2)
    x = layers.Conv1D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(seq_in)
    x = layers.Conv1D(filters * 2, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    newest = layers.Flatten()(layers.Cropping1D((window - 1, 0))(x))  # features at the latest month
    pooled = layers.Concatenate()([layers.GlobalAveragePooling1D()(x),
                                   layers.GlobalMaxPooling1D()(x), newest])
    z = layers.Concatenate()([pooled, static_in])
    z = layers.Dense(32, activation="relu", kernel_regularizer=reg)(z)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(1, kernel_initializer="zeros" if residual else "glorot_uniform")(z)
    if residual:
        out = layers.Add()([out, base_in])
    return keras.Model([seq_in, static_in, base_in], out)


def train_cnn_ensemble(train, val, tests, args, n_channels, n_static):
    """Returns ({split: mean prediction in scaled units}, [epochs per seed])."""
    from tensorflow import keras

    preds = {name: [] for name in ("train", "validation", "test")}
    epochs_used = []
    for seed in range(args.seeds):
        keras.utils.set_random_seed(args.random_state + seed)
        model = build_cnn(args.window, n_channels, n_static, args.filters,
                          args.dropout, args.l2, not args.no_residual)
        model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss="mse")
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                             restore_best_weights=True)
        hist = model.fit([train[0], train[1], train[2]], train[3],
                         validation_data=([val[0], val[1], val[2]], val[3]),
                         epochs=args.epochs, batch_size=args.batch_size,
                         callbacks=[stop], verbose=0)
        epochs_used.append(len(hist.history["loss"]))
        for name, data in (("train", train), ("validation", val), ("test", tests)):
            preds[name].append(model.predict([data[0], data[1], data[2]], verbose=0).ravel())
        keras.backend.clear_session()
    return {k: np.mean(v, axis=0) for k, v in preds.items()}, epochs_used


# --------------------------------------------------------------------------
# Uncertainty: is the CNN difference bigger than noise?
# --------------------------------------------------------------------------
def month_bootstrap(frame, model_a, model_b, n_boot, seed):
    """Resample whole target MONTHS (the 5 boroughs in a month are correlated)
    and report the distribution of RMSE(a) - RMSE(b). Negative => a is better."""
    pa = frame[frame.model == model_a].set_index(["county_fips", "target_month"])
    pb = frame[frame.model == model_b].set_index(["county_fips", "target_month"])
    idx = pa.index.intersection(pb.index)
    if len(idx) == 0:
        return None
    ea = (pa.loc[idx, "predicted_aqi"] - pa.loc[idx, "actual_aqi"]).to_numpy() ** 2
    eb = (pb.loc[idx, "predicted_aqi"] - pb.loc[idx, "actual_aqi"]).to_numpy() ** 2
    months = idx.get_level_values("target_month")
    codes, uniq = pd.factorize(months)
    sa = np.bincount(codes, ea, len(uniq)); sb = np.bincount(codes, eb, len(uniq))
    cnt = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    diff = np.sqrt(sa[draws].sum(1) / cnt[draws].sum(1)) - np.sqrt(sb[draws].sum(1) / cnt[draws].sum(1))
    return {"model_a": model_a, "model_b": model_b, "n_months": len(uniq),
            "rmse_diff_median": float(np.median(diff)),
            "ci90_low": float(np.percentile(diff, 5)), "ci90_high": float(np.percentile(diff, 95)),
            "pct_resamples_a_better": float((diff < 0).mean() * 100)}


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("datasets/processed/monthly/monthly_aqi_targets.csv"))
    p.add_argument("--weather", type=Path, default=None, help="Optional CSV: county_fips, target_month, weather_* (un-shifted).")
    p.add_argument("--output-dir", type=Path, default=Path("datasets/processed/monthly/model_results_cnn"))
    p.add_argument("--train-fraction", type=float, default=.70)
    p.add_argument("--validation-fraction", type=float, default=.15)
    p.add_argument("--window", type=int, default=12, help="Months of history shown to the CNN.")
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seeds", type=int, default=5, help="CNNs averaged into one ensemble prediction.")
    p.add_argument("--no-residual", action="store_true", help="Predict AQI directly instead of change vs last month.")
    p.add_argument("--accuracy-tolerance", type=float, default=5.0, help="AQI points for the 'within +/-X' accuracy.")
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    if args.window < 3:
        raise ValueError("--window must be at least 3.")

    targets = load_targets(args.input)
    weather = load_weather(args.weather)
    panels, wcols = build_panels(targets, weather, args.window)
    samples = list_samples(panels)
    table = tabular_frame(panels, samples, wcols)
    months = sorted(table.target_month.unique())
    m_train, m_val, m_test = split_months(months, args.train_fraction, args.validation_fraction)
    split_of = {**{m: "train" for m in m_train}, **{m: "validation" for m in m_val}, **{m: "test" for m in m_test}}
    table["split"] = table.target_month.map(split_of)
    train_end = pd.Timestamp(m_train[-1])

    # ---- train-only scaling statistics (months up to the last training target)
    fips_order = sorted(panels)
    train_aqi = np.concatenate([p["aqi"][p["index"] <= train_end] for p in panels.values()])
    mu, sd = np.nanmean(train_aqi), np.nanstd(train_aqi) or 1.0
    wmu, wsd = [], []
    for k in range(len(wcols)):
        vals = np.concatenate([p["wx"][p["index"] <= train_end, k] for p in panels.values()])
        wmu.append(np.nanmean(vals) if np.isfinite(vals).any() else 0.0)
        wsd.append(np.nanstd(vals) or 1.0 if np.isfinite(vals).any() else 1.0)
    stats = (mu, sd, wmu, wsd)

    sample_split = table.split.to_numpy()
    def subset(name):
        chosen = [s for s, sp in zip(samples, sample_split) if sp == name]
        return make_tensors(panels, chosen, wcols, args.window, stats, fips_order)
    train_t, val_t, test_t = subset("train"), subset("validation"), subset("test")
    n_channels, n_static = train_t[0].shape[2], train_t[1].shape[1]

    print(f"Rows: train={len(train_t[3])} validation={len(val_t[3])} test={len(test_t[3])}")
    print(f"Splits by target month: train {m_train[0]:%Y-%m}..{m_train[-1]:%Y-%m}; "
          f"validation {m_val[0]:%Y-%m}..{m_val[-1]:%Y-%m}; test {m_test[0]:%Y-%m}..{m_test[-1]:%Y-%m}")
    print(f"CNN: window={args.window} months, {n_channels} channels/step, "
          f"{args.seeds}-seed ensemble, residual={not args.no_residual}\n")

    # ---- CNN
    cnn_scaled, epochs_used = train_cnn_ensemble(train_t, val_t, test_t, args, n_channels, n_static)
    print(f"CNN epochs used per seed (early stopping): {epochs_used}")

    # ---- classical models, same rows / same split / same hyper-parameters
    features = ["aqi_lag_1", "aqi_lag_12", "target_month_number", *wcols]
    features = [c for c in features if table[c].notna().any()]
    frames = {s: table[table.split == s].reset_index(drop=True) for s in ("train", "validation", "test")}
    classical = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.03, min_child_weight=3,
                                subsample=.8, colsample_bytree=.8, reg_lambda=5,
                                objective="reg:squarederror", eval_metric="rmse",
                                random_state=42, n_jobs=-1),
    }
    all_preds = {"Persistence baseline": {s: frames[s].aqi_lag_1.to_numpy(float) for s in ("validation", "test")}}
    for name, est in classical.items():
        pipe = make_pipeline(SimpleImputer(strategy="median"), est)
        pipe.fit(frames["train"][features], frames["train"].aqi_median_daily)
        all_preds[name] = {s: pipe.predict(frames[s][features]) for s in frames}
    cnn_name = f"1D CNN ({args.seeds}-seed ensemble)"
    all_preds[cnn_name] = {s: cnn_scaled[s] * sd + mu for s in frames}

    # ---- assemble outputs (schema is compatible with evaluate_monthly_aqi_predictions.py)
    pred_rows, metric_rows = [], []
    for model, by_split in all_preds.items():
        for split, pred in by_split.items():
            fr = frames[split]
            metric_rows.append({"model": model, "split": split, "n_rows": len(fr),
                                **score(fr.aqi_median_daily, pred, args.accuracy_tolerance)})
            pred_rows.append(pd.DataFrame({
                "model": model, "split": split, "county_fips": fr.county_fips, "borough": fr.borough,
                "target_month": fr.target_month, "actual_aqi": fr.aqi_median_daily,
                "predicted_aqi": pred, "training_rows": len(frames["train"])}))
    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    boots = []
    held_out = predictions[predictions.split.isin(["validation", "test"])]
    for split in ("validation", "test"):
        sub = held_out[held_out.split == split]
        for other in [m for m in all_preds if m != cnn_name]:
            r = month_bootstrap(sub, cnn_name, other, args.bootstraps, args.random_state)
            if r:
                boots.append({"split": split, **r})
    boot = pd.DataFrame(boots)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "training_metrics.csv", index=False)
    boot.to_csv(args.output_dir / "cnn_vs_baselines_bootstrap.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.output_dir / "classical_features_used.csv", index=False)

    show = metrics[metrics.split.isin(["validation", "test"])].sort_values(["split", "RMSE"])
    pd.set_option("display.width", 200)
    print("\nHeld-out results (sorted by RMSE; lower RMSE/MAE and higher R2/accuracy are better):")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nCNN vs each model: RMSE(CNN) - RMSE(other), month-block bootstrap. "
          "Negative = CNN better; a CI that straddles 0 means NOT distinguishable from noise.")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
