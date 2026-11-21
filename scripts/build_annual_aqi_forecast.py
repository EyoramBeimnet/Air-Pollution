"""Build the one-year-ahead 2026 NYC borough AQI forecast artifact.

This script deliberately does not change the project's training experiments.  It
reuses the project's weather-enhanced annual XGBoost configuration, refits it on the
historically labelled 2014--2024 feature years, and applies it once to the
observed 2025 feature rows.  No values from 2026 are used as inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from train_aqi_models import (
    ABLATION_FEATURES,
    FIPS_COLUMN,
    FORECAST_YEAR_COLUMN,
    STAGING_FILE,
    TARGET_COLUMN,
    YEAR_COLUMN,
    build_models,
    load_staging_data,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_FILE = ROOT / "datasets" / "processed" / "master_borough_year_2014_2025.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "processed" / "model_results" / "annual_aqi_forecast_2026.csv"
DEFAULT_METADATA = ROOT / "datasets" / "processed" / "model_results" / "annual_aqi_forecast_2026_metadata.json"
FORECAST_YEAR = 2026
FEATURE_YEAR = 2025
EXPECTED_FIPS = {"36005", "36047", "36061", "36081", "36085"}

# This is the existing "Ablation weather" annual experiment: prior AQI,
# CAMPD NOx, temperature, and precipitation. It is the documented
# weather-enhanced XGBoost configuration selected for this integration.
FEATURES = ABLATION_FEATURES["Ablation weather"]


def normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)


def training_frame() -> pd.DataFrame:
    """Return only historically labelled rows usable to predict 2026."""
    data = load_staging_data(STAGING_FILE)
    data = data[data[YEAR_COLUMN].between(2014, 2024)].copy()
    if set(data[FIPS_COLUMN]) != EXPECTED_FIPS or len(data) != 55:
        raise ValueError("Expected 55 labelled 2014-2024 borough-year training rows.")
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    if not np.isfinite(data[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Historical AQI training targets must be finite.")
    return data.sort_values([YEAR_COLUMN, FIPS_COLUMN]).reset_index(drop=True)


def feature_frame() -> pd.DataFrame:
    """Create the 2025 feature rows using only observed 2025 inputs."""
    master = pd.read_csv(MASTER_FILE, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "borough", "year", "aqi_median", "campd_total_nox_mass_tons",
                "weather_temp_mean_c", "weather_precip_total_mm"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master data is missing required 2025 forecast inputs: {sorted(missing)}")
    frame = master[pd.to_numeric(master["year"], errors="coerce").eq(FEATURE_YEAR)].copy()
    frame[FIPS_COLUMN] = normalize_fips(frame["county_fips"])
    frame["last_observed_median_aqi"] = pd.to_numeric(frame["aqi_median"], errors="coerce")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if set(frame[FIPS_COLUMN]) != EXPECTED_FIPS or len(frame) != 5:
        raise ValueError("Expected one 2025 feature row for each NYC borough.")
    if frame[["last_observed_median_aqi", *FEATURES[1:]]].isna().any().any():
        raise ValueError("2025 forecast inputs are incomplete; refusing to impute unavailable future inputs.")
    return frame.sort_values(FIPS_COLUMN).reset_index(drop=True)


def build_forecast() -> tuple[pd.DataFrame, dict[str, object]]:
    train = training_frame()
    future = feature_frame()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_future = imputer.transform(future[FEATURES])
    model = build_models()["XGBoost"]
    model.fit(x_train, train[TARGET_COLUMN].to_numpy(dtype=float))
    predicted = model.predict(x_future)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Model produced invalid 2026 AQI predictions.")
    forecast = pd.DataFrame({
        "fips": future[FIPS_COLUMN],
        "borough": future["borough"].astype(str),
        "year": FORECAST_YEAR,
        "aqi": np.round(predicted, 3),
        "forecast": True,
    })
    metadata: dict[str, object] = {
        "purpose": "One-year-ahead 2026 NYC borough annual median daily AQI point forecast",
        "metric": "annual_median_daily_aqi",
        "target_column": TARGET_COLUMN,
        "model": "XGBoost from scripts/train_aqi_models.py (weather-enhanced annual configuration)",
        "features": FEATURES,
        "training_feature_years": [int(train[YEAR_COLUMN].min()), int(train[YEAR_COLUMN].max())],
        "training_target_years": [int(train[FORECAST_YEAR_COLUMN].min()), int(train[FORECAST_YEAR_COLUMN].max())],
        "training_rows": len(train),
        "forecast_feature_year": FEATURE_YEAR,
        "forecast_year": FORECAST_YEAR,
        "preprocessing": "Median imputation fit only on the historical training rows.",
        "inputs": "Observed 2025 AQI, CAMPD NOx, temperature, and precipitation.",
        "limitations": [
            "Point forecast only; no confidence interval is estimated or included.",
            "Small borough-year training sample; this is a research forecast, not an operational forecast.",
            "No forecasts beyond 2026 are generated.",
        ],
    }
    return forecast, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    forecast, metadata = build_forecast()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(forecast)} 2026 forecast rows to {args.output}")
    print(f"Wrote forecast metadata to {args.metadata}")


if __name__ == "__main__":
    main()
