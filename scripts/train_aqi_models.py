
# train_aqi_models.py

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "datasets" / "processed" / "handoff"
PROCESSED = ROOT / "datasets" / "processed"
OUTPUT = HANDOFF / "model_results"

OUTPUT.mkdir(parents=True, exist_ok=True)

PRIMARY_FILE = HANDOFF / "annual_next_year_RESEARCH_CANDIDATE.csv"
REDUCED_FILE = HANDOFF / "annual_next_year_REDUCED_CANDIDATE.csv"

SPLIT_FILES = {
    "train": HANDOFF / "research_train_CANDIDATE.csv",
    "validation": HANDOFF / "research_validation_CANDIDATE.csv",
    "test": HANDOFF / "research_test_CANDIDATE.csv",
}

PRIMARY_FEATURES = [
    "last_observed_median_aqi",
    "recorded_tree_plantings",
    "drive_alone_pct",
    "campd_total_nox_mass_tons",
    "weather_temp_mean_c",
    "weather_precip_total_mm",
]

REDUCED_FEATURES = [
    "last_observed_median_aqi",
    "recorded_tree_plantings",
    "weather_temp_mean_c",
    "weather_precip_total_mm",
]

# The script accepts common target column naming conventions.
TARGET_ALIASES = [
    "target_next_year_median_aqi",
    "target_median_aqi",
    "next_year_median_aqi",
    "target_aqi",
    "next_year_aqi",
    "aqi_target",
    "target",
]

YEAR_ALIASES = [
    "feature_year",
    "year",
    "Year",
]

# A fixed seed makes the exploratory runs reproducible.
RANDOM_STATE = 42


# ---------------------------------------------------------
# Dataset loading and schema checks
# ---------------------------------------------------------

def find_column(df, aliases, description):
    """Find a column using known aliases, ignoring capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {aliases}"
    )


def load_candidate(path):
    """Load a prepared candidate without merging duplicate source labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    target_col = find_column(df, TARGET_ALIASES, "next-year AQI target")
    year_col = find_column(df, YEAR_ALIASES, "feature year")

    df = df.copy()
    df["_target"] = pd.to_numeric(df[target_col], errors="coerce")
    df["_feature_year"] = pd.to_numeric(df[year_col], errors="coerce")

    if df["_feature_year"].isna().any():
        raise ValueError(f"Missing or invalid feature years in {path.name}")

    if df["_target"].isna().any():
        warnings.warn(
            f"Dropping {df['_target'].isna().sum()} rows with missing targets "
            f"from {path.name}."
        )
        df = df.dropna(subset=["_target"])

    # Keep one row per borough and feature year.
    fips_col = next(
        (
            col for col in df.columns
            if col.lower() in {"county_fips", "fips", "county_fips_code"}
        ),
        None,
    )

    if fips_col is None:
        raise ValueError(
            f"Cannot identify borough/FIPS key in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    if df.duplicated([fips_col, "_feature_year"]).any():
        raise ValueError(
            f"Duplicate borough-year rows found in {path.name}. "
            "Resolve duplicates before modeling."
        )

    return df, fips_col


def load_splits():
    """Use the supplied chronological split files when all are available."""
    if all(path.exists() for path in SPLIT_FILES.values()):
        frames = {}

        for split_name, path in SPLIT_FILES.items():
            frames[split_name], _ = load_candidate(path)

        print("Using the existing chronological split files.")
        return frames

    print(
        "Split files are incomplete. Building the documented chronological "
        "split from the primary candidate."
    )

    df, _ = load_candidate(PRIMARY_FILE)

    # Feature years 2020-2022 -> targets 2021-2023.
    # Feature year 2023 -> target 2024.
    # Feature year 2024 -> target 2025.
    return {
        "train": df[df["_feature_year"].between(2020, 2022)].copy(),
        "validation": df[df["_feature_year"] == 2023].copy(),
        "test": df[df["_feature_year"] == 2024].copy(),
    }


# ---------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": rmse,
        "R2": (
            r2_score(y_true, predictions)
            if len(y_true) >= 2
            else np.nan
        ),
    }


def run_experiment(split_frames, feature_names, experiment_name):
    """Fit preprocessing and models on training rows only."""

    for split_name, df in split_frames.items():
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"{split_name} data is missing columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    train = split_frames["train"].copy()
    validation = split_frames["validation"].copy()
    test = split_frames["test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Train, validation, and test must all contain observations. "
            "Check feature years and the candidate files."
        )

    # Force predictors to numeric; nonnumeric values become missing.
    for df in (train, validation, test):
        for feature in feature_names:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

    # All learned preprocessing is fitted on training data only.
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(train[feature_names])
    X_validation = imputer.transform(validation[feature_names])
    X_test = imputer.transform(test[feature_names])

    y_train = train["_target"].to_numpy(dtype=float)
    y_validation = validation["_target"].to_numpy(dtype=float)
    y_test = test["_target"].to_numpy(dtype=float)

    if not np.isfinite(y_train).all():
        raise ValueError("Training targets contain invalid values.")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    predictions_to_save = []

    # No-change baseline: next year's AQI equals the latest observed AQI.
    baseline_feature = "last_observed_median_aqi"

    if baseline_feature not in feature_names:
        raise ValueError(
            f"The no-change baseline requires {baseline_feature}."
        )

    for split_name, X, y, df in [
        ("validation", X_validation, y_validation, validation),
        ("test", X_test, y_test, test),
    ]:
        baseline_predictions = pd.to_numeric(
            df[baseline_feature], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(baseline_predictions).all():
            raise ValueError(
                f"Missing baseline AQI in the {split_name} split."
            )

        baseline_metrics = calculate_metrics(y, baseline_predictions)

        results.append({
            "experiment": experiment_name,
            "model": "No-change baseline",
            "split": split_name,
            "n_rows": len(y),
            **baseline_metrics,
        })

        for i, (_, row) in enumerate(df.iterrows()):
            predictions_to_save.append({
                "experiment": experiment_name,
                "model": "No-change baseline",
                "split": split_name,
                "feature_year": int(row["_feature_year"]),
                "target_year": int(row["_feature_year"]) + 1,
                "actual_aqi": y[i],
                "predicted_aqi": baseline_predictions[i],
            })

    for model_name, model in models.items():
        # Do not fit on validation or test observations.
        model.fit(X_train, y_train)

        for split_name, X, y, df in [
            ("train", X_train, y_train, train),
            ("validation", X_validation, y_validation, validation),
            ("test", X_test, y_test, test),
        ]:
            predictions = model.predict(X)

            results.append({
                "experiment": experiment_name,
                "model": model_name,
                "split": split_name,
                "n_rows": len(y),
                **calculate_metrics(y, predictions),
            })

            for i, (_, row) in enumerate(df.iterrows()):
                predictions_to_save.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "split": split_name,
                    "feature_year": int(row["_feature_year"]),
                    "target_year": int(row["_feature_year"]) + 1,
                    "actual_aqi": y[i],
                    "predicted_aqi": predictions[i],
                })

        # Save model feature importances for exploratory interpretation.
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance.to_csv(
            OUTPUT / (
                f"{experiment_name.lower().replace(' ', '_')}_"
                f"{model_name.lower().replace(' ', '_')}_"
                "feature_importance.csv"
            ),
            index=False,
        )

    return results, predictions_to_save


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    split_frames = load_splits()

    print("\nSplit sizes:")
    for name, df in split_frames.items():
        years = sorted(df["_feature_year"].unique().tolist())
        print(f"  {name}: {len(df)} rows, feature years {years}")

    # The reduced candidate is a separate prepared view.
    # Verify that its rows and target labels agree with the primary data.
    primary, primary_fips = load_candidate(PRIMARY_FILE)
    reduced, reduced_fips = load_candidate(REDUCED_FILE)

    # Validate reduced rows against primary by borough and feature year.
    primary_key = [primary_fips, "_feature_year"]
    reduced_key = [reduced_fips, "_feature_year"]

    primary_check = primary[
        primary_key + ["_target"]
    ].rename(columns={
        primary_fips: "_fips",
        "_target": "_primary_target",
    })

    reduced_check = reduced[
        reduced_key + ["_target"]
    ].rename(columns={
        reduced_fips: "_fips",
        "_target": "_reduced_target",
    })

    comparison = primary_check.merge(
        reduced_check,
        on=["_fips", "_feature_year"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not (comparison["_merge"] == "both").all():
        raise ValueError(
            "Primary and reduced candidates do not contain identical "
            "borough-year keys. Inspect the files before comparing models."
        )

    if not np.allclose(
        comparison["_primary_target"],
        comparison["_reduced_target"],
        equal_nan=True,
    ):
        raise ValueError(
            "Primary and reduced candidate targets disagree. "
            "Do not compare them until this is resolved."
        )

    all_results = []
    all_predictions = []

    for experiment_name, features in [
        ("Six-feature", PRIMARY_FEATURES),
        ("Reduced four-feature", REDUCED_FEATURES),
    ]:
        results, predictions = run_experiment(
            split_frames=split_frames,
            feature_names=features,
            experiment_name=experiment_name,
        )

        all_results.extend(results)
        all_predictions.extend(predictions)

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)

    results_path = OUTPUT / "model_comparison.csv"
    predictions_path = OUTPUT / "predictions.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    # Save a simple run manifest for reproducibility.
    manifest = {
        "purpose": "Exploratory annual next-year AQI prediction",
        "random_state": RANDOM_STATE,
        "features": {
            "six_feature": PRIMARY_FEATURES,
            "reduced_four_feature": REDUCED_FEATURES,
        },
        "split_sizes": {
            name: len(df) for name, df in split_frames.items()
        },
        "metrics": ["MAE", "RMSE", "R2"],
        "preprocessing": (
            "Median imputation fitted on training rows only."
        ),
        "limitations": [
            "Small, correlated borough-year sample.",
            "Point-in-time feature availability has not been certified.",
            "Results do not establish operational forecasting skill.",
            "Hyperparameters are exploratory, not tuned on the test set.",
        ],
    }

    with open(OUTPUT / "run_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nValidation and test results:")
    display_columns = [
        "experiment", "model", "split", "n_rows", "MAE", "RMSE", "R2"
    ]

    print(
        results_df[
            results_df["split"].isin(["validation", "test"])
        ][display_columns].round(3).to_string(index=False)
    )

    print(f"\nSaved results to: {results_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved run manifest to: {OUTPUT / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
