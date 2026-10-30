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
PROCESSED = ROOT / "datasets" / "processed"
OUTPUT = PROCESSED / "model_results"

# This is the one-year-ahead modeling table created by
# scripts/prepare_model_data.py from the expanded 2014-2025 datasets.
STAGING_FILE = PROCESSED / "next_year_aqi_model_STAGING.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target_next_year_median_aqi"
YEAR_COLUMN = "feature_year"
FORECAST_YEAR_COLUMN = "forecast_year"
FIPS_COLUMN = "county_fips"
BASELINE_FEATURE = "last_observed_median_aqi"

# These features are intended for the full 2014-2024 feature-year range.
# The selected columns are expected to have historical coverage throughout
# the expanded dataset. Missing values are still checked and imputed safely.
HISTORICAL_FEATURES = [
    "last_observed_median_aqi",
    "campd_total_nox_mass_tons",
    "traffic_sample_mean_15min_vehicle_count",
]

# These are the original six project predictors. Some sources begin later:
# tree plantings from 2016 and weather/ACS commuting data from 2019.
# This experiment therefore uses 2019-2024, when these predictors have
# meaningful historical coverage, instead of filling years before collection.
EXTENDED_FEATURES = [
    "last_observed_median_aqi",
    "recorded_tree_plantings",
    "drive_alone_pct",
    "campd_total_nox_mass_tons",
    "weather_temp_mean_c",
    "weather_precip_total_mm",
]

# Chronological splits use feature year t to predict AQI in year t+1.
HISTORICAL_SPLITS = {
    "train": (2014, 2020),
    "validation": (2021, 2022),
    "test": (2023, 2024),
}

EXTENDED_SPLITS = {
    "train": (2019, 2021),
    "validation": (2022, 2022),
    "test": (2023, 2024),
}

RANDOM_STATE = 42

# Minimum number of earlier feature years required before rolling-origin
# evaluation begins. This keeps the first training window reasonably sized.
ROLLING_MIN_TRAIN_YEARS = {
    "Historical core": 5,
    "Extended features": 3,
}


# ---------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------

def load_staging_data(path):
    """Load the expanded one-year-ahead staging dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model staging file: {path}\n"
            "Run: python scripts/prepare_model_data.py"
        )

    df = pd.read_csv(path, dtype={FIPS_COLUMN: "string"}, low_memory=False)
    df.columns = df.columns.str.strip()

    required = {
        FIPS_COLUMN,
        "borough",
        YEAR_COLUMN,
        FORECAST_YEAR_COLUMN,
        TARGET_COLUMN,
        BASELINE_FEATURE,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[FIPS_COLUMN] = (
        df[FIPS_COLUMN]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    for column in (YEAR_COLUMN, FORECAST_YEAR_COLUMN, TARGET_COLUMN):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[[FIPS_COLUMN, YEAR_COLUMN]].isna().any().any():
        raise ValueError("Staging data contains missing borough/year keys.")

    if df[[YEAR_COLUMN, FORECAST_YEAR_COLUMN]].isna().any().any():
        raise ValueError("Staging data contains invalid feature/forecast years.")

    if df[TARGET_COLUMN].isna().any():
        missing_targets = int(df[TARGET_COLUMN].isna().sum())
        warnings.warn(
            f"Dropping {missing_targets} rows without next-year AQI labels. "
            "These rows cannot be used for supervised evaluation."
        )
        df = df.dropna(subset=[TARGET_COLUMN]).copy()

    df[YEAR_COLUMN] = df[YEAR_COLUMN].astype(int)
    df[FORECAST_YEAR_COLUMN] = df[FORECAST_YEAR_COLUMN].astype(int)

    if not (df[FORECAST_YEAR_COLUMN] == df[YEAR_COLUMN] + 1).all():
        raise ValueError(
            "The staging file must use feature year t to predict forecast year t+1."
        )

    if df.duplicated([FIPS_COLUMN, YEAR_COLUMN]).any():
        duplicates = df.loc[
            df.duplicated([FIPS_COLUMN, YEAR_COLUMN], keep=False),
            [FIPS_COLUMN, "borough", YEAR_COLUMN],
        ]
        raise ValueError(
            "Duplicate borough/feature-year rows found:\n"
            f"{duplicates.head(10).to_string(index=False)}"
        )

    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Next-year AQI targets contain non-finite values.")

    return df.sort_values([FIPS_COLUMN, YEAR_COLUMN]).reset_index(drop=True)


def make_splits(df, year_ranges, experiment_name):
    """Build chronological splits from feature-year ranges."""
    split_frames = {}

    for split_name, (first_year, last_year) in year_ranges.items():
        split_frames[split_name] = df[
            df[YEAR_COLUMN].between(first_year, last_year)
        ].copy()

    for split_name, frame in split_frames.items():
        if frame.empty:
            raise ValueError(
                f"{experiment_name}: {split_name} split is empty. "
                f"Configured feature years: {year_ranges[split_name]}"
            )

    # Ensure each split contains only its configured years.
    return split_frames


# ---------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------

def calculate_metrics(actual, predictions):
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    return {
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": rmse,
        "R2": r2_score(actual, predictions) if len(actual) >= 2 else np.nan,
    }


def build_models():
    return {
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


def run_experiment(
    split_frames,
    feature_names,
    experiment_name,
    results,
    predictions_to_save,
):
    """Fit preprocessing/models on training rows and evaluate later years."""

    for split_name, frame in split_frames.items():
        missing = [feature for feature in feature_names if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"{experiment_name} {split_name} data is missing columns: {missing}"
            )

    train = split_frames["train"].copy()
    validation = split_frames["validation"].copy()
    test = split_frames["test"].copy()

    # Convert predictors to numeric; source blanks remain missing.
    for frame in (train, validation, test):
        for feature in feature_names:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    # Do not allow a feature with no training observations to be silently
    # discarded by SimpleImputer. This usually indicates a bad year range.
    empty_training_features = [
        feature for feature in feature_names if train[feature].notna().sum() == 0
    ]
    if empty_training_features:
        raise ValueError(
            f"{experiment_name}: these features have no observed training values: "
            f"{empty_training_features}. Adjust the feature set or training years."
        )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[feature_names])
    X_validation = imputer.transform(validation[feature_names])
    X_test = imputer.transform(test[feature_names])

    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    # No-change baseline: predict next year's AQI using the latest observed AQI.
    for split_name, frame, actual in [
        ("validation", validation, y_validation),
        ("test", test, y_test),
    ]:
        baseline_predictions = pd.to_numeric(
            frame[BASELINE_FEATURE], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(baseline_predictions).all():
            raise ValueError(
                f"{experiment_name}: missing/non-finite baseline AQI "
                f"in the {split_name} split."
            )

        results.append({
            "experiment": experiment_name,
            "model": "No-change baseline",
            "split": split_name,
            "n_rows": len(actual),
            **calculate_metrics(actual, baseline_predictions),
        })

        for i, (_, row) in enumerate(frame.iterrows()):
            predictions_to_save.append({
                "experiment": experiment_name,
                "model": "No-change baseline",
                "split": split_name,
                "county_fips": row[FIPS_COLUMN],
                "borough": row["borough"],
                "feature_year": int(row[YEAR_COLUMN]),
                "target_year": int(row[FORECAST_YEAR_COLUMN]),
                "actual_aqi": actual[i],
                "predicted_aqi": baseline_predictions[i],
                "training_rows": len(train),
                "training_feature_year_min": int(train[YEAR_COLUMN].min()),
                "training_feature_year_max": int(train[YEAR_COLUMN].max()),
            })

    models = build_models()

    for model_name, model in models.items():
        # Validation and test rows are never used to fit the model.
        model.fit(X_train, y_train)

        for split_name, X, actual, frame in [
            ("train", X_train, y_train, train),
            ("validation", X_validation, y_validation, validation),
            ("test", X_test, y_test, test),
        ]:
            predicted = model.predict(X)

            results.append({
                "experiment": experiment_name,
                "model": model_name,
                "split": split_name,
                "n_rows": len(actual),
                **calculate_metrics(actual, predicted),
            })

            for i, (_, row) in enumerate(frame.iterrows()):
                predictions_to_save.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "split": split_name,
                    "county_fips": row[FIPS_COLUMN],
                    "borough": row["borough"],
                    "feature_year": int(row[YEAR_COLUMN]),
                    "target_year": int(row[FORECAST_YEAR_COLUMN]),
                    "actual_aqi": actual[i],
                    "predicted_aqi": predicted[i],
                    "training_rows": len(train),
                    "training_feature_year_min": int(train[YEAR_COLUMN].min()),
                    "training_feature_year_max": int(train[YEAR_COLUMN].max()),
                })

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_path = OUTPUT / (
            f"{experiment_name.lower().replace(' ', '_')}_"
            f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        )
        importance.to_csv(importance_path, index=False)


def calculate_group_metrics(predictions_df, group_columns):
    """Calculate regression metrics for each requested group."""
    rows = []
    for group_values, group in predictions_df.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        actual = group["actual_aqi"].to_numpy(dtype=float)
        predicted = group["predicted_aqi"].to_numpy(dtype=float)
        row = dict(zip(group_columns, group_values))
        row["n_rows"] = len(group)
        row.update(calculate_metrics(actual, predicted))
        rows.append(row)
    return pd.DataFrame(rows)


def build_missingness_report(split_frames, feature_names, experiment_name):
    """Report missing feature values before training-only imputation."""
    rows = []
    for split_name, frame in split_frames.items():
        for feature in feature_names:
            values = pd.to_numeric(frame[feature], errors="coerce")
            missing_count = int(values.isna().sum())
            rows.append({
                "experiment": experiment_name,
                "split": split_name,
                "feature": feature,
                "n_rows": len(frame),
                "missing_count": missing_count,
                "missing_pct": (missing_count / len(frame) * 100) if len(frame) else np.nan,
                "observed_count": int(values.notna().sum()),
            })
    return pd.DataFrame(rows)


def run_rolling_backtest(
    df,
    feature_names,
    experiment_name,
    predictions_to_save,
):
    """Expanding-window one-year-ahead backtest with no future training rows."""
    df = df.sort_values([YEAR_COLUMN, FIPS_COLUMN]).copy()
    feature_years = sorted(df[YEAR_COLUMN].unique().tolist())
    min_train_years = ROLLING_MIN_TRAIN_YEARS[experiment_name]

    if len(feature_years) <= min_train_years:
        warnings.warn(
            f"{experiment_name}: not enough distinct years for rolling backtesting."
        )
        return

    for position in range(min_train_years, len(feature_years)):
        prediction_year = feature_years[position]
        train = df[df[YEAR_COLUMN] < prediction_year].copy()
        prediction_frame = df[df[YEAR_COLUMN] == prediction_year].copy()

        if train.empty or prediction_frame.empty:
            continue

        # Only use features that have at least one observed training value.
        empty_features = [
            feature for feature in feature_names
            if pd.to_numeric(train[feature], errors="coerce").notna().sum() == 0
        ]
        if empty_features:
            warnings.warn(
                f"{experiment_name}, rolling year {prediction_year}: "
                f"skipping features with no training values: {empty_features}"
            )
            continue

        for frame in (train, prediction_frame):
            for feature in feature_names:
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train[feature_names])
        X_prediction = imputer.transform(prediction_frame[feature_names])
        y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
        y_actual = prediction_frame[TARGET_COLUMN].to_numpy(dtype=float)

        baseline = pd.to_numeric(
            prediction_frame[BASELINE_FEATURE], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(baseline).all():
            raise ValueError(
                f"{experiment_name}: invalid baseline value in rolling year "
                f"{prediction_year}."
            )

        model_predictions = {
            "No-change baseline": baseline,
        }
        for model_name, model in build_models().items():
            model.fit(X_train, y_train)
            model_predictions[model_name] = model.predict(X_prediction)

        for model_name, predicted in model_predictions.items():
            for i, (_, row) in enumerate(prediction_frame.iterrows()):
                predictions_to_save.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "split": "rolling_backtest",
                    "county_fips": row[FIPS_COLUMN],
                    "borough": row["borough"],
                    "feature_year": int(row[YEAR_COLUMN]),
                    "target_year": int(row[FORECAST_YEAR_COLUMN]),
                    "actual_aqi": y_actual[i],
                    "predicted_aqi": predicted[i],
                    "training_rows": len(train),
                    "training_feature_year_min": int(train[YEAR_COLUMN].min()),
                    "training_feature_year_max": int(train[YEAR_COLUMN].max()),
                })


def print_split_summary(experiment_name, split_frames):
    print(f"\n{experiment_name} split sizes:")
    for split_name, frame in split_frames.items():
        years = sorted(frame[YEAR_COLUMN].unique().tolist())
        target_years = sorted(frame[FORECAST_YEAR_COLUMN].unique().tolist())
        print(
            f"  {split_name}: {len(frame)} rows, "
            f"feature years {years}, target years {target_years}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    df = load_staging_data(STAGING_FILE)

    print(f"Loaded expanded staging dataset: {STAGING_FILE}")
    print(f"Supervised rows: {len(df)}")
    print(
        f"Feature years: {df[YEAR_COLUMN].min()}-{df[YEAR_COLUMN].max()}; "
        f"target years: {df[FORECAST_YEAR_COLUMN].min()}-"
        f"{df[FORECAST_YEAR_COLUMN].max()}"
    )
    print(f"Boroughs: {sorted(df['borough'].dropna().unique().tolist())}")

    historical_df = df[
        df[YEAR_COLUMN].between(2014, 2024)
    ].copy()
    extended_df = df[
        df[YEAR_COLUMN].between(2019, 2024)
    ].copy()

    historical_splits = make_splits(
        historical_df, HISTORICAL_SPLITS, "Historical core"
    )
    extended_splits = make_splits(
        extended_df, EXTENDED_SPLITS, "Extended features"
    )

    print_split_summary("Historical core (2014-2024)", historical_splits)
    print_split_summary("Extended features (2019-2024)", extended_splits)

    all_results = []
    all_predictions = []
    all_missingness = []

    experiments = [
        ("Historical core", historical_df, historical_splits, HISTORICAL_FEATURES),
        ("Extended features", extended_df, extended_splits, EXTENDED_FEATURES),
    ]

    for experiment_name, experiment_df, split_frames, feature_names in experiments:
        run_experiment(
            split_frames=split_frames,
            feature_names=feature_names,
            experiment_name=experiment_name,
            results=all_results,
            predictions_to_save=all_predictions,
        )
        all_missingness.append(
            build_missingness_report(split_frames, feature_names, experiment_name)
        )
        run_rolling_backtest(
            df=experiment_df,
            feature_names=feature_names,
            experiment_name=experiment_name,
            predictions_to_save=all_predictions,
        )

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)
    missingness_df = pd.concat(all_missingness, ignore_index=True)

    # Compare each model's MAE with the no-change baseline for the same
    # experiment and split. Positive values mean lower MAE than baseline.
    baseline_mae = results_df[
        results_df["model"] == "No-change baseline"
    ][["experiment", "split", "MAE"]].rename(
        columns={"MAE": "baseline_MAE"}
    )
    results_df = results_df.merge(
        baseline_mae, on=["experiment", "split"], how="left"
    )
    results_df["MAE_improvement_vs_baseline_pct"] = (
        (results_df["baseline_MAE"] - results_df["MAE"])
        / results_df["baseline_MAE"] * 100
    )
    results_df.loc[
        results_df["model"] == "No-change baseline",
        "MAE_improvement_vs_baseline_pct",
    ] = 0.0

    # Borough-level metrics for the standard validation/test predictions.
    standard_predictions = predictions_df[
        predictions_df["split"].isin(["validation", "test"])
    ].copy()
    borough_metrics_df = calculate_group_metrics(
        standard_predictions,
        ["experiment", "model", "split", "borough"],
    )

    # Rolling-origin metrics are reported separately from the fixed test split.
    rolling_predictions = predictions_df[
        predictions_df["split"] == "rolling_backtest"
    ].copy()
    rolling_metrics_df = calculate_group_metrics(
        rolling_predictions,
        ["experiment", "model"],
    ) if not rolling_predictions.empty else pd.DataFrame()
    rolling_borough_metrics_df = calculate_group_metrics(
        rolling_predictions,
        ["experiment", "model", "borough"],
    ) if not rolling_predictions.empty else pd.DataFrame()

    results_path = OUTPUT / "model_comparison.csv"
    predictions_path = OUTPUT / "predictions.csv"
    borough_metrics_path = OUTPUT / "borough_metrics.csv"
    rolling_metrics_path = OUTPUT / "rolling_backtest_metrics.csv"
    rolling_borough_path = OUTPUT / "rolling_backtest_borough_metrics.csv"
    missingness_path = OUTPUT / "feature_missingness.csv"
    manifest_path = OUTPUT / "run_manifest.json"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    borough_metrics_df.to_csv(borough_metrics_path, index=False)
    rolling_metrics_df.to_csv(rolling_metrics_path, index=False)
    rolling_borough_metrics_df.to_csv(rolling_borough_path, index=False)
    missingness_df.to_csv(missingness_path, index=False)

    manifest = {
        "purpose": "Exploratory one-year-ahead NYC borough median AQI prediction",
        "source_file": str(STAGING_FILE.relative_to(ROOT)),
        "source_years": {
            "feature_years": [
                int(df[YEAR_COLUMN].min()),
                int(df[YEAR_COLUMN].max()),
            ],
            "target_years": [
                int(df[FORECAST_YEAR_COLUMN].min()),
                int(df[FORECAST_YEAR_COLUMN].max()),
            ],
        },
        "modeling_unit": "NYC borough (county FIPS) x feature year",
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "experiments": {
            "historical_core": {
                "features": HISTORICAL_FEATURES,
                "splits": HISTORICAL_SPLITS,
                "rows": {
                    name: len(frame) for name, frame in historical_splits.items()
                },
            },
            "extended_features": {
                "features": EXTENDED_FEATURES,
                "splits": EXTENDED_SPLITS,
                "rows": {
                    name: len(frame) for name, frame in extended_splits.items()
                },
            },
        },
        "preprocessing": (
            "Median imputation fitted on each fixed-split training set only; "
            "rolling backtests refit the imputer using only rows earlier than "
            "the prediction feature year."
        ),
        "rolling_backtest_minimum_training_years": ROLLING_MIN_TRAIN_YEARS,
        "additional_diagnostics": [
            "Borough-level validation/test metrics.",
            "Feature missingness by experiment, split, and feature before imputation.",
            "Expanding-window rolling-origin predictions and metrics.",
            "MAE improvement percentage relative to the no-change baseline.",
        ],
        "limitations": [
            "Small borough-year sample; observations are correlated across years and boroughs.",
            "Rolling-origin backtests provide more temporal checks but are not independent observations.",
            "Some expanded-source features begin later than 2014; the extended-feature "
            "experiment is restricted to 2019-2024 to reduce structural missingness.",
            "NYC DOT traffic values are sampled 15-minute counts, not annual traffic volume.",
            "Weather values are historical reanalysis point samples, not borough-area measured averages.",
            "Point-in-time availability of every predictor has not been independently certified.",
            "Hyperparameters are exploratory and have not been tuned on the test set.",
            "Results do not establish operational forecasting skill.",
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
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

    print("\nBorough-level validation/test metrics:")
    if not borough_metrics_df.empty:
        print(borough_metrics_df.round(3).to_string(index=False))

    print("\nRolling-origin backtest metrics:")
    if not rolling_metrics_df.empty:
        print(rolling_metrics_df.round(3).to_string(index=False))

    print("\nFeature missingness before imputation:")
    print(missingness_df.round(1).to_string(index=False))

    print(f"\nSaved results to: {results_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved borough metrics to: {borough_metrics_path}")
    print(f"Saved rolling metrics to: {rolling_metrics_path}")
    print(f"Saved rolling borough metrics to: {rolling_borough_path}")
    print(f"Saved feature missingness to: {missingness_path}")
    print(f"Saved run manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
