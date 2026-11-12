from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "datasets" / "processed" / "model_results"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.csv"

# Evaluate validation, test, and rolling-origin predictions.
# Training predictions are excluded because they are in-sample results.
EVALUATION_SPLITS = ["validation", "test", "rolling_backtest"]

REQUIRED_COLUMNS = {
    "experiment",
    "model",
    "split",
    "borough",
    "target_year",
    "actual_aqi",
    "predicted_aqi",
}


# ---------------------------------------------------------
# Metrics and reporting helpers
# ---------------------------------------------------------

def calculate_metrics(group):
    actual = group["actual_aqi"].to_numpy(dtype=float)
    predicted = group["predicted_aqi"].to_numpy(dtype=float)

    return pd.Series(
        {
            "n_rows": len(group),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            # R2 is undefined for a group with fewer than two observations.
            "R2": r2_score(actual, predicted) if len(group) >= 2 else np.nan,
        }
    )


def add_baseline_comparison(metrics_df, grouping_columns):
    """Add MAE improvement relative to the matching no-change baseline."""
    if metrics_df.empty:
        metrics_df["baseline_MAE"] = np.nan
        metrics_df["MAE_improvement_vs_baseline_pct"] = np.nan
        return metrics_df

    baseline_group_columns = [
        column for column in grouping_columns if column != "model"
    ]
    baseline = metrics_df[
        metrics_df["model"] == "No-change baseline"
    ][baseline_group_columns + ["MAE"]].rename(
        columns={"MAE": "baseline_MAE"}
    )

    result = metrics_df.merge(
        baseline,
        on=baseline_group_columns,
        how="left",
        validate="many_to_one",
    )

    result["MAE_improvement_vs_baseline_pct"] = np.where(
        result["baseline_MAE"].notna() & (result["baseline_MAE"] != 0),
        (result["baseline_MAE"] - result["MAE"])
        / result["baseline_MAE"]
        * 100,
        np.nan,
    )
    result.loc[
        result["model"] == "No-change baseline",
        "MAE_improvement_vs_baseline_pct",
    ] = 0.0

    return result


def build_metrics(predictions, grouping_columns):
    if predictions.empty:
        return pd.DataFrame()

    metrics = (
        predictions.groupby(grouping_columns, dropna=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )
    return add_baseline_comparison(metrics, grouping_columns)


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find predictions file: {PREDICTIONS_FILE}\n"
            "Run train_aqi_models.py first."
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "The predictions file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep only out-of-sample validation, test, and rolling predictions.
    predictions = predictions[
        predictions["split"].isin(EVALUATION_SPLITS)
    ].copy()

    if predictions.empty:
        raise ValueError(
            "No validation, test, or rolling_backtest predictions were found."
        )

    predictions["actual_aqi"] = pd.to_numeric(
        predictions["actual_aqi"], errors="coerce"
    )
    predictions["predicted_aqi"] = pd.to_numeric(
        predictions["predicted_aqi"], errors="coerce"
    )
    predictions["target_year"] = pd.to_numeric(
        predictions["target_year"], errors="coerce"
    )

    invalid_rows = predictions[
        predictions[["actual_aqi", "predicted_aqi", "target_year"]]
        .isna()
        .any(axis=1)
    ]
    if not invalid_rows.empty:
        raise ValueError(
            f"Found {len(invalid_rows)} prediction rows with missing or "
            "non-numeric AQI values or target years."
        )

    if not np.isfinite(
        predictions[["actual_aqi", "predicted_aqi"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("The predictions file contains non-finite AQI values.")

    predictions["target_year"] = predictions["target_year"].astype(int)
    predictions["signed_error"] = (
        predictions["predicted_aqi"] - predictions["actual_aqi"]
    )
    predictions["absolute_error"] = predictions["signed_error"].abs()

    # Overall metrics, separated by experiment and evaluation split.
    summary = build_metrics(
        predictions,
        ["experiment", "split", "model"],
    )

    # Borough-level metrics. R2 is NaN when a borough has only one row.
    by_borough = build_metrics(
        predictions,
        ["experiment", "split", "borough", "model"],
    )

    # Target-year metrics show whether performance changes over time.
    by_year = build_metrics(
        predictions,
        ["experiment", "split", "target_year", "model"],
    )

    # Save detailed out-of-sample predictions with signed and absolute errors.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rolling_summary = summary[
        summary["split"] == "rolling_backtest"
    ].copy()
    output_files = {
        "summary": RESULTS_DIR / "evaluation_summary.csv",
        "borough": RESULTS_DIR / "evaluation_by_borough.csv",
        "year": RESULTS_DIR / "evaluation_by_year.csv",
        "rolling": RESULTS_DIR / "evaluation_rolling_backtest.csv",
        "predictions": RESULTS_DIR / "evaluation_predictions.csv",
    }

    summary.to_csv(output_files["summary"], index=False)
    by_borough.to_csv(output_files["borough"], index=False)
    by_year.to_csv(output_files["year"], index=False)
    rolling_summary.to_csv(output_files["rolling"], index=False)
    predictions.to_csv(output_files["predictions"], index=False)

    # Console summaries.
    display_columns = [
        "experiment",
        "model",
        "split",
        "n_rows",
        "MAE",
        "RMSE",
        "R2",
        "MAE_improvement_vs_baseline_pct",
    ]

    print("\nOverall evaluation results:")
    print(
        summary[display_columns]
        .sort_values(["experiment", "split", "MAE"])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    print("\nBorough-level evaluation (MAE and RMSE are most useful here):")
    borough_columns = [
        "experiment",
        "model",
        "split",
        "borough",
        "n_rows",
        "MAE",
        "RMSE",
        "R2",
        "MAE_improvement_vs_baseline_pct",
    ]
    print(
        by_borough[borough_columns]
        .sort_values(["experiment", "split", "borough", "MAE"])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    print("\nYear-level evaluation:")
    year_columns = [
        "experiment",
        "model",
        "split",
        "target_year",
        "n_rows",
        "MAE",
        "RMSE",
        "R2",
        "MAE_improvement_vs_baseline_pct",
    ]
    print(
        by_year[year_columns]
        .sort_values(["experiment", "split", "target_year", "MAE"])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    print("\nSaved evaluation files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
