#!/usr/bin/env python3
"""Review EV source coverage and create SAFE candidate datasets for NYC AQI.

Run from Air-Pollution-main:
    python scripts/review_ev_and_stage_data.py

Requires: pandas (already used by prepare_model_data.py).
This script never edits your raw datasets, your current master, or your current
next-year modeling file. All new CSVs are explicitly marked CANDIDATE/REVIEW.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

NYC = {
    "36005": "Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island",
}
YEARS = range(2019, 2026)
EV_COLUMN = "ev_reported_registrations_count"

# Short feature shortlist for an ANNUAL t -> t+1 demonstration only.
# Not a feature-selection claim; do not train until the team agrees on the target.
BASELINE_FEATURES = [
    "last_observed_median_aqi",
    "recorded_tree_plantings",
    "drive_alone_pct",
    "campd_total_nox_mass_tons",
    "weather_temp_mean_c",
    "weather_precip_total_mm",
]


def normalize_keys(frame: pd.DataFrame, year_col: str, name: str) -> pd.DataFrame:
    """Normalize join keys and reject duplicates/unknown NYC counties."""
    data = frame.copy()
    if not {"county_fips", year_col}.issubset(data.columns):
        raise ValueError(f"{name} is missing county_fips or {year_col}")
    data["county_fips"] = (
        data["county_fips"].astype("string").str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    data[year_col] = pd.to_numeric(data[year_col], errors="raise").astype("int64")
    if not data["county_fips"].isin(NYC).all():
        raise ValueError(f"{name} has records outside the five NYC county FIPS codes")
    if not data[year_col].between(2019, 2025).all():
        raise ValueError(f"{name} has years outside 2019-2025")
    if data.duplicated(["county_fips", year_col]).any():
        raise ValueError(f"{name} contains duplicate county-year rows")
    return data


def save_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def run(root: Path) -> None:
    ev_path = root / "datasets/raw/ev/nyc_ev_registration_CANDIDATE_borough_year_2019_2025.csv"
    master_path = root / "datasets/processed/master_borough_year_2019_2025.csv"
    staging_path = root / "datasets/processed/next_year_aqi_model_STAGING.csv"
    for path in (ev_path, master_path, staging_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the earlier collectors/preparer first.")

    ev_raw = pd.read_csv(ev_path, dtype={"county_fips": "string"})
    ev = normalize_keys(ev_raw, "year", "EV")
    if EV_COLUMN not in ev.columns:
        raise ValueError(f"EV source missing {EV_COLUMN}")
    if ev[EV_COLUMN].isna().any():
        raise ValueError("EV candidate has missing counts in the observed rows")
    # Ensure every observed count is a nonnegative whole number.
    counts = pd.to_numeric(ev[EV_COLUMN], errors="raise")
    if not counts.ge(0).all() or not counts.mod(1).eq(0).all():
        raise ValueError("EV candidate contains negative or fractional registration counts")
    ev[EV_COLUMN] = counts.astype("int64")
    ev = ev[["county_fips", "year", EV_COLUMN]].copy()

    master = normalize_keys(pd.read_csv(master_path, dtype={"county_fips": "string"}), "year", "master")
    expected = {(fips, year) for fips in NYC for year in YEARS}
    if set(zip(master.county_fips, master.year)) != expected:
        raise ValueError("Master is not a complete five-borough x 2019-2025 grid")
    if EV_COLUMN in master.columns:
        raise ValueError("Master already contains the EV feature; inspect before rerunning this script")
    if "borough" not in master or any(master["borough"] != master["county_fips"].map(NYC)):
        raise ValueError("Master borough names and county FIPS do not match")

    actual_ev_keys = set(zip(ev.county_fips, ev.year))
    if not actual_ev_keys.issubset(expected):
        raise ValueError("EV observation outside the master year/borough grid")

    # IMPORTANT: These are a *candidate research join*. Original master remains unchanged.
    candidate = master.merge(ev, on=["county_fips", "year"], how="left", validate="one_to_one")
    candidate["ev_record_observed"] = candidate[EV_COLUMN].notna()

    # Re-check coverage, without filling missing years with zeros.
    coverage = candidate[["county_fips", "borough", "year", "ev_record_observed", EV_COLUMN]].copy()
    grouped = coverage.groupby("year", as_index=False).agg(
        boroughs_observed=("ev_record_observed", "sum"),
        boroughs_missing=("ev_record_observed", lambda s: (~s).sum()),
    )
    group_rows = [
        {"year": int(row.year), "boroughs_observed": int(row.boroughs_observed),
         "boroughs_missing": int(row.boroughs_missing)}
        for row in grouped.itertuples(index=False)
    ]

    staging = normalize_keys(
        pd.read_csv(staging_path, dtype={"county_fips": "string"}),
        "feature_year", "next-year staging",
    )
    if "target_next_year_median_aqi" not in staging.columns:
        raise ValueError("Next-year staging file is missing target_next_year_median_aqi")
    if EV_COLUMN in staging.columns:
        raise ValueError("Next-year staging already contains EV; inspect before rerunning")
    candidate_forecast = staging.merge(
        ev.rename(columns={"year": "feature_year"}),
        on=["county_fips", "feature_year"], how="left", validate="one_to_one",
    )
    candidate_forecast["ev_record_observed"] = candidate_forecast[EV_COLUMN].notna()

    # A narrow starting table for a future *ANNUAL* next-year experiment.
    # It deliberately does NOT require EV, LUCID, tree removals, or sampled
    # road traffic. Keeping full forecast_year and feature_year columns makes
    # chronological splitting possible when the team starts model training.
    missing_columns = [c for c in BASELINE_FEATURES if c not in staging.columns]
    if missing_columns:
        raise ValueError(
            "Current staging file is missing expected baseline columns: "
            + ", ".join(missing_columns)
            + ". Check that the historical ACS and weather updates were integrated."
        )
    baseline_cols = ["county_fips", "borough", "feature_year", "forecast_year",
                     "target_next_year_median_aqi", *BASELINE_FEATURES]
    baseline = staging[baseline_cols].copy()
    for col in ["target_next_year_median_aqi", *BASELINE_FEATURES]:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    # Restrict to full recent feature years where CAMPD and ACS were obtained.
    baseline = baseline[baseline["feature_year"].between(2020, 2024)].copy()
    missing_baseline = {
        col: int(baseline[col].isna().sum())
        for col in ["target_next_year_median_aqi", *BASELINE_FEATURES]
    }
    # Do NOT impute here. Emit complete cases as a clearly marked review dataset.
    complete = baseline.dropna(subset=["target_next_year_median_aqi", *BASELINE_FEATURES])
    complete = complete.sort_values(["forecast_year", "county_fips"])

    out = root / "datasets" / "processed" / "review"
    save_csv(coverage, out / "ev_source_coverage_review.csv")
    save_csv(candidate.sort_values(["county_fips", "year"]), out / "master_with_ev_CANDIDATE.csv")
    save_csv(candidate_forecast.sort_values(["county_fips", "feature_year"]),
             out / "next_year_with_ev_CANDIDATE.csv")
    save_csv(complete, out / "annual_next_year_baseline_COMPLETE_CASES_REVIEW_ONLY.csv")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "master_rows_unchanged": len(master),
        "ev_source_present_rows": int(coverage["ev_record_observed"].sum()),
        "ev_source_missing_rows": int((~coverage["ev_record_observed"]).sum()),
        "ev_coverage_by_year": group_rows,
        "annual_next_year_baseline_rows": int(len(complete)),
        "baseline_missing_by_feature_before_dropping_rows": missing_baseline,
        "baseline_features": BASELINE_FEATURES,
        "important_notes": [
            "These outputs are for REVIEW, not an automatically approved final training set.",
            "The original master CSV and next_year_aqi_model_STAGING.csv were not overwritten.",
            "EV data are compiled research registration counts; verify definitions before model use.",
            "EV does not provide a total-vehicle denominator; these counts are NOT an EV adoption percentage.",
            "Unavailable EV years are blank, never zero, and are never forward-filled.",
            "Avoid adding current-only official NY DMV counts as if they were historical 2024 or 2025 counts.",
            "Annual next-year demonstration uses feature year t to predict AQI median at t+1.",
            "Train/validation/test splits must be chronological; fit imputers and scalers only on train.",
            "Only 5 boroughs with a few years of labels; predictive metrics will be unstable.",
            "Census ACS five-year features and some environmental datasets have release lags: audit point-in-time availability for operational forecasting.",
        ],
        "outputs": [
            "review/ev_source_coverage_review.csv",
            "review/master_with_ev_CANDIDATE.csv",
            "review/next_year_with_ev_CANDIDATE.csv",
            "review/annual_next_year_baseline_COMPLETE_CASES_REVIEW_ONLY.csv",
        ],
    }
    (out / "ev_and_baseline_review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEV candidate review finished; the original master and staging files were NOT changed.")
    print(f"EV observed: {report['ev_source_present_rows']}/35; missing: {report['ev_source_missing_rows']}/35")
    for row in group_rows:
        print(f"  {row['year']}: {row['boroughs_observed']}/5 EV records")
    print(f"Annual next-year baseline complete cases: {len(complete)}")
    print(f"Review folder: {out}")
    print("Share ev_and_baseline_review_report.json and any warnings before modeling.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EV coverage; stage safe research and annual next-year review CSVs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root containing datasets/")
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
